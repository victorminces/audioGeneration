"""DDSP: differentiable harmonic-plus-noise synthesis with a timbre latent.

Analysis:  waveform -> (f0, loudness, z)   per 16 ms frame
Synthesis: (f0, loudness, z) -> waveform   (oscillator bank + filtered noise)

The decoder is trained end-to-end through the synthesizer with a multi-scale
spectral loss (Engel et al., ICLR 2020). Output is waveform directly — no
phase reconstruction needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchcrepe

SR          = 16000
HOP         = 256            # 16 ms frames, matches the rest of the project
N_HARMONICS = 60
N_NOISE     = 65             # noise filter bands
Z_DIM       = 16
N_MELS      = 80

_mel = T.MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=HOP, n_mels=N_MELS)


# ── analysis ──────────────────────────────────────────────────────────────────

def extract_f0(wav, n_frames):
    """(1, samples) -> (n_frames,) f0 in Hz (unvoiced frames get held/low values;
    the decoder learns to silence harmonics there via the amplitude head)."""
    # weighted_argmax decoder is pure torch (the default viterbi decoder pulls
    # in librosa -> numba, which breaks against numpy >= 2.2)
    f0 = torchcrepe.predict(wav, SR, hop_length=HOP, fmin=60, fmax=600,
                            model="tiny", batch_size=512, device="cpu",
                            decoder=torchcrepe.decode.weighted_argmax)[0]
    return F.interpolate(f0.view(1, 1, -1), size=n_frames,
                         mode="linear", align_corners=True)[0, 0]


def extract_loudness(wav, n_frames):
    """(1, samples) -> (n_frames,) log-RMS loudness."""
    frames = F.pad(wav, (0, (-wav.shape[1]) % HOP)).reshape(1, -1, HOP)
    rms = frames.pow(2).mean(dim=2).sqrt()[0]
    loud = (rms + 1e-5).log()
    return F.interpolate(loud.view(1, 1, -1), size=n_frames,
                         mode="linear", align_corners=True)[0, 0]


def mel_frames(wav, n_frames):
    """(1, samples) -> (n_frames, N_MELS) log-mel for the timbre encoder."""
    m = (_mel(wav) + 1e-6).log()[0].T                     # (T', N_MELS)
    return F.interpolate(m.T.unsqueeze(0), size=n_frames,
                         mode="linear", align_corners=True)[0].T


def scale_f0(f0):
    return (f0.clamp(min=1).log2() - 5.9) / 3.3           # ~[60, 600] Hz -> ~[0, 1]


def scale_loud(loud):
    return (loud + 8.0) / 8.0                             # log-rms ~[-8, 0] -> [0, 1]


# ── model ─────────────────────────────────────────────────────────────────────

def _exp_sigmoid(x):
    """DDSP paper's output activation: positive, saturating, ~exp scale."""
    return 2.0 * torch.sigmoid(x) ** 2.3 + 1e-7


class TimbreEncoder(nn.Module):
    """log-mel frames -> per-frame timbre code z."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(N_MELS, 128, batch_first=True, bidirectional=False)
        self.out = nn.Linear(128, Z_DIM)

    def forward(self, mels):                              # (B, T, N_MELS)
        h, _ = self.gru(mels)
        return self.out(h)                                # (B, T, Z_DIM)


class Decoder(nn.Module):
    """(f0, loudness, z) frames -> harmonic amps + noise band gains."""
    def __init__(self):
        super().__init__()
        in_dim = 2 + Z_DIM
        self.mlp_in = nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.LeakyReLU(),
                                    nn.Linear(256, 256), nn.LayerNorm(256), nn.LeakyReLU())
        self.gru = nn.GRU(256, 256, batch_first=True)
        self.mlp_out = nn.Sequential(nn.Linear(256 + in_dim, 256), nn.LayerNorm(256), nn.LeakyReLU())
        self.head_amp   = nn.Linear(256, 1)
        self.head_harm  = nn.Linear(256, N_HARMONICS)
        self.head_noise = nn.Linear(256, N_NOISE)

    def forward(self, f0_s, loud_s, z):                   # each (B, T[, .])
        x = torch.cat([f0_s.unsqueeze(-1), loud_s.unsqueeze(-1), z], dim=-1)
        h = self.mlp_in(x)
        h, _ = self.gru(h)
        h = self.mlp_out(torch.cat([h, x], dim=-1))
        amp   = _exp_sigmoid(self.head_amp(h))            # (B, T, 1)
        harm  = _exp_sigmoid(self.head_harm(h))           # (B, T, H)
        harm  = harm / (harm.sum(dim=-1, keepdim=True) + 1e-7)
        noise = _exp_sigmoid(self.head_noise(h))          # (B, T, N_NOISE)
        return amp, harm, noise


class DDSP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TimbreEncoder()
        self.decoder = Decoder()

    # -- synthesis ------------------------------------------------------------
    @staticmethod
    def _upsample(x, n_samples):
        """(B, T, C) frame signal -> (B, n_samples, C) linear interp."""
        return F.interpolate(x.transpose(1, 2), size=n_samples, mode="linear",
                             align_corners=True).transpose(1, 2)

    def synthesize(self, f0, amp, harm, noise):
        """f0 (B,T) Hz; amp (B,T,1); harm (B,T,H); noise (B,T,N) -> (B, samples)."""
        B, Tf = f0.shape
        n = Tf * HOP
        f0_up   = self._upsample(f0.unsqueeze(-1), n)[..., 0]        # (B, n)
        amp_up  = self._upsample(amp, n)[..., 0]
        harm_up = self._upsample(harm, n)                            # (B, n, H)
        phase = 2 * torch.pi * torch.cumsum(f0_up / SR, dim=1)       # (B, n)
        k = torch.arange(1, N_HARMONICS + 1, device=f0.device).float()
        # zero harmonics above Nyquist
        alias = (f0_up.unsqueeze(-1) * k) < (SR / 2)                 # (B, n, H)
        harmonics = torch.sin(phase.unsqueeze(-1) * k) * harm_up * alias
        audio_h = amp_up * harmonics.sum(dim=-1)                     # (B, n)

        # filtered noise: white noise STFT x upsampled band magnitudes
        n_fft = 1024
        bins = n_fft // 2 + 1
        wn = torch.rand(B, n, device=f0.device) * 2 - 1
        win = torch.hann_window(n_fft, device=f0.device)
        S = torch.stft(wn, n_fft=n_fft, hop_length=HOP, window=win,
                       return_complex=True)                          # (B, bins, T')
        filt = F.interpolate(noise.transpose(1, 2), size=S.shape[2],
                             mode="linear", align_corners=True)      # (B, N_NOISE, T')
        filt = F.interpolate(filt.transpose(1, 2).reshape(-1, 1, N_NOISE),
                             size=bins, mode="linear", align_corners=True)
        filt = filt.reshape(B, S.shape[2], bins).transpose(1, 2)     # (B, bins, T')
        audio_n = torch.istft(S * filt, n_fft=n_fft, hop_length=HOP,
                              window=win, length=n)
        return audio_h + audio_n

    # -- end to end -----------------------------------------------------------
    def forward(self, mels, f0, loud):
        """mels (B,T,80); f0 (B,T) Hz; loud (B,T) log-rms -> (B, samples)."""
        z = self.encoder(mels)
        amp, harm, noise = self.decoder(scale_f0(f0), scale_loud(loud), z)
        return self.synthesize(f0, amp, harm, noise)


# ── loss ──────────────────────────────────────────────────────────────────────

_FFT_SIZES = (2048, 1024, 512, 256, 128, 64)

def multiscale_spectral_loss(x, y):
    """L1 on magnitude + log magnitude across several STFT resolutions."""
    loss = 0.0
    for n_fft in _FFT_SIZES:
        win = torch.hann_window(n_fft, device=x.device)
        Sx = torch.stft(x, n_fft=n_fft, hop_length=n_fft // 4, window=win,
                        return_complex=True).abs()
        Sy = torch.stft(y, n_fft=n_fft, hop_length=n_fft // 4, window=win,
                        return_complex=True).abs()
        loss = loss + F.l1_loss(Sx, Sy) + F.l1_loss((Sx + 1e-5).log(), (Sy + 1e-5).log())
    return loss / len(_FFT_SIZES)


# ── high-level analyze / morph / synthesize ──────────────────────────────────

def analyze(model, wav):
    """(1, samples) -> dict of per-frame params (f0 Hz, loud, z)."""
    n_frames = wav.shape[1] // HOP
    f0   = extract_f0(wav, n_frames)
    loud = extract_loudness(wav, n_frames)
    mels = mel_frames(wav, n_frames)
    with torch.no_grad():
        z = model.encoder(mels.unsqueeze(0))[0]           # (T, Z_DIM)
    return {"f0": f0, "loud": loud, "z": z}


def blend_params(params_list, weights):
    """IDW blend of analyzed parameter dicts. f0 blends in log space
    (linear in pitch/cents -> true glissandi); loudness and z linearly."""
    T_min = min(p["f0"].shape[0] for p in params_list)
    logf0 = sum(w * p["f0"][:T_min].clamp(min=1).log() for w, p in zip(weights, params_list))
    loud  = sum(w * p["loud"][:T_min] for w, p in zip(weights, params_list))
    z     = sum(w * p["z"][:T_min] for w, p in zip(weights, params_list))
    return {"f0": logf0.exp(), "loud": loud, "z": z}


def synthesize_params(model, params):
    """Parameter dict -> (1, samples) waveform."""
    with torch.no_grad():
        amp, harm, noise = model.decoder(scale_f0(params["f0"]).unsqueeze(0),
                                         scale_loud(params["loud"]).unsqueeze(0),
                                         params["z"].unsqueeze(0))
        wav = model.synthesize(params["f0"].unsqueeze(0), amp, harm, noise)
    return wav
