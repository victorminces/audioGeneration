"""DDSP: differentiable harmonic-plus-noise synthesis with a timbre latent.

Analysis:  waveform -> (f0, relative level, z)   per 16 ms frame
Synthesis: (f0, relative level, z) -> waveform   (oscillator bank + filtered noise)

The decoder is trained end-to-end through the synthesizer with a multi-scale
spectral loss (Engel et al., ICLR 2020). Output is waveform directly — no
phase reconstruction needed.

Level is factored out of the network. The encoder sees log-mel with the take's
own level subtracted, the decoder takes no loudness input and predicts no
amplitude, and the performed level re-enters only as a multiply in
synthesize(). So moving away from the microphone, or changing its gain, can
change how loud the instrument is but never what it sounds like — by
construction, rather than by every call site remembering to normalize.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

SR          = 16000
HOP         = 256            # 16 ms frames, matches the rest of the project
N_HARMONICS = 60
N_NOISE     = 65             # noise filter bands
Z_DIM       = 16
N_MELS      = 80

# Harmonic amplitudes come from a spectral envelope on absolute frequency,
# sampled at k*f0 — the source-filter model. Formants stay where they are as
# the pitch moves, instead of the network relearning them at every f0.
#
# The alternative, predicting all 60 amplitudes directly indexed by harmonic
# number, was built and compared on identical demonstrations. Trained to the
# same budget the two landed 0.6% apart on validation loss and were
# indistinguishable by ear, differing only in harmonics 40 dB down. This one
# was kept because it is the one that scales: its head size is independent of
# how many harmonics are rendered, so a higher sample rate costs nothing here,
# where the direct version would need 400 outputs to fill 24 kHz and waste 376
# of them on a high note. Inharmonic partials, for bells and struck strings,
# also read off the same curve — there is no slot for them in a head indexed by
# harmonic number.
#
# Envelope bands are log-spaced. 65 bands from 40 Hz to Nyquist is ~8.6 per
# octave everywhere; a linear grid over the same range would put 123 Hz between
# neighbours and barely resolve the first two harmonics of a low voice.
N_ENV       = 65
ENV_F_MIN   = 40.0           # a little below the lowest f0 the analyser reports

# Two different ranges, easily confused.
#
# DETECT is what the analyser can measure. pYIN is autocorrelation-based, so
# this is a search range and not an architectural limit. CREPE, which this used
# to use, is a classifier over fixed bins ending at 2093 Hz, and that ceiling
# was a real problem: a whistle at 2041 Hz sits on its last few bins and reads
# 1939, audible as a warbling electronic tone.
#
# That ceiling is a known, unfixed limitation, not an oversight. A real whistle
# at 2041 Hz sits on the last few bins and reads 1939, with only 30 dB of
# spectral support and a whole octave wrong on 29% of frames — audible as a
# warbling, electronic-sounding tone. Four alternatives were measured against
# the spectrum itself, and every one trades the ceiling for something worse:
#
#   pYIN            right on the clean whistle (2058 Hz, 70 dB support) but its
#                   voicing gate refuses noisy frames and rails to the search
#                   bounds - 3998 Hz for a 220 Hz tone under half noise.
#   YIN (no gate)   collapses to ~150 Hz on the same noisy material.
#   half-speed      shifts CREPE's window up an octave and fixes the whistle
#                   (2050 Hz), but its 1024-sample window then spans 64 ms of
#                   real audio instead of 128, and it loses the pitch under
#                   noise: 112 Hz for a 220 Hz tone at half noise.
#   lower ceiling   forces the whistle to a subharmonic, which is worse.
#
# CREPE at normal speed is the only one robust across noise levels, so it
# stays. Whistling above ~1900 Hz is the price, and it is a real one.
#
# SCALE is how a trained decoder reads its f0 input, and is baked into the
# weights — a checkpoint carries the one it was trained with. It stays at
# 60-600 because that is where the resolution should be: it gives 0.30 of
# input per octave rather than 0.20, and pitches above it simply scale past
# 1.0, which the decoder extrapolates through cleanly (measured: 1500 Hz in,
# 1500 Hz out, at the same level as an in-range pitch). Widening SCALE to
# match DETECT would only spend resolution to avoid an extrapolation that
# already works.
F0_DETECT_MIN = 60.0
F0_DETECT_MAX = 4000.0        # a search bound now, not a model limit
F0_SCALE_MIN = 60.0
F0_SCALE_MAX = 600.0

# Half-width of the harmonic windows used to measure aperiodicity, as a
# fraction of f0. Constant on purpose; see extract_aperiodicity.
AP_WIDTH = 0.15

# Written into every checkpoint's config and checked when one is loaded.
# Decoder input layouts, all incompatible with each other:
#   "ddsp"              [f0, loudness, z16]      ddsp_001..005
#   "ddsp-factored-v1"  [f0, z16]                ddsp_006..007
#   "ddsp-factored-v2"  [f0, aperiodicity, z16]  ddsp_008: CREPE pitch
#   "ddsp-factored-v3"  [f0, aperiodicity, z16]  current: pYIN pitch
# v1 is 17 wide; the rest are 18. v0 and v2 differ in what column 1 means, and
# v2 and v3 are identical in shape while differing in how the pitch was
# measured — the two trackers disagree by an octave on a whistle, so a model
# trained against one reads the other wrongly. No shape check catches either
# case, which is why the name is what gates loading.
ARCH_V1 = "ddsp-factored-v1"
ARCH_V2 = "ddsp-factored-v2"
ARCH = "ddsp-factored-v3"

# What "level" means on the way out. Analysis subtracts each take's performed
# level and stores relative dynamics; this puts them back on a fixed absolute
# footing at synthesis time. -2.0 is the median performed level of the
# training set (measured over 2419 clips: p25 -2.70, p50 -2.29, p75 -1.31).
REF_LOUD = -2.0

# The RMS a unit-gain filter produces through the noise branch. Once the gains
# are normalized per FFT bin (see synthesize) this is a property of the
# transform — window, hop, overlap-add — and not of the band shape. The two
# branches are mixed by weights that assume both arrive at unit level, so it
# has to be divided out. Measured at 0.576 for both the torch stft/istft path
# here and worklet.js's FFT/overlap-add path.
#
# Not exact: end to end the noise branch lands about 2% low through the batch
# synthesizer and 4% low through the streaming one, the residue of windowing
# and overlap-add that a single scalar cannot track. Under half a dB, against a
# harmonic branch that is exact, so the balance the mix head learned survives.
NOISE_UNIT_RMS = 0.576

# How far below a take's own peak the analysis floor sits, in amplitude (80 dB).
#
# The floor has to be relative. An absolute one — the `+1e-6` this module used
# to add before taking a log — is itself an assumption about input scale: it
# clamps a quiet recording's silences and not a loud one's, so the log stops
# shifting uniformly with gain and the invariance below quietly stops holding.
# Measured with an absolute floor, z moved by 0.42 and relative loudness by 1.5
# across a 512x gain sweep; with a relative floor both are exact to 1e-6.
#
# 80 dB is past the usable range of any real recording, so this only engages on
# digital silence. Like mel_reference(), it makes analysis non-causal: the
# floor depends on the whole take.
_DYN_FLOOR = 1e-4

# Everything below the lowest pitch we can detect is rejected before analysis.
#
# Not converter rumble and not the room: measured on real takes, the sub-60 Hz
# level sits at a constant floor during silence and rises 6-15 dB *while you
# perform* — breath hitting the capsule, and the microphone moving. It is
# inaudible (nothing reproduces 20 Hz, and ears roll off below that) so it
# survives every listening test, while arithmetic sees it plainly: it was 51%
# of the energy in a take recorded two steps further away, which made a voiced
# vowel measure 89% noise.
#
# Being linear, this cannot disturb gain invariance: filtering g*x gives
# g*filter(x) exactly.
HP_CUTOFF = F0_DETECT_MIN               # by construction: below this is not signal
_HP_TAPS = 1025                         # 64 ms, ~40 dB down an octave below cutoff
_hp_kernel = None


def highpass(wav, fc=HP_CUTOFF, n=_HP_TAPS, chunk=128):
    """(..., samples) -> same shape, linear phase so nothing is time-shifted.

    Shape-general so the training cache can be filtered in one call, and
    chunked because filtering 2419 clips at once wants more memory than this
    machine has spare.
    """
    global _hp_kernel
    if _hp_kernel is None or _hp_kernel.shape[-1] != n:
        k = np.arange(n) - (n - 1) // 2
        lp = np.sinc(2 * fc / SR * k) * (2 * fc / SR) * np.hamming(n)
        lp = lp / lp.sum()
        hp = -lp
        hp[(n - 1) // 2] += 1.0         # spectral inversion of the low-pass
        _hp_kernel = torch.tensor(hp, dtype=torch.float32).view(1, 1, -1)
    pad = n // 2
    if wav.shape[-1] <= pad:            # too short to reflect: leave it alone
        return wav
    x = wav.reshape(-1, 1, wav.shape[-1])
    ker = _hp_kernel.to(wav.dtype)
    out = torch.empty_like(x)
    for i in range(0, x.shape[0], chunk):
        blk = F.pad(x[i:i + chunk], (pad, pad), mode="reflect")
        out[i:i + chunk] = F.conv1d(blk, ker)
    return out.reshape(wav.shape)


_mel = T.MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=HOP, n_mels=N_MELS)


# ── analysis ──────────────────────────────────────────────────────────────────

def extract_f0(wav, n_frames, f0_min=F0_DETECT_MIN, f0_max=F0_DETECT_MAX):
    """(1, samples) -> (n_frames,) f0 in Hz, a value on every frame.

    pYIN (Mauch & Dixon 2014): YIN's difference function with an HMM over pitch
    candidates. Chosen over CREPE by measuring both against the spectrum rather
    than against each other — for a claimed pitch, how far it stands above the
    frame's median spectral level, and whether more energy sits an octave above:

                          support   unsupported   octave errors
          CREPE   1939 Hz    30 dB        12%          29%
          pYIN    2058 Hz    70 dB         0%           0%

    on a real whistle whose true pitch is 2041 Hz by direct spectral
    measurement. On the other real takes the two agree: 928 against 932 for a
    low whistle, 132 against 133 for a vowel.

    pYIN also declines to answer on frames it judges unvoiced. On constructed
    tone-plus-noise mixtures at half noise that makes it rail to the search
    bounds, where CREPE holds the pitch — the one place CREPE is better. It is
    accepted because voicing is now measured separately: a frame with no pitch
    gets aperiodicity near 1, the harmonic branch is silenced there, and what
    the tracker said stops mattering. The failure that was audible was CREPE's.

    Unvoiced gaps are bridged by interpolation rather than zeroed. The
    synthesiser needs a pitch every frame, and the value is averaged into the
    map, so a smooth line through a gap beats a jump to nothing and back.
    """
    import librosa                                   # heavy; imported on use
    x = wav[0].detach().cpu().numpy().astype(np.float64)
    f0, _, _ = librosa.pyin(x, fmin=f0_min, fmax=f0_max, sr=SR,
                            frame_length=2048, hop_length=HOP, fill_na=np.nan)
    ok = np.isfinite(f0)
    if ok.any():
        idx = np.arange(len(f0))
        f0 = np.interp(idx, idx[ok], f0[ok])
    else:
        f0 = np.full(len(f0), f0_min)
    f0 = torch.tensor(f0, dtype=torch.float32)
    return F.interpolate(f0.view(1, 1, -1), size=n_frames,
                         mode="linear", align_corners=True)[0, 0]


def extract_loudness(wav, n_frames):
    """(1, samples) -> (n_frames,) log-RMS loudness, floored relative to peak."""
    frames = F.pad(wav, (0, (-wav.shape[1]) % HOP)).reshape(1, -1, HOP)
    rms = frames.pow(2).mean(dim=2).sqrt()[0]
    floor = torch.clamp(rms.max() * _DYN_FLOOR, min=1e-12)
    loud = rms.clamp(min=floor).log()
    return F.interpolate(loud.view(1, 1, -1), size=n_frames,
                         mode="linear", align_corners=True)[0, 0]


def mel_frames(wav, n_frames):
    """(1, samples) -> (n_frames, N_MELS) log-mel for the timbre encoder."""
    p = _mel(wav)                                         # power, not amplitude
    floor = torch.clamp(p.max() * _DYN_FLOOR ** 2, min=1e-20)
    m = p.clamp(min=floor).log()[0].T                     # (T', N_MELS)
    return F.interpolate(m.T.unsqueeze(0), size=n_frames,
                         mode="linear", align_corners=True)[0].T


def extract_aperiodicity(wav, f0, n_frames, n_fft=2048, width=AP_WIDTH):
    """(1, samples) + f0 -> (n_frames,) noise fraction per frame, in [0, 1].

    Measured, not learned, for the same reason f0 is. The decoder used to
    predict this balance and was demonstrably bad at it: on a whispered vowel
    its head answered 0.078 where the truth is 1.0, rebuilding a whisper as a
    pitched voice, because it had learned to read voicing from which
    frequencies carry energy — true across read speech and false the moment you
    whisper. Handing it in also makes the balance interpolate as a straight line
    between demonstrations rather than stepping off a cliff.

    The question asked is "is the energy sitting on a harmonic grid?": with f0
    known, harmonics belong at k*f0, and whatever falls between those positions
    is noise.

    Boersma's harmonics-to-noise ratio was tried instead, as the thirty-year
    standard, and rejected on measurement. It asks a different question — "does
    the waveform repeat at this period?" — which is undefined when there is no
    period, and Praat accordingly reports it for voiced frames only. We need a
    number on every frame. On a real whisper, where the tracker returns an
    arbitrary 832 Hz and so a very short lag, short-lag autocorrelation of noise
    is not small and HNR read 0.530 where the truth is ~1; it could not exceed
    0.891 even on pure noise. This measure reads the same whisper at 0.888.
    HNR was the more accurate of the two on clean mixtures (0.014 mean absolute
    error against 0.003) — it simply fails where it matters most here.

    Two things make this work, both found by calibrating against mixtures whose
    noise fraction is known by construction:

    The result is debiased for coverage. If the windows occupy a fraction c of
    the spectrum, pure noise still lands c of its energy inside them, so raw
    off-window energy underreads by exactly that much. With h the fraction of
    energy inside the windows, a mixture with noise fraction a has
    h = (1-a) + a*c, hence a = (1-h)/(1-c).

    And the windows are a constant fraction of f0 wide, so coverage is the same
    at every frequency and one global c is valid. An earlier version scaled the
    width with harmonic number, which made windows narrow low down and wide up
    high; coverage then varied across the spectrum while the debias used a
    single number, and anything with its energy concentrated low — every vowel —
    read far too noisy. Mean absolute error over formant-shaped mixtures went
    from 0.160 to 0.001 on making the width uniform.

    It inherits the pitch tracker's errors: a comb in the wrong place makes real
    harmonics look like noise. And it cannot separate noise from *inharmonic*
    partials — a bell-like tone with no noise reads about 0.15 — because
    partials off the harmonic grid are, to this measure, indistinguishable from
    noise. That needs free partials in the synthesiser, not a better measurement.
    """
    win = torch.hann_window(n_fft)
    S = torch.stft(wav, n_fft=n_fft, hop_length=HOP, window=win,
                   return_complex=True)
    P = (S.abs() ** 2)[0]                                  # (bins, T')
    bins, n_t = P.shape
    bin_hz = SR / n_fft
    f0t = F.interpolate(f0.view(1, 1, -1), size=n_t, mode="linear",
                        align_corners=True)[0, 0].clamp(min=1.0)

    # loop over harmonic number rather than over frames: 60 iterations instead
    # of thousands, which keeps this affordable on a whole training cache
    idx = torch.arange(bins).view(1, -1)                   # (1, bins)
    mask = torch.zeros(n_t, bins, dtype=torch.bool)
    hw = torch.clamp(width * f0t, min=1.5 * bin_hz).view(-1, 1)
    for k in range(1, N_HARMONICS + 1):
        c = (k * f0t).view(-1, 1)                          # (T', 1)
        mask |= ((idx * bin_hz - c).abs() <= hw) & (c < SR / 2)

    Pt = P.T                                               # (T', bins)
    tot = Pt.sum(dim=1)
    inside = (Pt * mask).sum(dim=1)
    cov = mask.to(Pt.dtype).mean(dim=1)
    h = inside / tot.clamp(min=1e-12)
    ap = ((1.0 - h) / (1.0 - cov).clamp(min=1e-6)).clamp(0.0, 1.0)
    ap = torch.where(tot > 1e-12, ap, torch.ones_like(ap))  # silence has no tone
    return F.interpolate(ap.view(1, 1, -1), size=n_frames, mode="linear",
                         align_corners=True)[0, 0]


def _top_frames_mean(v, q=0.6):
    """Mean of the loudest (1-q) of a per-frame level curve, along the last dim.

    Any gain-equivariant statistic preserves the invariance below; the loudest
    frames are used so that how much silence a take happens to contain does not
    move the reference. Falls back to the plain mean when the quantile leaves
    nothing (a perfectly flat curve).
    """
    thr = torch.quantile(v, q, dim=-1, keepdim=True)
    w = (v > thr).to(v.dtype)
    w = torch.where(w.sum(-1, keepdim=True) > 0, w, torch.ones_like(w))
    return (v * w).sum(-1) / w.sum(-1)


def take_level(loud):
    """(..., T) log-RMS -> (...) the level each take was performed at.

    Batch-friendly, which is what training needs: every clip is normalized to
    the same performed level before the loss sees it.
    """
    return _top_frames_mean(loud)


def mel_reference(mel):
    """(..., T, N_MELS) log-mel -> (..., 1, 1) the take's level, in log-power.

    Scaling a waveform by g multiplies every mel power bin by g^2, which in the
    log is a constant +2*log(g) added to every bin of every frame — and so to
    this. Subtracting it therefore cancels microphone distance and input gain
    *exactly*, while leaving spectral shape untouched.

    Deliberately a per-take offset rather than a per-frame one. A per-frame
    mean would also strip each frame's energy relative to the rest of the take,
    which is the performer's own dynamics: a frame that is quiet within a
    phrase genuinely sounds different from a loud one, and the timbre code
    should be allowed to see that. It costs nothing in invariance, since a gain
    change is constant across the whole take either way.

    The cost is that analysis is no longer causal — it needs a take-global
    statistic, so this cannot be driven straight from a live microphone.
    """
    return _top_frames_mean(mel.mean(dim=-1))[..., None, None]


def performed_level(loud, mask=None):
    """(T,) log-RMS -> the level this take was performed at.

    Everything downstream is expressed relative to this, which is what makes
    the instrument invariant to how far you stood from the microphone. `mask`
    marks the frames that carry the performance when the caller knows them;
    without it the loudest frames stand in.
    """
    if mask is not None:
        m = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        if int(m.sum()) >= 4:
            return float(loud[m].mean())
    return float(take_level(loud))


def scale_f0(f0, f0_min=F0_SCALE_MIN, f0_max=F0_SCALE_MAX):
    """Hz -> ~[0, 1] over the model's trained pitch range."""
    lo = float(np.log2(f0_min))
    return (f0.clamp(min=1).log2() - lo) / float(np.log2(f0_max / f0_min))


def level_to_amp(rel):
    """Relative level (0 = the take's performed level) -> per-frame RMS."""
    return torch.exp(torch.as_tensor(rel) + REF_LOUD)


def env_band(f):
    """Frequency in Hz -> fractional index into the N_ENV log-spaced bands.

    Shared by the torch path, the streaming path and worklet.js, so the three
    read the same envelope the same way.
    """
    span = float(np.log2((SR / 2) / ENV_F_MIN))
    lo = float(np.log2(ENV_F_MIN))
    if torch.is_tensor(f):
        pos = (f.clamp(min=ENV_F_MIN).log2() - lo) / span * (N_ENV - 1)
        return pos.clamp(0.0, N_ENV - 1.0)
    return float(np.clip((np.log2(max(f, ENV_F_MIN)) - lo) / span * (N_ENV - 1),
                         0.0, N_ENV - 1.0))


def env_to_harm(env, f0):
    """(B, T, N_ENV) envelope + (B, T) f0 in Hz -> (B, T, N_HARMONICS).

    Reads harmonic k's amplitude off the envelope at k*f0 by linear
    interpolation between bands, then normalizes to sum 1 — the envelope sets
    shape only, exactly as the direct amplitudes do, so level stays factored
    out either way.
    """
    k = torch.arange(1, N_HARMONICS + 1, device=f0.device, dtype=f0.dtype)
    pos = env_band(f0.unsqueeze(-1) * k)                   # (B, T, H)
    i0 = pos.floor().clamp(max=N_ENV - 2).long()
    frac = pos - i0.to(pos.dtype)
    e0 = torch.gather(env, 2, i0)
    e1 = torch.gather(env, 2, i0 + 1)
    harm = e0 + (e1 - e0) * frac
    return harm / (harm.sum(dim=-1, keepdim=True) + 1e-7)


# ── model ─────────────────────────────────────────────────────────────────────

def _exp_sigmoid(x):
    """DDSP paper's output activation: positive, saturating, ~exp scale."""
    return 2.0 * torch.sigmoid(x) ** 2.3 + 1e-7


class TimbreEncoder(nn.Module):
    """log-mel frames -> per-frame timbre code z.

    The level subtraction lives here rather than at the call sites so that z is
    gain-invariant by construction: there is no way to obtain a z that depends
    on recording level, because nothing else can reach the GRU.
    """
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(N_MELS, 128, batch_first=True, bidirectional=False)
        self.out = nn.Linear(128, Z_DIM)

    def forward(self, mels):                              # (B, T, N_MELS)
        h, _ = self.gru(mels - mel_reference(mels))
        return self.out(h)                                # (B, T, Z_DIM)


class Decoder(nn.Module):
    """(f0, z) frames -> harmonic shape, noise shape, and the balance between.

    Level is neither an input nor an output, and neither is the harmonic/noise
    balance. Everything here is dimensionless shape: `harm` sums to 1 and
    `noise` has unit RMS across bands. Level and balance both multiply the
    result in synthesize() and touch nothing the network computes.

    The balance used to be a `mix` head predicted from z. It is now measured
    from the signal instead (see extract_aperiodicity) and passed in as an
    input, for the same reason f0 is measured rather than learned: the network
    was demonstrably bad at it. On a whispered vowel the head answered 0.078
    where the truth is 1.0, rebuilding a whisper as a pitched voice — it had
    learned to read voicing from which frequencies carry energy, which holds
    across LibriSpeech and breaks the moment you whisper.

    Handing it in also makes the balance interpolate honestly. As a head output
    it was a sharp function of averaged z: blending a whistle with a fricative
    held at 0% noise through 25% and then jumped to 96%. A measured quantity
    stored in the map is a straight line between the two.
    """
    def __init__(self):
        super().__init__()
        in_dim = 2 + Z_DIM                      # f0, aperiodicity, z
        self.mlp_in = nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.LeakyReLU(),
                                    nn.Linear(256, 256), nn.LayerNorm(256), nn.LeakyReLU())
        self.gru = nn.GRU(256, 256, batch_first=True)
        self.mlp_out = nn.Sequential(nn.Linear(256 + in_dim, 256), nn.LayerNorm(256), nn.LeakyReLU())
        self.head_env = nn.Linear(256, N_ENV)
        self.head_noise = nn.Linear(256, N_NOISE)

    def forward(self, f0_s, ap, z, f0_hz):               # (B, T), (B, T), (B, T, Z)
        x = torch.cat([f0_s.unsqueeze(-1), ap.unsqueeze(-1), z], dim=-1)
        h = self.mlp_in(x)
        h, _ = self.gru(h)
        h = self.mlp_out(torch.cat([h, x], dim=-1))
        harm = env_to_harm(_exp_sigmoid(self.head_env(h)), f0_hz)
        noise = _exp_sigmoid(self.head_noise(h))          # (B, T, N_NOISE)
        noise = noise / (noise.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-7)
        return harm, noise


class DDSP(nn.Module):
    def __init__(self, f0_min=F0_SCALE_MIN, f0_max=F0_SCALE_MAX):
        super().__init__()
        self.encoder = TimbreEncoder()
        self.decoder = Decoder()
        # not parameters — the range the weights were trained to interpret,
        # restored from the checkpoint's config rather than the state dict
        self.f0_min, self.f0_max = f0_min, f0_max

    # -- synthesis ------------------------------------------------------------
    @staticmethod
    def _upsample(x, n_samples):
        """(B, T, C) frame signal -> (B, n_samples, C) linear interp."""
        return F.interpolate(x.transpose(1, 2), size=n_samples, mode="linear",
                             align_corners=True).transpose(1, 2)

    def synthesize(self, f0, amp, harm, noise, mix):
        """f0 (B,T) Hz; amp (B,T,1) target RMS; harm (B,T,H); noise (B,T,N);
        mix (B,T,1) -> (B, samples).

        `amp` is the only place level enters. Both branches are brought to unit
        RMS and mixed in quadrature, so their sum is unit RMS at any mix and the
        output RMS is `amp` by construction.
        """
        B, Tf = f0.shape
        n = Tf * HOP
        f0_up   = self._upsample(f0.unsqueeze(-1), n)[..., 0]        # (B, n)
        amp_up  = self._upsample(amp, n)[..., 0]
        mix_up  = self._upsample(mix, n)[..., 0].clamp(0.0, 1.0)
        harm_up = self._upsample(harm, n)                            # (B, n, H)
        phase = 2 * torch.pi * torch.cumsum(f0_up / SR, dim=1)       # (B, n)
        k = torch.arange(1, N_HARMONICS + 1, device=f0.device).float()
        # zero harmonics above Nyquist
        alias = (f0_up.unsqueeze(-1) * k) < (SR / 2)                 # (B, n, H)
        w = harm_up * alias
        # sum_k w_k sin(k phase) has RMS sqrt(sum w_k^2 / 2) for distinct k
        h_rms = (w.pow(2).sum(dim=-1) / 2).sqrt()
        audio_h = (torch.sin(phase.unsqueeze(-1) * k) * w).sum(dim=-1) / (h_rms + 1e-7)

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
        # Normalize over the *bin* gains, not the 65 band values. Filtered noise
        # has RMS proportional to the root-mean-square of the gains actually
        # applied per FFT bin, and the bands span a fixed 0..SR/2 that gets
        # interpolated onto however many bins the transform has — so band-level
        # normalization leaves a shape-dependent error (measured 15% low on a
        # learned band shape) and would be far worse in a browser running at
        # 48 kHz, where two thirds of the bins fall above the band range and are
        # zero. Normalizing here makes NOISE_UNIT_RMS a property of the
        # transform alone, which is what it claims to be.
        filt = filt / (filt.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-7)
        audio_n = torch.istft(S * filt, n_fft=n_fft, hop_length=HOP,
                              window=win, length=n) / NOISE_UNIT_RMS
        return amp_up * ((1 - mix_up).sqrt() * audio_h + mix_up.sqrt() * audio_n)

    # -- end to end -----------------------------------------------------------
    def forward(self, mels, f0, amp, ap):
        """mels (B,T,80); f0 (B,T) Hz; amp (B,T) target RMS; ap (B,T) noise
        fraction -> (B, samples)."""
        z = self.encoder(mels)
        harm, noise = self.decoder(scale_f0(f0, self.f0_min, self.f0_max), ap, z, f0)
        return self.synthesize(f0, amp.unsqueeze(-1), harm, noise, ap.unsqueeze(-1))


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

def analyze(model, wav, mask=None):
    """(1, samples) -> dict of per-frame params (f0 Hz, rel level, z).

    `rel` is loudness relative to the take's performed level, so the same
    performance recorded at two microphone distances analyses identically.
    `ref` is the level that was subtracted, reported for inspection only —
    nothing downstream reads it. `mask` marks the frames that count as
    performance. The waveform is high-passed first, so every feature here is
    derived from the same filtered signal.
    """
    n_frames = wav.shape[1] // HOP
    wav  = highpass(wav)                      # before anything reads the signal
    f0   = extract_f0(wav, n_frames)          # measure as high as we can see
    loud = extract_loudness(wav, n_frames)
    ap   = extract_aperiodicity(wav, f0, n_frames)
    mels = mel_frames(wav, n_frames)
    ref  = performed_level(loud, mask)
    with torch.no_grad():
        z = model.encoder(mels.unsqueeze(0))[0]           # (T, Z_DIM)
    return {"f0": f0, "rel": loud - ref, "ap": ap, "z": z, "ref": ref}


def blend_params(params_list, weights):
    """IDW blend of analyzed parameter dicts. f0 blends in log space
    (linear in pitch/cents -> true glissandi); level and z linearly."""
    T_min = min(p["f0"].shape[0] for p in params_list)
    logf0 = sum(w * p["f0"][:T_min].clamp(min=1).log() for w, p in zip(weights, params_list))
    rel   = sum(w * p["rel"][:T_min] for w, p in zip(weights, params_list))
    ap    = sum(w * p["ap"][:T_min] for w, p in zip(weights, params_list))
    z     = sum(w * p["z"][:T_min] for w, p in zip(weights, params_list))
    return {"f0": logf0.exp(), "rel": rel, "ap": ap, "z": z}


def synthesize_params(model, params):
    """Parameter dict -> (1, samples) waveform."""
    with torch.no_grad():
        ap = params["ap"].unsqueeze(0)
        harm, noise = model.decoder(
            scale_f0(params["f0"], model.f0_min, model.f0_max).unsqueeze(0),
            ap, params["z"].unsqueeze(0), params["f0"].unsqueeze(0))
        amp = level_to_amp(params["rel"]).unsqueeze(0).unsqueeze(-1)
        wav = model.synthesize(params["f0"].unsqueeze(0), amp, harm, noise,
                               ap.unsqueeze(-1))
    return wav


# ── streaming synthesis ───────────────────────────────────────────────────────

class StreamingSynth:
    """Frame-by-frame synthesis with carried state: decoder GRU hidden state,
    oscillator phase, parameter ramps, and noise overlap-add tail. Feed one
    (f0, rel, z) per call; get HOP samples back. Causal, ~ms per frame."""

    NOISE_FFT = 512   # per-block noise filtering (kernel <= 257 taps via OLA)

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.h = None                          # decoder GRU hidden state
        self.phase = 0.0                       # running oscillator phase (rad)
        self.prev = None                       # previous frame (f0, amp, harm, mix)
        self.noise_tail = np.zeros(self.NOISE_FFT - HOP, dtype=np.float64)

    def _decode_frame(self, f0, ap, z):
        """One decoder step with carried GRU state -> (harm, noise)."""
        d = self.model.decoder
        x = torch.empty(1, 1, 2 + Z_DIM, dtype=torch.float32)
        lo = np.log2(self.model.f0_min)                       # scale_f0, scalar
        span = np.log2(self.model.f0_max / self.model.f0_min)
        x[0, 0, 0] = (np.log2(max(f0, 1.0)) - lo) / span
        x[0, 0, 1] = float(ap)
        x[0, 0, 2:] = torch.from_numpy(np.asarray(z, dtype=np.float32))
        with torch.inference_mode():
            hh = d.mlp_in(x)
            hh, h_new = d.gru(hh, self.h)
            self.h = h_new.clone()
            hh = d.mlp_out(torch.cat([hh, x], dim=-1))
            harm = env_to_harm(_exp_sigmoid(d.head_env(hh)),
                               torch.tensor([[float(f0)]]))[0, 0]
            noise = _exp_sigmoid(d.head_noise(hh))[0, 0]
            noise = noise / (noise.pow(2).mean().sqrt() + 1e-7)
        return (harm.numpy().astype(np.float64),
                noise.numpy().astype(np.float64))

    def render(self, f0, rel, ap, z):
        """One frame of parameters -> HOP samples (numpy float32)."""
        harm, noise = self._decode_frame(f0, ap, z)
        mix = float(np.clip(ap, 0.0, 1.0))
        amp = float(np.exp(rel + REF_LOUD))
        if self.prev is None:
            self.prev = (f0, amp, harm, mix)
        f0_a, amp_a, harm_a, mix_a = self.prev

        # linear ramps from previous frame's values across the block
        r = (np.arange(HOP) + 1.0) / HOP
        f0_up   = f0_a + (f0 - f0_a) * r
        amp_up  = amp_a + (amp - amp_a) * r
        mix_up  = np.clip(mix_a + (mix - mix_a) * r, 0.0, 1.0)
        harm_up = harm_a[None, :] + (harm - harm_a)[None, :] * r[:, None]

        phase = self.phase + 2 * np.pi * np.cumsum(f0_up) / SR
        self.phase = float(phase[-1] % (2 * np.pi))
        k = np.arange(1, N_HARMONICS + 1)
        alias = (f0_up[:, None] * k[None, :]) < (SR / 2)
        w = harm_up * alias
        h_rms = np.sqrt((w ** 2).sum(axis=1) / 2.0)
        audio_h = (np.sin(phase[:, None] * k[None, :]) * w).sum(axis=1) / (h_rms + 1e-7)

        # filtered noise: block FFT multiply + overlap-add tail
        wn = np.random.uniform(-1, 1, HOP)
        spec = np.fft.rfft(wn, n=self.NOISE_FFT)
        filt = np.interp(np.linspace(0, 1, self.NOISE_FFT // 2 + 1),
                         np.linspace(0, 1, N_NOISE), noise)
        filt = filt / (np.sqrt((filt ** 2).mean()) + 1e-7)   # see synthesize()
        block = np.fft.irfft(spec * filt, n=self.NOISE_FFT)
        audio_n = (block[:HOP] + np.concatenate(
            [self.noise_tail, np.zeros(HOP - min(HOP, len(self.noise_tail)))])[:HOP]
        ) / NOISE_UNIT_RMS
        self.noise_tail = block[HOP:]

        self.prev = (f0, amp, harm, mix)
        return (amp_up * (np.sqrt(1 - mix_up) * audio_h
                          + np.sqrt(mix_up) * audio_n)).astype(np.float32)
