import torch
import torch.nn as nn


# ── mel-spectrogram models (2D conv) ─────────────────────────────────────────

LATENT_H = 5   # N_MELS(80) / 2^4  (freq compressed 4 times)
LATENT_W = 1   # CLIP_FRAMES(1)    (time not compressed)


def _conv2d_stack():
    # stride=(2,1): compress mel-freq by 2x each layer, leave time unchanged
    return nn.Sequential(
        nn.Conv2d(1,  16, kernel_size=3, stride=(2, 1), padding=1), nn.ELU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1), nn.ELU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1), nn.ELU(),
    )


class Encoder(nn.Module):
    def __init__(self, latent_ch=8):
        super().__init__()
        self.layers = nn.Sequential(
            *_conv2d_stack(),
            nn.Conv2d(64, latent_ch, kernel_size=3, stride=(2, 1), padding=1),
        )

    def forward(self, x):
        return self.layers(x)


class VAEEncoder(nn.Module):
    def __init__(self, latent_ch=8):
        super().__init__()
        self.shared     = _conv2d_stack()
        self.mu_head     = nn.Conv2d(64, latent_ch, kernel_size=3, stride=(2, 1), padding=1)
        self.logvar_head = nn.Conv2d(64, latent_ch, kernel_size=3, stride=(2, 1), padding=1)

    def forward(self, x):
        h = self.shared(x)
        return self.mu_head(h), self.logvar_head(h)


class Decoder(nn.Module):
    def __init__(self, latent_ch=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)), nn.ELU(),
            nn.ConvTranspose2d(64, 32,        kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)), nn.ELU(),
            nn.ConvTranspose2d(32, 16,        kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)), nn.ELU(),
            nn.ConvTranspose2d(16,  1,        kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)),
        )

    def forward(self, x):
        return self.layers(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_ch=8):
        super().__init__()
        self.latent_ch = latent_ch
        self.encoder = Encoder(latent_ch)
        self.decoder = Decoder(latent_ch)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, latent_ch=8):
        super().__init__()
        self.latent_ch = latent_ch
        self.encoder = VAEEncoder(latent_ch)
        self.decoder = Decoder(latent_ch)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        logvar = logvar.clamp(-4, 4)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z)


# ── MLP models (per-frame, no time axis) ─────────────────────────────────────

_N_MELS = 80   # must match dataset.N_MELS


class MLPEncoder(nn.Module):
    def __init__(self, latent_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_N_MELS, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, latent_ch),
        )

    def forward(self, x):   # x: (batch, 1, _N_MELS, 1)
        return self.net(x.squeeze(1).squeeze(-1))   # (batch, latent_ch)


class MLPVAEEncoder(nn.Module):
    def __init__(self, latent_ch=16):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(_N_MELS, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.mu_head     = nn.Linear(128, latent_ch)
        self.logvar_head = nn.Linear(128, latent_ch)

    def forward(self, x):   # x: (batch, 1, _N_MELS, 1)
        h = self.shared(x.squeeze(1).squeeze(-1))
        return self.mu_head(h), self.logvar_head(h)


class MLPDecoder(nn.Module):
    def __init__(self, latent_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_ch, 128), nn.ELU(),
            nn.Linear(128, 256), nn.ELU(),
            nn.Linear(256, _N_MELS),
        )

    def forward(self, z):   # z: (batch, latent_ch)
        return self.net(z).unsqueeze(1).unsqueeze(-1)   # (batch, 1, _N_MELS, 1)


class MLPAutoencoder(nn.Module):
    def __init__(self, latent_ch=16):
        super().__init__()
        self.latent_ch = latent_ch
        self.encoder   = MLPEncoder(latent_ch)
        self.decoder   = MLPDecoder(latent_ch)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class MLPVAE(nn.Module):
    def __init__(self, latent_ch=16):
        super().__init__()
        self.latent_ch = latent_ch
        self.encoder   = MLPVAEEncoder(latent_ch)
        self.decoder   = MLPDecoder(latent_ch)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        logvar = logvar.clamp(-4, 4)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z)


# ── legacy waveform models (1D conv) — kept for loading old checkpoints ───────

def _conv1d_stack():
    return nn.Sequential(
        nn.Conv1d(1,  16, kernel_size=9, stride=2, padding=4), nn.ELU(),
        nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4), nn.ELU(),
        nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4), nn.ELU(),
    )


class _WaveformEncoder(nn.Module):
    def __init__(self, latent_ch=2):
        super().__init__()
        self.layers = nn.Sequential(
            *_conv1d_stack(),
            nn.Conv1d(64, latent_ch, kernel_size=9, stride=2, padding=4),
        )

    def forward(self, x):
        return self.layers(x)


class _WaveformVAEEncoder(nn.Module):
    def __init__(self, latent_ch=2):
        super().__init__()
        self.shared      = _conv1d_stack()
        self.mu_head     = nn.Conv1d(64, latent_ch, kernel_size=9, stride=2, padding=4)
        self.logvar_head = nn.Conv1d(64, latent_ch, kernel_size=9, stride=2, padding=4)

    def forward(self, x):
        h = self.shared(x)
        return self.mu_head(h), self.logvar_head(h)


class _WaveformDecoder(nn.Module):
    def __init__(self, latent_ch=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(latent_ch, 64, kernel_size=9, stride=2, padding=4, output_padding=1), nn.ELU(),
            nn.ConvTranspose1d(64, 32,        kernel_size=9, stride=2, padding=4, output_padding=1), nn.ELU(),
            nn.ConvTranspose1d(32, 16,        kernel_size=9, stride=2, padding=4, output_padding=1), nn.ELU(),
            nn.ConvTranspose1d(16,  1,        kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)


class WaveformAutoencoder(nn.Module):
    def __init__(self, latent_ch=2):
        super().__init__()
        self.latent_ch = latent_ch
        self.encoder = _WaveformEncoder(latent_ch)
        self.decoder = _WaveformDecoder(latent_ch)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class WaveformVAE(nn.Module):
    def __init__(self, latent_ch=2):
        super().__init__()
        self.latent_ch = latent_ch
        self.encoder = _WaveformVAEEncoder(latent_ch)
        self.decoder = _WaveformDecoder(latent_ch)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        logvar = logvar.clamp(-4, 4)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z)
