import torch
import torch.nn as nn


def _conv_stack():
    return nn.Sequential(
        nn.Conv1d(1,  16, kernel_size=9, stride=2, padding=4),
        nn.ELU(),
        nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
        nn.ELU(),
        nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
        nn.ELU(),
    )


class Encoder(nn.Module):
    def __init__(self, latent_ch=2):
        super().__init__()
        self.layers = nn.Sequential(
            *_conv_stack(),
            nn.Conv1d(64, latent_ch, kernel_size=9, stride=2, padding=4),
        )

    def forward(self, x):
        return self.layers(x)


class VAEEncoder(nn.Module):
    def __init__(self, latent_ch=2):
        super().__init__()
        self.shared = _conv_stack()
        self.mu_head     = nn.Conv1d(64, latent_ch, kernel_size=9, stride=2, padding=4)
        self.logvar_head = nn.Conv1d(64, latent_ch, kernel_size=9, stride=2, padding=4)

    def forward(self, x):
        h = self.shared(x)
        return self.mu_head(h), self.logvar_head(h)


class Decoder(nn.Module):
    def __init__(self, latent_ch=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(latent_ch, 64, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(16,  1, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_ch=2):
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
    def __init__(self, latent_ch=2):
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
