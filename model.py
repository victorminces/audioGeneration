import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_ch=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=9, stride=2, padding=4),
            nn.ELU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.ELU(),
            nn.Conv1d(64, latent_ch, kernel_size=9, stride=2, padding=4),
        )

    def forward(self, x):
        return self.layers(x)


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
