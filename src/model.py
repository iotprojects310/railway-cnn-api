import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x):
        # x: (B, C, L)
        z = self.encoder(x)
        out = self.decoder(z)
        # if shapes mismatch due to odd lengths, trim or pad
        if out.size(-1) != x.size(-1):
            out = nn.functional.interpolate(out, size=x.size(-1), mode='linear', align_corners=False)
        return out
