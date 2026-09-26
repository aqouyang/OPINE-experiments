"""CNN autoencoder for 2-D Kolmogorov vorticity.

Adapted from channel2d/cnn_yz_joint.py (JointCNN_YZ),
keeping its encoder/decoder philosophy unchanged: four stride-2 stages,
base_channels=32, channel doubling capped at 8x, BatchNorm+ReLU, a linear
bottleneck of exactly `latent_dim` real scalars, and a nearest-upsample
decoder to avoid checkerboarding.

The only changes are the ones the physics forces:
  * 1 input/output channel (scalar vorticity) instead of 3
  * 128 x 128 instead of 128 x 64
  * circular padding in BOTH directions instead of circular-y / zero-z,
    because this domain is doubly periodic

Circular padding in y is the correct boundary condition, not a symmetry
assumption: the forcing -k_f cos(k_f y) makes the statistics y-dependent, and
nothing here ties weights across y or restricts the latent subspace.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularPadConv2d(nn.Module):
    """Conv2d with circular padding in both spatial directions."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=0)

    def forward(self, x):
        p = self.pad
        if p:
            x = F.pad(x, [p, p, p, p], mode="circular")
        return self.conv(x)


class CircularPadConvUp2d(nn.Module):
    """Nearest upsample + circular-pad conv (no checkerboard)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        p = self.pad
        if p:
            x = F.pad(x, [p, p, p, p], mode="circular")
        return self.conv(x)


class KolmogorovCNN(nn.Module):
    """Vorticity autoencoder.  Input (B, 1, 128, 128), latent (B, latent_dim)."""

    def __init__(self, input_hw=(128, 128), latent_dim=32,
                 base_channels=32, num_down=4, channels=1):
        super().__init__()
        H, W = input_hw
        factor = 2 ** num_down
        assert H % factor == 0 and W % factor == 0
        self.input_hw = input_hw
        self.latent_dim = latent_dim
        self.channels = channels

        enc = []
        c_in, c = channels, base_channels
        for _ in range(num_down):
            enc += [CircularPadConv2d(c_in, c, 3, stride=2),
                    nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            c_in = c
            c = min(c * 2, base_channels * 8)
        self.encoder = nn.Sequential(*enc)

        self.bh, self.bw, self.bc = H // factor, W // factor, c_in
        flat = c_in * self.bh * self.bw
        self.enc_linear = nn.Linear(flat, latent_dim)
        self.dec_linear = nn.Linear(latent_dim, flat)

        dec = []
        c = c_in
        for i in range(num_down):
            c_out = base_channels if i == num_down - 1 else max(base_channels,
                                                                c // 2)
            dec += [CircularPadConvUp2d(c, c_out, 3, scale_factor=2),
                    nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            c = c_out
        dec.append(CircularPadConv2d(c, channels, 3, stride=1))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.enc_linear(self.encoder(x).flatten(1))

    def decode(self, z):
        h = self.dec_linear(z).reshape(-1, self.bc, self.bh, self.bw)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
