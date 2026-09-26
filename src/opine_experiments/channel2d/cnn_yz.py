"""
CNN autoencoder for y-z slices.

y (dim -2): periodic → circular padding
z (dim -1): wall-normal bounded → zero padding

Same architecture scale as the x-y periodic CNN (4 stride-2 stages,
base_channels=32), adapted for mixed boundary conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedPadConv2d(nn.Module):
    """Conv2d: circular along y (dim -2), zero along z (dim -1)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=0)

    def forward(self, x):
        p = self.pad
        x = F.pad(x, [0, 0, p, p], mode="circular")   # y circular
        x = F.pad(x, [p, p, 0, 0], mode="constant")    # z zero
        return self.conv(x)


class MixedPadConvUp2d(nn.Module):
    """Nearest upsample + mixed-pad conv (no checkerboard)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        p = self.pad
        x = F.pad(x, [0, 0, p, p], mode="circular")
        x = F.pad(x, [p, p, 0, 0], mode="constant")
        return self.conv(x)


class CNN_YZ(nn.Module):
    """CNN autoencoder for y-z slices with mixed boundary padding.

    Parameters
    ----------
    input_hw : (H_y, W_z)
    latent_dim : int
        Bottleneck vector dimension (= real DOF).
    base_channels : int
    num_down : int
        Number of stride-2 downsampling stages.
    """

    def __init__(self, input_hw=(128, 64), latent_dim=32,
                 base_channels=32, num_down=4):
        super().__init__()
        H, W = input_hw
        factor = 2 ** num_down
        assert H % factor == 0 and W % factor == 0, \
            f"{input_hw} not divisible by 2**{num_down}={factor}"
        self.input_hw = input_hw
        self.latent_dim = latent_dim
        self.num_down = num_down

        # Encoder
        enc = []
        c_in = 1
        c = base_channels
        for _ in range(num_down):
            enc += [MixedPadConv2d(c_in, c, 3, stride=2),
                    nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            c_in = c
            c = min(c * 2, base_channels * 8)
        self.encoder = nn.Sequential(*enc)

        self.bh = H // factor
        self.bw = W // factor
        self.bc = c_in
        flat = c_in * self.bh * self.bw

        self.enc_linear = nn.Linear(flat, latent_dim)
        self.dec_linear = nn.Linear(latent_dim, flat)

        # Decoder
        dec = []
        c = c_in
        for i in range(num_down):
            c_out = max(base_channels, c // 2)
            if i == num_down - 1:
                c_out = base_channels
            dec += [MixedPadConvUp2d(c, c_out, 3, scale_factor=2),
                    nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            c = c_out
        dec.append(MixedPadConv2d(c, 1, 3, stride=1))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.enc_linear(self.encoder(x).flatten(1))

    def decode(self, z):
        h = self.dec_linear(z).reshape(-1, self.bc, self.bh, self.bw)
        return self.decoder(h)

    def forward(self, x):
        """x: (B, 1, H, W) → (recon, latent)"""
        z = self.encode(x)
        recon = self.decode(z)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return recon, z

    def count_bottleneck_dof(self):
        return self.latent_dim
