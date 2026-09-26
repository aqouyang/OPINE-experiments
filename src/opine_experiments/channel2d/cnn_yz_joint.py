"""
Joint three-component CNN autoencoder for y-z slices.

Input: (B, 3, ny, nz).  Latent: (B, latent_dim).
y (dim -2): periodic -> circular padding.
z (dim -1): wall-normal bounded -> zero padding.
"""

import torch.nn as nn
import torch.nn.functional as F

from opine_experiments.channel2d.cnn_yz import (
    MixedPadConv2d, MixedPadConvUp2d,
)


class JointCNN_YZ(nn.Module):
    """Joint 3-component CNN autoencoder with mixed boundary padding.

    Parameters
    ----------
    input_hw : (H_y, W_z)
    latent_dim : int
        Bottleneck dimension (= total real DOF for all 3 components).
    base_channels : int
    num_down : int
    """

    def __init__(self, input_hw=(128, 64), latent_dim=32,
                 base_channels=32, num_down=4):
        super().__init__()
        H, W = input_hw
        factor = 2 ** num_down
        assert H % factor == 0 and W % factor == 0
        self.input_hw = input_hw
        self.latent_dim = latent_dim
        self.num_down = num_down

        # Encoder: 3 -> 32 -> 64 -> 128 -> 256
        enc = []
        c_in = 3
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
        dec.append(MixedPadConv2d(c, 3, 3, stride=1))  # output 3 channels
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.enc_linear(self.encoder(x).flatten(1))

    def decode(self, z):
        h = self.dec_linear(z).reshape(-1, self.bc, self.bh, self.bw)
        return self.decoder(h)

    def forward(self, x):
        """x: (B, 3, H, W) -> (recon, latent)"""
        z = self.encode(x)
        recon = self.decode(z)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return recon, z

    def count_bottleneck_dof(self):
        return self.latent_dim
