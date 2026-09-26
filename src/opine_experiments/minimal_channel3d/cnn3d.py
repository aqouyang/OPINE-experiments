"""CNN autoencoder baseline for the 3-D minimal channel.

A direct transcription of `JointCNN_YZ` (experiments/channel_retau180_yz/
cnn_yz_joint.py) to three dimensions.  Nothing about the design is new: same
four stride-2 stages, same 3->32->64->128->256 channel ladder, same
BatchNorm+ReLU, same linear bottleneck, same nearest-upsample decoder.  Only
the convolution rank and the padding axes change.

Tensor layout follows dataset3d: (B, 3, z, y, x) = (B, 3, 64, 32, 32).

PADDING.  x and y are homogeneous and periodic -> circular.  z is wall-bounded
-> zero.  This is the same rule `MixedPadConv3d` uses inside OPINE, so the two
models see the boundaries identically.  Zero padding is an architectural
choice, not a physical boundary condition: the lower wall is no-slip and the
upper boundary is stress-free, and plain zero padding implements neither.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mixed_pad(v, p):
    """Circular in x (dim -1) and y (dim -2), zero in z (dim -3)."""
    if p:
        v = F.pad(v, (p, p, p, p, 0, 0), mode="circular")
        v = F.pad(v, (0, 0, 0, 0, p, p), mode="constant")
    return v


class MixedPadConv3d(nn.Module):
    """Conv3d with circular x/y padding and zero z padding."""

    def __init__(self, c_in, c_out, k=3, stride=1):
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv3d(c_in, c_out, k, stride=stride, padding=0)

    def forward(self, v):
        return self.conv(_mixed_pad(v, self.pad))


class MixedPadConvUp3d(nn.Module):
    """Nearest upsample followed by a mixed-pad conv (no checkerboard)."""

    def __init__(self, c_in, c_out, k=3, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.pad = k // 2
        self.conv = nn.Conv3d(c_in, c_out, k, padding=0)

    def forward(self, v):
        v = F.interpolate(v, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(_mixed_pad(v, self.pad))


class JointCNN3D(nn.Module):
    """Joint three-component 3-D CNN autoencoder.

    Parameters
    ----------
    input_shape : (NZ, NY, NX)
    latent_dim : int
        Bottleneck dimension = the reported DOF, exactly as for POD and OPINE.
    base_channels, num_down : as in JointCNN_YZ.
    """

    def __init__(self, input_shape=(64, 32, 32), latent_dim=32,
                 base_channels=32, num_down=4):
        super().__init__()
        Z, Y, X = input_shape
        f = 2 ** num_down
        assert Z % f == 0 and Y % f == 0 and X % f == 0
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_down = num_down

        enc, c_in, c = [], 3, base_channels
        for _ in range(num_down):
            enc += [MixedPadConv3d(c_in, c, 3, stride=2),
                    nn.BatchNorm3d(c), nn.ReLU(inplace=True)]
            c_in, c = c, min(c * 2, base_channels * 8)
        self.encoder = nn.Sequential(*enc)

        self.bz, self.by, self.bx = Z // f, Y // f, X // f
        self.bc = c_in
        flat = c_in * self.bz * self.by * self.bx
        self.flat = flat
        self.enc_linear = nn.Linear(flat, latent_dim)
        self.dec_linear = nn.Linear(latent_dim, flat)

        dec, c = [], c_in
        for i in range(num_down):
            c_out = base_channels if i == num_down - 1 \
                else max(base_channels, c // 2)
            dec += [MixedPadConvUp3d(c, c_out, 3, scale_factor=2),
                    nn.BatchNorm3d(c_out), nn.ReLU(inplace=True)]
            c = c_out
        dec.append(MixedPadConv3d(c, 3, 3, stride=1))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.enc_linear(self.encoder(x).flatten(1))

    def decode(self, z):
        h = self.dec_linear(z).reshape(-1, self.bc, self.bz, self.by, self.bx)
        return self.decoder(h)

    def forward(self, x):
        """x: (B, 3, Z, Y, X) -> (recon, latent)"""
        z = self.encode(x)
        r = self.decode(z)
        if r.shape[-3:] != x.shape[-3:]:
            r = F.interpolate(r, size=x.shape[-3:], mode="trilinear",
                              align_corners=False)
        return r, z

    def count_bottleneck_dof(self):
        return self.latent_dim

    def param_groups(self):
        """Parameter counts by block, for the capacity table."""
        g = {"encoder_conv": sum(p.numel() for p in self.encoder.parameters()),
             "enc_linear": sum(p.numel() for p in self.enc_linear.parameters()),
             "dec_linear": sum(p.numel() for p in self.dec_linear.parameters()),
             "decoder_conv": sum(p.numel() for p in self.decoder.parameters())}
        g["total"] = sum(p.numel() for p in self.parameters())
        g["latent_adjacent"] = g["enc_linear"] + g["dec_linear"]
        g["latent_adjacent_frac"] = g["latent_adjacent"] / g["total"]
        g["flat_dim"] = self.flat
        return g
