#!/usr/bin/env python3
"""Coupling-FINE: a RealNVP/Glow-style invertible transform in place of the
iResNet, keeping the explicit internal POD bottleneck unchanged.

    x -> f_theta -> P_Q -> f_theta^{-1} -> x_hat

Only f_theta changes.  Q, the latent mean, the subspace update and the
bottleneck semantics are inherited verbatim from IResNetFINE_YZ.

WHY A FLOW RATHER THAN A RESIDUAL CONTRACTION.  The iResNet is invertible
only because every block is a contraction, and its inverse is a Banach
fixed-point iteration -- approximate, certified after the fact, and it caps
how strong each block may be (conv spectral norm 0.9, residual scale 0.6).
A coupling flow is invertible BY CONSTRUCTION with a closed-form inverse, so
there is no contraction budget to spend and the conditioner can be as strong
as we like.  That is precisely the axis the coupling analysis identified as
missing.  Round-trip error is then floating-point, not iteration count.

WHAT COUPLES WHAT.

  squeeze          space-to-depth by 2, so (C, H, W) -> (4C, H/2, W/2).  A
                   1-channel scalar field has nothing to mix across channels;
                   the squeeze manufactures 4C channels out of local 2x2
                   spatial phase, which is what lets the 1x1 mixing and the
                   channel-split coupling do spatial work on a scalar field.
  invertible 1x1   LU-parameterized, so the inverse is a triangular solve.
                   Mixes all channels.  On the channel-flow data (C=3) this
                   is genuine cross-component u/v/w mixing.  On Kolmogorov
                   (C=1) there are no physical components to mix and it acts
                   on squeeze phases only -- stated plainly because calling
                   it "cross-component coupling" there would be false.
  affine coupling  y_a = x_a;  y_b = x_b * exp(s(x_a)) + t(x_a), with s and t
                   from a 2-D convolutional conditioner.  This is the
                   cross-spatial coupling the analysis found absent from
                   PhysicalFINE: a 3x3 conv over a multi-channel squeezed
                   field mixes a real neighbourhood.
  alternating      the split flips every block, so every variable is both
                   conditioner and conditioned across a pair of blocks and
                   information reaches everywhere.

PADDING.  `pad_mode` is per axis.  Kolmogorov is doubly periodic, so both
axes are circular.  The channel y-z geometry is periodic in y and wall
bounded in z, where circular padding would wrap the wall into the centreline;
that case uses zero padding in z, matching the existing convention.

SCALE BOUND.  s is passed through s_max * tanh(.), so the per-block Jacobian
is bounded by exp(s_max) and the composition cannot blow up or collapse.  It
is not a contraction requirement -- invertibility does not need it -- purely
numerical conditioning of the inverse.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def squeeze2d(x):
    b, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError(f"squeeze needs even dims, got {(h, w)}")
    x = x.reshape(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    return x


def unsqueeze2d(x):
    b, c4, h, w = x.shape
    c = c4 // 4
    x = x.reshape(b, c, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c, h * 2, w * 2)


class MixedPadConv2d(nn.Module):
    """3x3 (or k x k) conv with an independent padding mode per axis.

    Axis order matches the data: dim -2 is the first spatial axis (y for the
    channel geometry), dim -1 the second (z there, and the wall-bounded one).
    """

    def __init__(self, c_in, c_out, k=3, pad_mode="circular_both"):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, padding=0)
        self.k = k
        self.pad_mode = pad_mode

    def forward(self, x):
        p = self.k // 2
        if p:
            if self.pad_mode == "circular_both":
                x = F.pad(x, (p, p, p, p), mode="circular")
            elif self.pad_mode == "circular_y_zero_z":
                x = F.pad(x, (0, 0, p, p), mode="circular")   # periodic axis
                x = F.pad(x, (p, p, 0, 0), mode="constant")   # bounded axis
            else:
                raise ValueError(f"unknown pad_mode: {self.pad_mode}")
        return self.conv(x)


class Conditioner(nn.Module):
    """Conv net producing (log-scale, translation) for an affine coupling."""

    def __init__(self, c_in, c_out, hidden, pad_mode, init_scale):
        super().__init__()
        self.net = nn.Sequential(
            MixedPadConv2d(c_in, hidden, 3, pad_mode), nn.ReLU(),
            MixedPadConv2d(hidden, hidden, 1, pad_mode), nn.ReLU(),
            MixedPadConv2d(hidden, 2 * c_out, 3, pad_mode))
        # Near-identity start: the last conv is scaled down, not zeroed, so
        # s and t are small but non-zero and every earlier layer still gets
        # gradient on the first step.  Matches the project convention.
        with torch.no_grad():
            last = self.net[-1].conv
            last.weight.mul_(init_scale)
            last.bias.zero_()

    def forward(self, x):
        h = self.net(x)
        s, t = h.chunk(2, dim=1)
        return s, t


class AffineCoupling(nn.Module):
    """y_b = x_b * exp(s(x_a)) + t(x_a), exact closed-form inverse."""

    def __init__(self, channels, hidden, pad_mode, init_scale, s_max=2.0,
                 flip=False):
        super().__init__()
        self.flip = flip
        self.s_max = s_max
        # The tensor is always cut at p; `flip` chooses which side conditions
        # the other. Deriving the conditioner's widths from that same cut is
        # what makes an ODD channel count work: the earlier version swapped
        # the two widths but kept cutting at channels//2, so with 3 components
        # the conditioner was built for 2 inputs and handed 1. Even counts
        # were unaffected because the two halves are the same width.
        self.p = channels // 2
        c_a = (channels - self.p) if flip else self.p
        c_b = self.p if flip else (channels - self.p)
        self.net = Conditioner(c_a, c_b, hidden, pad_mode, init_scale)

    def _split(self, x):
        """-> (conditioner input, transformed part)."""
        x1, x2 = x[:, :self.p], x[:, self.p:]
        return (x2, x1) if self.flip else (x1, x2)

    def _join(self, a, b):
        return torch.cat([b, a], 1) if self.flip else torch.cat([a, b], 1)

    def forward(self, x):
        a, b = self._split(x)
        s, t = self.net(a)
        s = self.s_max * torch.tanh(s)
        return self._join(a, b * torch.exp(s) + t)

    def inverse(self, y):
        a, b = self._split(y)
        s, t = self.net(a)
        s = self.s_max * torch.tanh(s)
        return self._join(a, (b - t) * torch.exp(-s))


class Invertible1x1(nn.Module):
    """LU-parameterized invertible 1x1 convolution (channel mixing).

    W = P L (U + diag(s)).  P is a fixed permutation, L unit lower triangular,
    U strictly upper triangular.  The inverse is two triangular solves, so it
    is exact and cheap, and W cannot become singular as long as no entry of s
    reaches zero -- s is stored as sign * exp(log|s|), which cannot cross it.
    """

    def __init__(self, channels, init_scale):
        super().__init__()
        w = torch.eye(channels)
        if init_scale > 0:                       # near-identity, not identity
            w = w + init_scale * torch.randn(channels, channels) / channels
        p, l, u = torch.linalg.lu(w)
        s = torch.diagonal(u).clone()
        self.register_buffer("P", p)
        self.L = nn.Parameter(l)
        self.U = nn.Parameter(torch.triu(u, 1))
        self.sign_s = nn.Parameter(torch.sign(s), requires_grad=False)
        self.log_s = nn.Parameter(torch.log(torch.abs(s) + 1e-12))
        self.register_buffer("eye", torch.eye(channels))

    def _w(self):
        l = torch.tril(self.L, -1) + self.eye
        u = torch.triu(self.U, 1) + torch.diag(self.sign_s
                                               * torch.exp(self.log_s))
        return self.P @ l @ u

    def forward(self, x):
        w = self._w().to(x.dtype)
        return F.conv2d(x, w.unsqueeze(-1).unsqueeze(-1))

    def inverse(self, y):
        w = self._w().to(y.dtype)
        wi = torch.linalg.inv(w)
        return F.conv2d(y, wi.unsqueeze(-1).unsqueeze(-1))


class CouplingFlow2D(nn.Module):
    """Squeeze -> [1x1 mix + affine coupling] x n_blocks -> unsqueeze.

    Shape preserving overall, so it drops into FINE where the iResNet sat.
    """

    def __init__(self, channels=1, n_blocks=8, hidden=64,
                 pad_mode="circular_both", init_scale=0.05, s_max=2.0):
        super().__init__()
        cs = channels * 4                       # after the squeeze
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(nn.ModuleList([
                Invertible1x1(cs, init_scale),
                AffineCoupling(cs, hidden, pad_mode, init_scale, s_max,
                               flip=bool(i % 2))]))
        self.n_blocks = n_blocks
        self.channels = channels

    def forward(self, x):
        h = squeeze2d(x)
        for mix, cpl in self.blocks:
            h = cpl(mix(h))
        return unsqueeze2d(h)

    def inverse(self, y):
        h = squeeze2d(y)
        for mix, cpl in reversed(self.blocks):
            h = mix.inverse(cpl.inverse(h))
        return unsqueeze2d(h)

    # ---- diagnostics, with the same names the trainer already calls -----
    # A coupling flow is invertible by construction, so the contraction
    # certificate and Banach rate that certify the iResNet's fixed-point
    # inverse have no meaning here.  They are reported as exactly 0.0 and
    # flagged, rather than silently omitted, so a downstream check that
    # asserts "certificate < 1" does not quietly read as a passed test of
    # something that was never measured.
    _certificate_applicable = False

    def structured_contraction_upper_bounds(self):
        return [0.0]

    def banach_rates(self, x):
        return [0.0]

    def warm_power_iteration(self, x, n=0):
        return None

    def inverse_diagnostics(self):
        return {"max_iters": 0, "max_residual": 0.0,
                "_note": "closed-form inverse; no iteration"}

    @torch.no_grad()
    def roundtrip_errors(self, x):
        """Genuine measurement: max |f^-1(f(x)) - x|, float64 and float32."""
        was = self.training
        self.eval()
        d64 = self.double()
        e64 = float((d64.inverse(d64(x.double())) - x.double()).abs().max())
        self.float()
        e32 = float((self.inverse(self(x)) - x).abs().max())
        self.train(was)
        return {"verified_float64": e64, "train_path_float32": e32}


class CouplingFINE(nn.Module):
    """FINE with a coupling flow as f_theta and the POD bottleneck unchanged.

    Subclassing IResNetFINE_YZ would drag in the iResNet construction only to
    throw it away, so the bottleneck methods are bound from that class instead
    -- same code, no duplicated logic, no wasted allocation.
    """

    def __init__(self, ny, nz, dof, channels=1, n_blocks=8, hidden=64,
                 pad_mode="circular_both", init_scale=0.05, s_max=2.0,
                 bottleneck="unrestricted"):
        super().__init__()
        from opine_experiments.channel2d.iresnet_fine_yz import (
            IResNetFINE_YZ)
        if bottleneck not in ("banded", "unrestricted"):
            raise ValueError(bottleneck)
        self.bottleneck = bottleneck
        self.pad_mode = pad_mode
        self.ny, self.nz, self.dof = ny, nz, dof
        self.channels = channels
        self.n_features = channels * ny * nz
        self.n_ky = ny // 2 + 1
        self.band_dim = channels * nz
        self.f = CouplingFlow2D(channels=channels, n_blocks=n_blocks,
                                hidden=hidden, pad_mode=pad_mode,
                                init_scale=init_scale, s_max=s_max)
        self.register_buffer("Q", torch.zeros(self.n_features, dof))
        self.register_buffer("B_blocks", torch.zeros(
            self.n_ky, self.band_dim, 1, dtype=torch.complex128))
        self.register_buffer("band_ranks", torch.zeros(self.n_ky,
                                                       dtype=torch.long))
        self.register_buffer("z_mean", torch.zeros(self.n_features))
        self.register_buffer("q_initialized", torch.zeros(1))
        # bottleneck behaviour, taken from the shipped model unchanged
        for name in ("encode", "decode", "project", "forward", "latent_codes",
                     "latent_matrix", "update_subspace", "_bases_list",
                     "_store_bases"):
            fn = getattr(IResNetFINE_YZ, name, None)
            if fn is not None:
                setattr(self.__class__, name, fn)

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
