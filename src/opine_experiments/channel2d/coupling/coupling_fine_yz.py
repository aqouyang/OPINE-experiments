#!/usr/bin/env python3
"""Coupling-FINE for the channel y-z geometry.

Same idea as the Kolmogorov version -- replace f_theta with a coupling flow,
leave the POD bottleneck alone -- but with one structural difference that is
NOT optional here.

WHY NO SQUEEZE. The Kolmogorov flow squeezes space into channels because the
field is a single scalar and there is otherwise nothing to split on. Squeeze
is space-to-depth by 2, so a shift of one grid point permutes channels: it is
equivariant only to EVEN shifts. That is harmless on Kolmogorov, whose
bottleneck is unrestricted. It is not harmless here. The channel bottleneck is
`banded` -- the retained subspace is a direct sum of whole k_y bands precisely
so the projector is exactly y-translation equivariant, which is the physics of
a homogeneous spanwise direction. Composing an equivariant projector with a
transform that is only half-equivariant would silently break the property the
band structure exists to guarantee.

The channel field already has three components, so no squeeze is needed: the
affine coupling splits on u/v/w directly (1 | 2, alternating), and every
operation in the flow is either pointwise in y or a circular convolution, so
exact y-translation equivariance is preserved. `equivariance_error` measures
it rather than asserting it.

The invertible 1x1 therefore mixes GENUINE velocity components here, unlike
the Kolmogorov case where it could only mix squeeze phases. Cross-component
coupling is the thing `coupling_analysis.md` identified as structurally absent
from PhysicalFINE, so on this dataset it is finally under test.

PADDING. Circular in y (periodic, Ly = pi) and zero in z (wall-bounded,
Lz = 1). Circular padding in z would wrap the wall into the centreline.
"""
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opine_experiments.kolmogorov.coupling.coupling_flow import (      # noqa: E402
    AffineCoupling, Invertible1x1)
from opine_experiments.channel2d.iresnet_fine_yz import (    # noqa: E402
    IResNetFINE_YZ)


class CouplingFlowYZ(nn.Module):
    """[ invertible 1x1 over u/v/w + affine coupling ] x n_blocks.

    Shape preserving, invertible by construction, closed-form inverse, and
    exactly y-translation equivariant.
    """

    def __init__(self, channels=3, n_blocks=8, hidden=64,
                 pad_mode="circular_y_zero_z", init_scale=0.05, s_max=2.0):
        super().__init__()
        if channels < 2:
            raise ValueError("a channel split needs at least 2 components; "
                             "with 1 component a squeeze would be required, "
                             "and that breaks y-equivariance")
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                Invertible1x1(channels, init_scale),
                AffineCoupling(channels, hidden, pad_mode, init_scale, s_max,
                               flip=bool(i % 2))])
            for i in range(n_blocks)])
        self.channels, self.n_blocks = channels, n_blocks

    def forward(self, x):
        for mix, cpl in self.blocks:
            x = cpl(mix(x))
        return x

    def inverse(self, y):
        for mix, cpl in reversed(self.blocks):
            y = mix.inverse(cpl.inverse(y))
        return y

    # ---- diagnostics, named as the channel trainer expects ----
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
        was = self.training
        self.eval()
        d = self.double()
        e64 = float((d.inverse(d(x.double())) - x.double()).abs().max())
        self.float()
        e32 = float((self.inverse(self(x)) - x).abs().max())
        self.train(was)
        return {"verified_float64": e64, "train_path_float32": e32}

    @torch.no_grad()
    def equivariance_error(self, x, shifts=(1, 2, 3, 7)):
        """max relative error of f(shift(x)) vs shift(f(x)) over y.

        Odd shifts are included deliberately: a squeeze-based flow passes even
        shifts and fails odd ones, so testing only shift=2 would hide exactly
        the failure this design avoids.
        """
        worst = 0.0
        for n in shifts:
            a = self(torch.roll(x, n, dims=-2))
            b = torch.roll(self(x), n, dims=-2)
            worst = max(worst, float((a - b).abs().max()
                                     / b.abs().max().clamp(min=1e-12)))
        return worst


class CouplingFINE_YZ(IResNetFINE_YZ):
    """IResNetFINE_YZ with f_theta swapped for a coupling flow.

    Subclassed rather than reimplemented so the bottleneck -- band ranks,
    buffers, `init_subspace`, `update_subspace`, the checkpoint resize hook --
    is the shipped code unchanged, and only the transform differs.
    """

    def __init__(self, ny, nz, dof, channels=3, n_blocks=8, hidden=64,
                 pad_mode="circular_y_zero_z", init_scale=0.05, s_max=2.0,
                 bottleneck="banded"):
        # a minimal iResNet is built and then discarded; that costs a few
        # thousand parameters of construction and buys an identical bottleneck
        super().__init__(ny, nz, dof, n_blocks=1, hidden=8, channels=channels,
                         pad_mode=pad_mode, bottleneck=bottleneck)
        self.f = CouplingFlowYZ(channels=channels, n_blocks=n_blocks,
                                hidden=hidden, pad_mode=pad_mode,
                                init_scale=init_scale, s_max=s_max)

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
