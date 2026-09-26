#!/usr/bin/env python3
"""OPINE for the 3-D minimal channel. Direct 3-D analogue of the existing
coupling flow -- Conv2d -> Conv3d, nothing else redesigned.

    x --f_theta--> z --P_Q--> z_hat --f_theta^{-1}--> x_hat
    P_Q(z) = mu_z + Q Q^T (z - mu_z)

Layout is (B, C, z, y, x) = torch (N, C, D, H, W), C = (u, v, w).

NO SQUEEZE. The three velocity components already provide channels, so the
affine coupling splits on u/v/w directly (1|2, alternating). That also keeps
f_theta exactly equivariant under the two homogeneous translations, which a
space-to-depth squeeze would break for odd shifts.

BOUNDARY TREATMENT, per direction:
    x  (dim -1)  circular   streamwise, periodic, Lx = pi
    y  (dim -2)  circular   spanwise,  periodic, Ly = pi/2
    z  (dim -3)  zero pad   wall-normal, NOT periodic: no-slip floor,
                            stress-free lid. Wrapping z would join wall to lid.
No spatial downsampling; the flow is shape preserving.

BOTTLENECK IS UNRESTRICTED, a deliberate departure from the y-z channel runs
that must be recorded. There, `banded` builds the retained subspace out of
whole k_y bands so the projector is exactly y-translation equivariant. That
machinery is one-dimensional in the periodic index; here there are TWO
periodic directions and it does not apply. Extending it to (k_x, k_y) bands
would be a new bottleneck, i.e. a redesign. Consequence: f_theta is exactly
equivariant in x and y, P_Q is not, so the composed operator is not.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)          # append: the repo root holds a stray
                                   # inspect.py that shadows the stdlib

NC, NZ, NY, NX = 3, 64, 32, 32
N_FEAT = NC * NZ * NY * NX


class MixedPadConv3d(nn.Module):
    """3-D conv: circular in x and y, zero-padded in z."""

    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, k, padding=0)
        self.k = k

    def forward(self, v):
        p = self.k // 2
        if p:
            v = F.pad(v, (p, p, p, p, 0, 0), mode="circular")
            v = F.pad(v, (0, 0, 0, 0, p, p), mode="constant")
        return self.conv(v)


class Conditioner3D(nn.Module):
    def __init__(self, c_in, c_out, hidden, init_scale):
        super().__init__()
        self.net = nn.Sequential(
            MixedPadConv3d(c_in, hidden, 3), nn.ReLU(),
            MixedPadConv3d(hidden, hidden, 1), nn.ReLU(),
            MixedPadConv3d(hidden, 2 * c_out, 3))
        with torch.no_grad():
            last = self.net[-1].conv
            last.weight.mul_(init_scale)
            last.bias.zero_()

    def forward(self, v):
        return self.net(v).chunk(2, dim=1)


class AffineCoupling3D(nn.Module):
    def __init__(self, channels, hidden, init_scale, s_max=2.0, flip=False):
        super().__init__()
        self.flip, self.s_max = flip, s_max
        self.p = channels // 2
        c_a = (channels - self.p) if flip else self.p
        c_b = self.p if flip else (channels - self.p)
        self.net = Conditioner3D(c_a, c_b, hidden, init_scale)

    def _split(self, v):
        v1, v2 = v[:, :self.p], v[:, self.p:]
        return (v2, v1) if self.flip else (v1, v2)

    def _join(self, a, b):
        return torch.cat([b, a], 1) if self.flip else torch.cat([a, b], 1)

    def forward(self, v):
        a, b = self._split(v)
        s, t = self.net(a)
        return self._join(a, b * torch.exp(self.s_max * torch.tanh(s)) + t)

    def inverse(self, y):
        a, b = self._split(y)
        s, t = self.net(a)
        return self._join(a, (b - t) * torch.exp(-self.s_max * torch.tanh(s)))


class Invertible1x1_3D(nn.Module):
    """LU-parameterized mixing of the physical u/v/w channels."""

    def __init__(self, channels, init_scale):
        super().__init__()
        w = torch.eye(channels)
        if init_scale > 0:
            w = w + init_scale * torch.randn(channels, channels) / channels
        p, l, u = torch.linalg.lu(w)
        s = torch.diagonal(u).clone()
        self.register_buffer("P", p)
        self.register_buffer("eye", torch.eye(channels))
        self.L = nn.Parameter(l)
        self.U = nn.Parameter(torch.triu(u, 1))
        self.sign_s = nn.Parameter(torch.sign(s), requires_grad=False)
        self.log_s = nn.Parameter(torch.log(torch.abs(s) + 1e-12))

    def _w(self):
        l = torch.tril(self.L, -1) + self.eye
        u = torch.triu(self.U, 1) + torch.diag(self.sign_s
                                               * torch.exp(self.log_s))
        return self.P @ l @ u

    def forward(self, v):
        return torch.einsum("oi,bizyx->bozyx", self._w().to(v.dtype), v)

    def inverse(self, y):
        wi = torch.linalg.inv(self._w()).to(y.dtype)
        return torch.einsum("oi,bizyx->bozyx", wi, y)


class CouplingFlow3D(nn.Module):
    exact_inverse = True

    def __init__(self, channels=NC, n_blocks=8, hidden=32, init_scale=0.05,
                 s_max=2.0):
        super().__init__()
        if channels < 2:
            raise ValueError("needs >= 2 channels to split without a squeeze")
        self.blocks = nn.ModuleList([
            nn.ModuleList([Invertible1x1_3D(channels, init_scale),
                           AffineCoupling3D(channels, hidden, init_scale,
                                            s_max, flip=bool(i % 2))])
            for i in range(n_blocks)])

    def forward(self, v):
        for mix, cpl in self.blocks:
            v = cpl(mix(v))
        return v

    def inverse(self, y):
        for mix, cpl in reversed(self.blocks):
            y = mix.inverse(cpl.inverse(y))
        return y

    @torch.no_grad()
    def roundtrip_errors(self, v):
        was = self.training
        self.eval()
        d = self.double()
        e64 = float((d.inverse(d(v.double())) - v.double()).abs().max())
        self.float()
        e32 = float((self.inverse(self(v)) - v).abs().max())
        self.train(was)
        return {"verified_float64": e64, "train_path_float32": e32}

    @torch.no_grad()
    def equivariance_error(self, v, shifts=(1, 2, 3, 7)):
        """Equivariance of f_theta in the two PERIODIC directions only.

        z is excluded deliberately: it is not periodic, so a shift there is
        not a symmetry of the problem and the test would be meaningless.
        """
        out = {}
        for dim, name in ((-1, "x"), (-2, "y")):
            worst = 0.0
            for n in shifts:
                a = self(torch.roll(v, n, dims=dim))
                b = torch.roll(self(v), n, dims=dim)
                worst = max(worst, float((a - b).abs().max()
                                         / b.abs().max().clamp(min=1e-12)))
            out[name] = worst
        return out


class OPINE3D(nn.Module):
    """f_theta + rank-D orthogonal projection. Q and mu_z are buffers."""

    def __init__(self, dof, channels=NC, n_blocks=8, hidden=32,
                 init_scale=0.05, s_max=2.0):
        super().__init__()
        self.dof, self.channels = dof, channels
        self.shape = (channels, NZ, NY, NX)
        self.n_features = N_FEAT
        self.f = CouplingFlow3D(channels, n_blocks, hidden, init_scale, s_max)
        self.register_buffer("Q", torch.zeros(N_FEAT, dof))
        self.register_buffer("z_mean", torch.zeros(N_FEAT))

    def encode(self, x):
        return self.f(x).reshape(x.shape[0], -1)

    def decode(self, z):
        return self.f.inverse(z.reshape(-1, *self.shape))

    def project(self, z):
        zc = z - self.z_mean.unsqueeze(0)
        return (zc @ self.Q) @ self.Q.T + self.z_mean.unsqueeze(0)

    def forward(self, x):
        return self.decode(self.project(self.encode(x)))

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Identity3D(nn.Module):
    """f = I exactly. OPINE3D around it IS plain POD -- the control that shows
    the harness contributes nothing of its own."""
    exact_inverse = True

    def forward(self, v):
        return v

    def inverse(self, y):
        return y

    @torch.no_grad()
    def roundtrip_errors(self, v):
        return {"verified_float64": 0.0, "train_path_float32": 0.0}

    @torch.no_grad()
    def equivariance_error(self, v, shifts=(1, 2, 3, 7)):
        return {"x": 0.0, "y": 0.0}


# ── streaming subspace maintenance ─────────────────────────────────────
# The y-z code materialises the latent matrix Z (M x N) before updating Q.
# Here N = 196,608, so even a 4,000-sample subset is 3.1 GB and the full
# training split 12.6 GB. Both routines below stream and never form Z.

@torch.no_grad()
def latent_mean(model, loader, device):
    tot, n = torch.zeros(N_FEAT, dtype=torch.float64, device=device), 0
    model.eval()
    for xb in loader:
        z = model.encode(xb.to(device, non_blocking=True)).double()
        tot += z.sum(0); n += z.shape[0]
    return (tot / n).float()


@torch.no_grad()
def streaming_subspace_step(model, loader, device, alpha, n_power=1):
    """One relaxed subspace-iteration step on the latent covariance.

    Same algebra as subspace_iteration_step: Y = C Q with
    C = (1/M) Z_c^T Z_c, formed as two tall-skinny products per batch, then
    orthonormalised and blended with the previous Q at rate alpha. alpha = 0
    leaves Q bitwise unchanged.
    """
    from opine_experiments.channel2d.iresnet_fine_yz import (
        orth, principal_angles)
    mu = latent_mean(model, loader, device)
    model.z_mean = mu
    Q_old = model.Q.clone()
    if alpha <= 0.0:
        return {"max_principal_angle_deg": 0.0, "alpha": 0.0}
    Q_cur = Q_old
    for _ in range(max(1, n_power)):
        Y = torch.zeros(N_FEAT, model.dof, dtype=torch.float64, device=device)
        n = 0
        for xb in loader:
            z = model.encode(xb.to(device, non_blocking=True)).double()
            z -= mu.double().unsqueeze(0)
            Y += z.T @ (z @ Q_cur.double())
            n += z.shape[0]
        Q_cur = orth((Y / n).float())
    if alpha >= 1.0:
        Q_new = Q_cur
    else:
        u, _, vt = torch.linalg.svd(Q_old.T @ Q_cur, full_matrices=False)
        Q_new = orth((1 - alpha) * Q_old + alpha * (Q_cur @ (u @ vt).T))
    model.Q = Q_new
    return {"max_principal_angle_deg":
            float(principal_angles(Q_old, Q_new).max()) * 180 / math.pi,
            "alpha": alpha}


@torch.no_grad()
def init_subspace_pod(model, loader, device, n_power=12, seed=0):
    """Initialise Q by randomised subspace iteration on the latent covariance.

    With f_theta near the identity this is the POD basis of the (normalized)
    training fields, so round 0 reproduces the POD baseline up to the
    near-identity initialisation.
    """
    from opine_experiments.channel2d.iresnet_fine_yz import orth
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.Q = orth(torch.randn(N_FEAT, model.dof, generator=g).to(device))
    model.z_mean = latent_mean(model, loader, device)
    for _ in range(n_power):
        streaming_subspace_step(model, loader, device, alpha=1.0, n_power=1)
    return {"n_power": n_power}
