#!/usr/bin/env python3
"""1-D invertible transforms + a shared FINE wrapper, for the coupling toy.

Every method plugs the SAME bottleneck and the SAME metric around a different
invertible transform, so the only thing that varies is f_theta:

    x -> f -> flatten -> P_Q -> unflatten -> f^{-1} -> x_hat

`Identity` therefore reproduces plain POD exactly, which is the control that
proves the harness is not doing anything on its own.

NO SQUEEZE. The field already has 2 components, so the affine coupling can
split on channels directly (1 | 1). Squeezing would interleave neighbouring
grid points and make the signal jagged, which the toy is specifically designed
to avoid.

RECEPTIVE FIELD. The ideal transform is
    y2(s) = x2(s) - beta * [x1(s - delta)]^2
with delta = 8 grid points. A conditioner can only represent that if it can
see 8 points away, i.e. receptive field >= 17. With kernel k and L conv layers
the field is 1 + L*(k-1), so the default k=5, L=4 gives exactly 17. A 3-tap
stack (field 5) provably cannot express the target no matter how it is
trained, so this is a correctness requirement, not a tuning knob.
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
    sys.path.insert(0, ROOT)


# ───────────────────────── shared pieces ─────────────────────────
class CircConv1d(nn.Module):
    """1-D convolution with circular padding: the domain is periodic."""

    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, padding=0)
        self.k = k

    def forward(self, x):
        p = self.k // 2
        if p:
            x = F.pad(x, (p, p), mode="circular")
        return self.conv(x)

    def operator_norm(self, n):
        """Exact 2-norm of the circular convolution operator on length n.

        A circular conv is block-diagonal in the DFT basis: the symbol at
        frequency f is the (c_out, c_in) matrix W_hat(f), and the operator
        norm is max_f sigma_max(W_hat(f)). Exact, not a bound.
        """
        w = self.conv.weight                       # (c_out, c_in, k)
        pad = torch.zeros(w.shape[0], w.shape[1], n - w.shape[2],
                          device=w.device, dtype=w.dtype)
        wf = torch.fft.fft(torch.cat([w, pad], dim=2), dim=2)
        s = torch.linalg.svdvals(wf.permute(2, 0, 1))
        return s.max()


class Conditioner1D(nn.Module):
    """Conv stack producing (log-scale, translation) for an affine coupling."""

    def __init__(self, c_in, c_out, hidden, k, n_layers, init_scale):
        super().__init__()
        layers, c = [], c_in
        for _ in range(n_layers - 1):
            layers += [CircConv1d(c, hidden, k), nn.GELU()]
            c = hidden
        layers += [CircConv1d(c, 2 * c_out, k)]
        self.net = nn.Sequential(*layers)
        self.receptive_field = 1 + n_layers * (k - 1)
        with torch.no_grad():
            last = self.net[-1].conv
            last.weight.mul_(init_scale)
            last.bias.zero_()

    def forward(self, x):
        return self.net(x).chunk(2, dim=1)


class AffineCoupling1D(nn.Module):
    def __init__(self, channels, hidden, k, n_layers, init_scale, s_max=2.0,
                 flip=False):
        super().__init__()
        self.flip, self.s_max = flip, s_max
        c_a = channels // 2
        c_b = channels - c_a
        if flip:
            c_a, c_b = c_b, c_a
        self.c_a = c_a
        self.net = Conditioner1D(c_a, c_b, hidden, k, n_layers, init_scale)

    def _split(self, x):
        a, b = x[:, :self.c_a], x[:, self.c_a:]
        return (b, a) if self.flip else (a, b)

    def _join(self, a, b):
        return torch.cat([b, a], 1) if self.flip else torch.cat([a, b], 1)

    def forward(self, x):
        a, b = self._split(x)
        s, t = self.net(a)
        return self._join(a, b * torch.exp(self.s_max * torch.tanh(s)) + t)

    def inverse(self, y):
        a, b = self._split(y)
        s, t = self.net(a)
        return self._join(a, (b - t) * torch.exp(-self.s_max * torch.tanh(s)))


class Invertible1x1_1D(nn.Module):
    """LU-parameterized invertible channel mixing. With 2 physical components
    this is genuine cross-component (u <-> v) mixing, not a squeeze artefact."""

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

    def forward(self, x):
        return torch.einsum("oi,bil->bol", self._w().to(x.dtype), x)

    def inverse(self, y):
        wi = torch.linalg.inv(self._w()).to(y.dtype)
        return torch.einsum("oi,bil->bol", wi, y)


# ───────────────────────── transforms ─────────────────────────
class Identity1D(nn.Module):
    """f = I exactly. The FINE wrapper around it IS plain POD."""
    exact_inverse = True

    def forward(self, x):
        return x

    def inverse(self, y):
        return y


class CouplingFlow1D(nn.Module):
    """[ invertible 1x1 + affine coupling ] x n_blocks, alternating splits."""
    exact_inverse = True

    def __init__(self, channels=2, n_blocks=6, hidden=64, kernel=5,
                 cond_layers=4, init_scale=0.05, s_max=2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([Invertible1x1_1D(channels, init_scale),
                           AffineCoupling1D(channels, hidden, kernel,
                                            cond_layers, init_scale, s_max,
                                            flip=bool(i % 2))])
            for i in range(n_blocks)])
        self.receptive_field = 1 + cond_layers * (kernel - 1)

    def forward(self, x):
        for mix, cpl in self.blocks:
            x = cpl(mix(x))
        return x

    def inverse(self, y):
        for mix, cpl in reversed(self.blocks):
            y = mix.inverse(cpl.inverse(y))
        return y


class IResBlock1D(nn.Module):
    """x + s*g(x) with Lip(g) < 1 enforced by exact circular-conv norms."""

    def __init__(self, channels, hidden, k, lipschitz, coeff, n):
        super().__init__()
        self.c1 = CircConv1d(channels, hidden, k)
        self.c2 = CircConv1d(hidden, channels, k)
        self.s, self.coeff, self.n = lipschitz, coeff, n
        with torch.no_grad():
            self.c2.conv.weight.mul_(0.05)     # near-identity, project-standard
            self.c2.conv.bias.zero_()

    def _scaled(self, conv):
        w = conv.conv.weight
        sig = conv.operator_norm(self.n)
        return w * torch.clamp(self.coeff / (sig + 1e-12), max=1.0)

    def weights(self):
        """Normalized weights for one call.

        The exact operator norm costs an FFT plus a batch of SVDs, and the
        weights do not change during a forward or a fixed-point solve.
        Computing them once per call instead of once per g() evaluation takes
        the inverse from 2*iters SVD batches down to two.
        """
        return self._scaled(self.c1), self._scaled(self.c2)

    def g(self, x, w=None):
        w1, w2 = self.weights() if w is None else w
        p = self.c1.k // 2
        h = F.conv1d(F.pad(x, (p, p), mode="circular"), w1, self.c1.conv.bias)
        h = F.elu(h)
        p = self.c2.k // 2
        return F.conv1d(F.pad(h, (p, p), mode="circular"), w2,
                        self.c2.conv.bias)

    def forward(self, x):
        return x + self.s * self.g(x)

    def inverse(self, y, iters, n_grad=3):
        """Banach fixed point. All but the last `n_grad` sweeps run without
        graph, which is the truncated-gradient scheme the production iResNet
        uses -- the discarded terms are O(Lip^n_grad) and it keeps the memory
        and time from scaling with `iters`."""
        w = self.weights()
        x = y
        with torch.no_grad():
            for _ in range(max(0, iters - n_grad)):
                x = y - self.s * self.g(x, w)
        for _ in range(min(n_grad, iters)):
            x = y - self.s * self.g(x, w)
        return x

    def lipschitz_bound(self):
        """s * Lip(g) <= s * coeff^2.

        Each conv weight is rescaled so its EXACT circular operator norm is at
        most `coeff`, and ELU is 1-Lipschitz, so Lip(g) <= coeff^2. The block
        is a contraction whenever this is < 1.
        """
        return float(self.s * self.coeff ** 2)


class IResNet1D(nn.Module):
    """Convolutional iResNet: the 1-D analogue of the Kolmogorov comparator."""
    exact_inverse = False

    def __init__(self, channels=2, n_blocks=6, hidden=64, kernel=5,
                 lipschitz=0.6, coeff=0.9, n=128, inverse_iters=40):
        super().__init__()
        self.blocks = nn.ModuleList([
            IResBlock1D(channels, hidden, kernel, lipschitz, coeff, n)
            for _ in range(n_blocks)])
        self.inverse_iters = inverse_iters

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def inverse(self, y):
        for b in reversed(self.blocks):
            y = b.inverse(y, self.inverse_iters)
        return y

    def contraction_bound(self):
        return max(b.lipschitz_bound() for b in self.blocks)


class ElementwiseFINE1D(nn.Module):
    """Original-FINE structure: element-wise invertible map + diagonal Fourier
    filter, applied per component with shared weights. No cross-spatial and no
    cross-component coupling by construction -- that is the point."""
    exact_inverse = True

    def __init__(self, channels=2, n=128, n_kc=2, hidden=32):
        super().__init__()
        self.n, self.channels = n, channels
        nk = n // 2 + 1
        self.log_mag = nn.ParameterList(
            [nn.Parameter(torch.zeros(nk)) for _ in range(n_kc)])
        self.phase = nn.ParameterList(
            [nn.Parameter(torch.zeros(nk)) for _ in range(n_kc)])
        # element-wise monotone map: x + sum_j w_j tanh(a_j x + b_j), with the
        # w small enough that the derivative stays positive
        self.a = nn.ParameterList(
            [nn.Parameter(torch.randn(hidden) * 0.5) for _ in range(n_kc)])
        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden)) for _ in range(n_kc)])
        self.w = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden)) for _ in range(n_kc)])
        self.n_kc = n_kc

    def _k(self, x, i, inverse=False):
        a, b = self.a[i], self.b[i]
        w = 0.5 * torch.tanh(self.w[i]) / (a.abs() + 1e-3).sum().clamp(min=1.0)

        def fwd(z):
            return z + (w * torch.tanh(a * z.unsqueeze(-1) + b)).sum(-1)
        if not inverse:
            return fwd(x)
        z = x.clone()                      # monotone, so fixed point converges
        for _ in range(30):
            z = z - (fwd(z) - x)
        return z

    def _c(self, x, i, inverse=False):
        Xf = torch.fft.rfft(x, dim=-1)
        m = torch.exp(self.log_mag[i].clamp(-3, 3))
        p = self.phase[i]
        h = m * torch.exp(1j * p)
        Xf = Xf / h if inverse else Xf * h
        return torch.fft.irfft(Xf, n=self.n, dim=-1)

    def forward(self, x):
        for i in range(self.n_kc):
            x = self._c(self._k(x, i), i)
        return x

    def inverse(self, y):
        for i in reversed(range(self.n_kc)):
            y = self._k(self._c(y, i, inverse=True), i, inverse=True)
        return y


# ───────────────────────── the shared FINE wrapper ─────────────────────────
class FINE1D(nn.Module):
    """x -> f -> P_Q -> f^{-1} -> x_hat, with Q a buffer refreshed by the
    canonical subspace iteration (never by gradient)."""

    def __init__(self, transform, channels, n, dof):
        super().__init__()
        self.f = transform
        self.channels, self.n, self.dof = channels, n, dof
        self.n_features = channels * n
        self.register_buffer("Q", torch.zeros(self.n_features, dof))
        self.register_buffer("z_mean", torch.zeros(self.n_features))

    def encode(self, x):
        return self.f(x).reshape(x.shape[0], -1)

    def decode(self, z):
        return self.f.inverse(z.reshape(-1, self.channels, self.n))

    def project(self, z):
        zc = z - self.z_mean.unsqueeze(0)
        return (zc @ self.Q) @ self.Q.T + self.z_mean.unsqueeze(0)

    def forward(self, x):
        return self.decode(self.project(self.encode(x)))

    @torch.no_grad()
    def latent_matrix(self, X, batch=512):
        zs = [self.encode(X[i:i + batch]) for i in range(0, len(X), batch)]
        Z = torch.cat(zs)
        mean = Z.mean(0)
        return Z - mean, mean

    @torch.no_grad()
    def update_subspace(self, X, alpha, n_power=1):
        from opine_experiments.channel2d.iresnet_fine_yz import (
            subspace_iteration_step, retained_energy, principal_angles)
        Z, mean = self.latent_matrix(X)
        Q_old = self.Q.clone()
        Q_new, _ = subspace_iteration_step(Z, Q_old, alpha=alpha,
                                           n_power=n_power, symmetrized=False)
        self.Q, self.z_mean = Q_new, mean.to(self.z_mean.dtype)
        th = principal_angles(Q_old, Q_new)
        return {"max_principal_angle_deg": float(th.max()) * 180 / math.pi,
                "retained_energy_after": retained_energy(Z, Q_new)}

    @torch.no_grad()
    def roundtrip_error(self, x):
        was = self.training
        self.eval()
        d = self.double()
        e = float((d.f.inverse(d.f(x.double())) - x.double()).abs().max())
        self.float()
        self.train(was)
        return e
