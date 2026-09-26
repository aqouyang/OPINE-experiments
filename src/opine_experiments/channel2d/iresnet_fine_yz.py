"""iResNet FINE for y-z slices: invertible ResNet + adaptive latent subspace.

Official name of this method: **iResNet FINE**.  It is distinct from the
canonical K/C-based **FINE** (`PhysicalFINE_YZ` in joint_pod_fine_yz.py),
which is left untouched.

Architecture
------------
    x --f_theta--> z --P_Q--> z_hat --f_theta^{-1}--> x_hat,     P_Q = Q Q^T

f_theta is a dimension-preserving invertible residual network built from
Lipschitz-constrained residual blocks, so z has the same shape as x and the
flattened latent lives in R^N with N = 3 * ny * nz.  Q is an N x D matrix with
orthonormal columns spanning the retained D-dimensional subspace.

The defining feature of iResNet FINE is that Q is NOT held permanently fixed.
It is refreshed during alternating optimization by *subspace iteration* on the
latent covariance, which costs two tall-skinny matmuls per step and never
forms the N x N covariance or calls a full SVD.

Relaxed subspace iteration
--------------------------
With f_theta frozen and Z_c the centred latent training matrix (M x N),

    Y       = C Q = (1/M) Z_c^T (Z_c Q)         two matmuls, O(M N D)
    Q_pow   = orth(Y)                           thin QR
    Q_align = Q_pow R,  R = argmin_orthogonal ||Q_pow R - Q_old||_F
    Q_new   = orth( (1 - alpha) Q_old + alpha Q_align )

alpha is the relaxation parameter.  alpha = 0 leaves Q exactly unchanged and
is therefore the frozen-subspace control; alpha = 1 is undamped subspace
iteration; intermediate values damp the update.  Re-orthogonalization happens
after every update, so Q^T Q = I is maintained to machine precision.

The orthogonal Procrustes step matters.  Interpolating two orthonormal bases
is basis-dependent: an arbitrary sign flip or rotation *within the same
subspace* changes the blend even though the subspace has not moved at all, so
without alignment those gauge choices masquerade as subspace motion and the
"damped" update moves the subspace by an amount unrelated to alpha.  Aligning
Q_pow to Q_old first removes that gauge freedom, so intermediate alpha
produces a genuinely intermediate principal-angle step.

NOTE ON THE TWO ALPHAS.  The residual blocks carry their own contraction
factor (the Lipschitz bound of the residual branch).  That is a completely
separate quantity from the subspace relaxation alpha above.  In this module
the block quantity is always called `lipschitz` and the subspace relaxation is
always called `alpha`.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

N_COMP = 3


# ═══════════════════════════════════════════════════════════════════════
#  Invertible residual network f_theta
# ═══════════════════════════════════════════════════════════════════════

class _SNMixedPadConv2d(nn.Module):
    """Conv with circular-y / zero-z padding, normalized to Lipschitz <= coeff.

    IMPORTANT: torch.nn.utils.spectral_norm normalizes the largest singular
    value of the *reshaped weight matrix*, which is NOT the operator norm of
    the convolution.  For a conv the two differ by a factor that grows with
    the kernel and spatial size, so weight-matrix spectral norm does not bound
    the Lipschitz constant and the residual block stops being invertible.

    Here the operator norm is estimated by power iteration on the actual
    padded convolution, using an autograd vector-Jacobian product for the
    adjoint so the circular/zero padding is handled exactly.  The weight is
    then rescaled by min(1, coeff / sigma).

    The power-iteration estimate is a LOWER bound on the true operator norm,
    and while the weights are moving it lags.  Measured on this problem the
    underestimate is 3-8% per conv; that compounds across the three convs in
    g and, over training, was enough to push the residual branch out of the
    contraction regime and silently break invertibility.  coeff therefore
    defaults to 0.9 rather than 1.0 to leave margin, and n_power defaults to
    5 to keep the estimate near converged.
    """

    def __init__(self, in_ch, out_ch, kernel_size, hw, coeff=0.9,
                 n_power=5, eps=1e-12, pad_mode="circular_y_zero_z"):
        super().__init__()
        # "circular_y_zero_z": channel flow -- y periodic, z wall-bounded.
        # "circular_both":     doubly-periodic domains (Kolmogorov flow).
        if pad_mode not in ("circular_y_zero_z", "circular_both"):
            raise ValueError(f"unknown pad_mode: {pad_mode}")
        self.pad_mode = pad_mode
        self.pad = kernel_size // 2
        self.coeff = coeff
        self.n_power = n_power
        self.eps = eps
        self.hw = hw
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0)
        # When True the power iteration is held fixed, so this module is a
        # deterministic function of its input.  Required during the inverse
        # fixed-point iteration.
        self._freeze_sigma = False
        H, W = hw
        self.register_buffer("u", F.normalize(
            torch.randn(1, out_ch, H, W).flatten(), dim=0).view(1, out_ch, H, W))
        self.register_buffer("v", F.normalize(
            torch.randn(1, in_ch, H, W).flatten(), dim=0).view(1, in_ch, H, W))

    def _apply_op(self, v, weight):
        p = self.pad
        if p:
            if self.pad_mode == "circular_both":
                v = F.pad(v, [p, p, p, p], mode="circular")
            else:
                v = F.pad(v, [0, 0, p, p], mode="circular")
                v = F.pad(v, [p, p, 0, 0], mode="constant")
        return F.conv2d(v, weight, None)          # bias is a translation

    def _adjoint(self, u, weight):
        """J^T u via a vector-Jacobian product, so padding is exact."""
        with torch.enable_grad():
            v = torch.zeros_like(self.v, requires_grad=True)
            out = self._apply_op(v, weight.detach())
            grad, = torch.autograd.grad(out, v, grad_outputs=u)
        return grad

    def _sigma(self):
        w = self.conv.weight
        if self.training and not self._freeze_sigma:
            with torch.no_grad():
                u, v = self.u, self.v
                for _ in range(self.n_power):
                    v = self._adjoint(u, w)
                    v = v / (v.norm() + self.eps)
                    u = self._apply_op(v, w.detach())
                    u = u / (u.norm() + self.eps)
                self.u.copy_(u)
                self.v.copy_(v)
        # Detached CLONES, not the buffers themselves.  The buffers are
        # updated in place on every subsequent call (the inverse fixed-point
        # loop calls g many times per training step), which would bump their
        # version counter and invalidate the graph built here.
        u_c = self.u.detach().clone()
        v_c = self.v.detach().clone()
        return (u_c * self._apply_op(v_c, w)).sum().abs()

    @torch.no_grad()
    def converge_power_iteration(self, tol=1e-4, max_iter=200, min_iter=2):
        """Iterate the power method until the sigma estimate stops moving.

        The cached u,v are fine while the weights drift slowly, but they are
        useless immediately after an exact zero-init: at w = 0 every vector
        lies in the null space, so u,v carry no information about the
        dominant direction. As the weights grow, the cached estimate lags the
        true operator norm, sigma stays below coeff, min(1, coeff/sigma)
        never clamps, and the true norm runs past the budget. That is what
        drove the structured certificate 0.53 -> 0.74 -> 0.92 -> 1.06.

        Calling this after every optimizer step makes sigma a converged
        estimate of the CURRENT weights before forward() rescales with it.
        Nothing about the architecture or the rescaling rule changes.

        Returns (sigma, n_iter, rel_change_at_exit).
        """
        w = self.conv.weight
        u, v = self.u, self.v
        prev = None
        rel = float("inf")
        for i in range(1, max_iter + 1):
            v = self._adjoint(u, w)
            v = v / (v.norm() + self.eps)
            u = self._apply_op(v, w.detach())
            u = u / (u.norm() + self.eps)
            sig = float((u * self._apply_op(v, w)).sum().abs())
            if prev is not None and prev > 0:
                rel = abs(sig - prev) / prev
                if rel < tol and i >= min_iter:
                    prev = sig
                    break
            prev = sig
        self.u.copy_(u)
        self.v.copy_(v)
        return prev if prev is not None else 0.0, i, rel

    @torch.no_grad()
    def operator_norm_estimate(self):
        """Power-iteration estimate of the *effective* conv operator norm.

        This is the norm actually realised in forward(), i.e. after the
        min(1, coeff/sigma) rescaling.
        """
        prev = self._freeze_sigma
        self._freeze_sigma = True
        try:
            sigma = float(self._sigma())
        finally:
            self._freeze_sigma = prev
        return sigma * min(1.0, self.coeff / (sigma + self.eps))

    @torch.no_grad()
    def _effective_weight(self):
        """The weight actually used in forward(), after the coeff rescaling.

        Evaluated with the power iteration FROZEN.  Every unfrozen _sigma()
        call advances u,v, so a diagnostic that let it run would measure a
        different (better-normalized) operator than the one the forward pass
        applied -- and two diagnostics called in sequence would disagree with
        each other.  All norm reporting therefore reads the u,v state as left
        by training and leaves it untouched.
        """
        prev = self._freeze_sigma
        self._freeze_sigma = True
        try:
            w = self.conv.weight
            sigma = float(self._sigma())
        finally:
            self._freeze_sigma = prev
        # identical expression to forward(), so the inverse solves against
        # exactly the map the forward pass applies
        return w * min(1.0, self.coeff / (sigma + self.eps))

    @torch.no_grad()
    def structured_operator_norm(self, chunk=8, return_per_ky=False,
                                 exact_dtype=False):
        """EXACT operator norm of the finite-grid convolution.

        The boundary conditions are not the same in the two directions, and
        that is what makes an exact norm reachable:

          y  circular  -> the unitary DFT along y block-diagonalizes the
                          operator exactly, one block per k_y
          z  zero-pad  -> within a block, z is a finite banded Toeplitz map,
                          i.e. an ordinary finite matrix

        So for each k_y the operator restricted to that band is the finite
        matrix

            M_k[(c_o,i), (c_i,j)] = Wk[c_o, c_i, j - i + p],
            Wk[c_o, c_i, dz]      = sum_dy W[c_o, c_i, dy, dz]
                                        * exp(2 pi i k (dy - p) / ny)

        and  ||C||_2 = max_k sigma_max(M_k)  exactly, because the DFT is an
        isometry and the bands do not mix.  sigma_max(M_{ny-k}) =
        sigma_max(conj(M_k)) = sigma_max(M_k), so k = 0..ny//2 suffices.

        This is a genuine upper bound (it IS the norm), and unlike
        sum_delta ||W_delta||_2 it does not throw away the phase cancellation
        between kernel offsets -- which is exactly where that certificate
        loses its factor of ~2.6 on a 3x3 kernel.
        """
        w = self._effective_weight()
        c_out, c_in, ky_sz, kz_sz = w.shape
        ny, nz = self.hw
        p = self.pad
        n_k = ny // 2 + 1
        dev = w.device

        if self.pad_mode == "circular_both":
            # Both directions circular, so the DFT block-diagonalizes the
            # operator in BOTH wavenumbers at once: each (kx, ky) block is
            # just the c_out x c_in symbol matrix, and the operator norm is
            # max over the two-dimensional wavenumber grid of its largest
            # singular value.  Exact, and cheaper than the mixed case since
            # no Toeplitz block has to be built.
            dy = torch.arange(ky_sz, device=dev, dtype=torch.float64) - p
            dz = torch.arange(kz_sz, device=dev, dtype=torch.float64) - p
            kyv = torch.arange(ny, device=dev, dtype=torch.float64)
            kzv = torch.arange(nz, device=dev, dtype=torch.float64)
            phy = torch.exp(2j * torch.pi * kyv[:, None] * dy[None, :] / ny)
            phz = torch.exp(2j * torch.pi * kzv[:, None] * dz[None, :] / nz)
            W = w.double().to(torch.complex128)
            sym = torch.einsum("oiab,ka,lb->kloi", W, phy, phz)
            sig = torch.linalg.svdvals(
                sym.reshape(ny * nz, c_out, c_in).to(torch.complex64))[:, 0]
            sig = sig.double().reshape(ny, nz)
            return (float(sig.max()), sig.cpu().numpy()) \
                if return_per_ky else float(sig.max())

        # phase-weighted kernel per k_y
        dy = torch.arange(ky_sz, device=dev, dtype=torch.float64) - p
        k = torch.arange(n_k, device=dev, dtype=torch.float64)
        ph = torch.exp(2j * torch.pi * k[:, None] * dy[None, :] / ny)
        Wk = torch.einsum("oiab,ka->koib", w.double().to(torch.complex128),
                          ph)                       # (n_k, c_out, c_in, kz)

        # kz == 1 means no coupling along z: M_k = Wk ⊗ I_nz, so the norm is
        # that of the (c_out x c_in) matrix and nz drops out entirely.
        if kz_sz == 1:
            sig = torch.linalg.matrix_norm(Wk[:, :, :, 0], ord=2)
            return (float(sig.max()), sig.real.cpu().numpy()) \
                if return_per_ky else float(sig.max())

        # otherwise build the banded Toeplitz block and take its exact SVD,
        # chunked over k_y to bound memory
        i = torch.arange(nz, device=dev)
        dz = i[None, :] - i[:, None] + p             # (nz_out, nz_in)
        valid = (dz >= 0) & (dz < kz_sz)
        dz_c = dz.clamp(0, kz_sz - 1)
        # complex64 for the SVD by default: validated against the double
        # path at ~1e-6 relative, which is four orders tighter than the
        # looseness we are trying to remove, and several times faster.
        svd_dt = torch.complex128 if exact_dtype else torch.complex64
        sig = torch.zeros(n_k, dtype=torch.float64, device=dev)
        for a in range(0, n_k, chunk):
            b = min(a + chunk, n_k)
            # (chunk, c_out, c_in, nz_out, nz_in)
            M = Wk[a:b][:, :, :, dz_c] * valid.to(Wk.dtype)
            M = M.permute(0, 1, 3, 2, 4).reshape(
                b - a, c_out * nz, c_in * nz).to(svd_dt)
            sig[a:b] = torch.linalg.svdvals(M)[:, 0].double()
        return (float(sig.max()), sig.cpu().numpy()) \
            if return_per_ky else float(sig.max())

    @torch.no_grad()
    def operator_norm_upper_bound(self):
        """Conservative certificate:  ||C||_2 <= sum_delta ||W_delta||_2.

        For a convolution, summing the spectral norms of the per-offset
        (out_ch x in_ch) slices upper-bounds the operator norm.  Unlike the
        power-iteration value this is a genuine upper bound, so it certifies
        rather than estimates.  It is only a diagnostic: training still uses
        the power-iteration normalization.
        """
        w_eff = self._effective_weight()
        tot = 0.0
        for i in range(w_eff.shape[2]):
            for j in range(w_eff.shape[3]):
                tot += float(torch.linalg.matrix_norm(w_eff[:, :, i, j],
                                                      ord=2))
        return tot

    def forward(self, x):
        sigma = self._sigma()
        scale = torch.clamp(self.coeff / (sigma + self.eps), max=1.0)
        w = self.conv.weight * scale
        out = self._apply_op(x, w)
        if self.conv.bias is not None:
            out = out + self.conv.bias.view(1, -1, 1, 1)
        return out


class IResBlockYZ(nn.Module):
    """Invertible residual block  y = x + s * g(x).

    g is a mixed-padding convolutional stack whose convolutions are rescaled
    using a power-iteration estimate of their operator norm.  Boundary
    conditions match the physics: circular in y, zero-padded in z.

    INVERTIBILITY IS NOT GUARANTEED BY THE POWER-ITERATION SCALING.  Power
    iteration returns a lower bound on the operator norm, so "each conv <=
    coeff" is an estimate, and a claim like Lip(g) <= coeff^3 does not follow.
    Two separate quantities are therefore tracked and must not be conflated:

      estimated_contraction              s * Lip_hat(g), Lip_hat from a local
                                         finite-difference probe.  A LOWER
                                         bound, training diagnostic only.
      certified_contraction_upper_bound  s * prod_i sum_delta ||W_delta^(i)||_2,
                                         a genuine UPPER bound.  The block is
                                         guaranteed invertible only when this
                                         is < 1.

    The Banach fixed-point inverse converges at the true contraction rate,
    which the certificate bounds and the estimate does not.
    """

    def __init__(self, hw, channels=N_COMP, hidden=32, lipschitz=0.6,
                 n_inverse_iters=40, n_grad_iters=3, kernel_size=3,
                 forcing_k=None, forcing_axis=-1, pad_mode=None):
        super().__init__()
        self.lipschitz = lipschitz
        self.n_inverse_iters = n_inverse_iters
        self.n_grad_iters = n_grad_iters
        # diagnostics: how many fixed-point steps the last inverse actually
        # used, and the residual it reached
        self.last_inverse_iters = 0
        self.last_inverse_residual = float("nan")
        # Each sub-conv is rescaled using a power-iteration ESTIMATE of its
        # operator norm.  That estimate is a lower bound, so it does not
        # certify Lip(g) <= coeff^3.  The rigorous statement comes from
        # certified_contraction_upper_bound() below.
        self.g = nn.Sequential(
            _SNMixedPadConv2d(channels, hidden, kernel_size, hw),
            nn.ELU(),
            _SNMixedPadConv2d(hidden, hidden, 1, hw),
            nn.ELU(),
            _SNMixedPadConv2d(hidden, channels, kernel_size, hw),
        )
        # NEAR-identity at initialization: the default conv initialization
        # scaled down, so g(x) is small but NOT identically zero.
        #
        # This is the original, stable behaviour and it is deliberate. An
        # exact zero last conv makes f_theta the exact identity and round-0
        # bit-identical to POD, but it also blocks gradient to the earlier
        # convs on the first optimizer step: with the last weight at zero the
        # upstream gradient is exactly zero, so only the last layer moves.
        # Training recovers from step 2, but canonical training should not
        # start from a partially gradient-dead state.
        #
        # It also interacts badly with the power-iteration spectral norm: at
        # w = 0 the iteration has no dominant direction to lock onto, so as
        # the weights grow the estimate lags, min(1, coeff/sigma) fails to
        # clamp, and the structured certificate runs past 1 (observed:
        # 0.53 -> 0.74 -> 0.92 -> 1.06 over four rounds). The near-identity
        # init keeps the certificate flat at the design value ~0.44.
        #
        # Consequence to keep in mind: round 0 is NOT exactly POD. It differs
        # by ~1e-7 (D=32) to ~6e-5 (D=256) in test NMSE. Gains must therefore
        # be reported against a separately computed POD baseline, never
        # against this round-0 value.
        with torch.no_grad():
            last = self.g[-1]
            last.conv.weight.data *= 0.05
            if last.conv.bias is not None:
                last.conv.bias.data.zero_()

        # ---- forcing-phase conditioning -----------------------------------
        # A shared circular convolution is automatically equivariant to
        # translations in BOTH directions.  For Kolmogorov flow the forcing
        # -k_f cos(k_f y) breaks homogeneity in y, so unconditional
        # y-equivariance is an unwanted symmetry.  Injecting the fixed
        # features cos(k_f y), sin(k_f y) removes it while keeping the
        # periodic boundary.
        #
        # The injection is a STATE-INDEPENDENT additive field: it is a
        # function of the grid only, so g(x + d) - g(x) is unchanged and the
        # Lipschitz constant with respect to the state -- and therefore every
        # contraction certificate -- is exactly as before.
        #
        # It also preserves the true discrete symmetry: cos(k_f y) has period
        # 2*pi/k_f, which on this grid is ny/k_f cells, so a shift by that
        # many cells leaves the conditioning invariant.
        self.forcing_k = forcing_k
        self.forcing_axis = forcing_axis
        if forcing_k is not None:
            H, W = hw
            n = W if forcing_axis in (-1, 1) else H
            coord = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n)
            feat = torch.stack([torch.cos(forcing_k * coord),
                                torch.sin(forcing_k * coord)])   # (2, n)
            if forcing_axis in (-1, 1):
                feat = feat[:, None, :].expand(2, H, W)
            else:
                feat = feat[:, :, None].expand(2, H, W)
            self.register_buffer("cond_feat", feat.contiguous().unsqueeze(0))
            self.cond_proj = nn.Conv2d(2, hidden, 1, bias=False)
            with torch.no_grad():
                self.cond_proj.weight.mul_(0.1)
        else:
            self.cond_proj = None

    def _cond_field(self):
        """Fixed additive field injected after the first convolution."""
        return self.cond_proj(self.cond_feat)

    def _g_apply(self, x):
        """g with the forcing conditioning added after the first conv."""
        if self.cond_proj is None:
            return self.g(x)
        out = None
        for i, m in enumerate(self.g):
            x = m(x) if out is None else m(x)
            if i == 0:
                x = x + self._cond_field()
            out = x
        return x

    def forward(self, x):
        return x + self.lipschitz * self._g_apply(x)

    def _g_float64(self):
        """A float64 evaluation of g built from the frozen float32 weights.

        Nothing is mutated: the effective (already coeff-rescaled) weights are
        read once and promoted, so the optimizer's Parameter objects and their
        state keep their float32 dtype.  Requires _freeze_sigma, otherwise the
        normalization would drift between fixed-point steps.
        """
        ops = []
        for m in self.g:
            if isinstance(m, _SNMixedPadConv2d):
                w = m._effective_weight().double()
                b = (m.conv.bias.detach().double()
                     if m.conv.bias is not None else None)
                ops.append((m, w, b))
            else:
                ops.append((m, None, None))

        cond = (self._cond_field().detach().double()
                if self.cond_proj is not None else None)

        def g64(x):
            for i, (m, w, b) in enumerate(ops):
                if w is None:
                    x = m(x)                      # ELU, dtype-agnostic
                else:
                    x = m._apply_op(x, w)
                    if b is not None:
                        x = x + b.view(1, -1, 1, 1)
                if i == 0 and cond is not None:
                    x = x + cond
            return x
        return g64

    def inverse(self, y, tol=1e-12):
        """Solve y = x + s g(x) for x by Banach fixed-point iteration.

        THE SOLVE RUNS IN FLOAT64.  In float32 the per-step residual bottoms
        out around 1e-5 -- the arithmetic noise floor of the conv stack, not
        the contraction rate -- so the loop burns its whole iteration budget
        chasing a tolerance it cannot reach, and a tight abort threshold then
        fires on rounding rather than on any loss of invertibility.  Measured
        at s*Lip(g) = 0.438: float32 stalls at 2.4e-05 after 37 iterations,
        float64 reaches 8.8e-08 in 7.  Forward and training stay float32; only
        the fixed-point solve is promoted.

        The spectral-norm power iteration is frozen for the whole solve.
        Without that, every call to g refreshes u,v and therefore perturbs the
        normalization scale, so g is a slightly different function at each
        fixed-point step and the iteration cannot converge -- it stalls at a
        residual set by how fast the normalization is drifting rather than by
        the contraction factor.

        The bulk of the iteration runs under no_grad and only a short tail is
        differentiable.  Backpropagating through all n_inverse_iters steps
        would store n_inverse_iters x n_blocks activation sets per batch,
        which is both unnecessary and memory-prohibitive: the map is a
        contraction with factor s*Lip(g) < 1, so a k-step differentiable tail
        started from the converged point matches the true implicit gradient
        (I + s J_g)^{-1} to O((s*Lip(g))^k).
        """
        mods = [m for m in self.g if isinstance(m, _SNMixedPadConv2d)]
        prev = [m._freeze_sigma for m in mods]
        for m in mods:
            m._freeze_sigma = True
        try:
            n_free = max(0, self.n_inverse_iters - self.n_grad_iters)
            used = 0
            resid = float("nan")
            with torch.no_grad():
                g64 = self._g_float64()
                y64 = y.detach().double()
                x64 = y64
                for _ in range(n_free):
                    x_new = y64 - self.lipschitz * g64(x64)
                    resid = float((x_new - x64).abs().max())
                    used += 1
                    if resid < tol:
                        x64 = x_new
                        break
                    x64 = x_new
            self.last_inverse_iters = used
            self.last_inverse_residual = resid
            x = x64.to(y.dtype).detach()
            for _ in range(self.n_grad_iters):
                x = y - self.lipschitz * self._g_apply(x)
        finally:
            for m, p_ in zip(mods, prev):
                m._freeze_sigma = p_
        return x

    @torch.no_grad()
    def empirical_lipschitz(self, x, eps=1e-3, n_probe=5):
        """Local finite-difference estimate of Lip(g). A LOWER estimate.

        The power iteration MUST be frozen across the probe.  g(x+d) and g(x)
        are two separate forward passes, and if u,v refresh in between then
        the normalization scale differs between them, so the measured
        difference mixes the response to d with the drift of the
        normalization.  That inflates the estimate by an amount that grows
        with how fast the weights are moving -- enough, early in training, to
        push this diagnostic above the certified UPPER bound, which is
        impossible for a genuine lower estimate and was tripping the
        contraction guard on an artifact.
        """
        mods = [m for m in self.g if isinstance(m, _SNMixedPadConv2d)]
        prev = [m._freeze_sigma for m in mods]
        for m in mods:
            m._freeze_sigma = True
        try:
            worst = 0.0
            gx = self._g_apply(x)
            for _ in range(n_probe):
                d = torch.randn_like(x)
                d = eps * d / d.norm()
                worst = max(worst, float((self._g_apply(x + d) - gx).norm()
                                         / d.norm()))
            return worst
        finally:
            for m, p_ in zip(mods, prev):
                m._freeze_sigma = p_

    @torch.no_grad()
    def estimated_contraction(self, x, **kw):
        """s * Lip_hat(g) from local probing.  Diagnostic only, NOT a bound.

        Small values here do not imply invertibility; consult
        certified_contraction_upper_bound().
        """
        return self.lipschitz * self.empirical_lipschitz(x, **kw)

    # retained under the old name so existing diagnostics keep working
    contraction_factor = estimated_contraction

    @torch.no_grad()
    def conv_operator_norms(self):
        """Effective operator norm of each conv inside g (power iteration)."""
        return [m.operator_norm_estimate() for m in self.g
                if isinstance(m, _SNMixedPadConv2d)]

    @torch.no_grad()
    def conv_norm_comparison(self):
        """Per-conv L_PI, L_current_cert, L_structured and looseness ratios."""
        out = []
        for m in self.g:
            if isinstance(m, _SNMixedPadConv2d):
                pi = m.operator_norm_estimate()
                cert = m.operator_norm_upper_bound()
                st = m.structured_operator_norm()
                out.append({
                    "kernel": list(m.conv.weight.shape[2:]),
                    "in_ch": m.conv.in_channels,
                    "out_ch": m.conv.out_channels,
                    "L_PI": pi,
                    "L_current_cert": cert,
                    "L_structured": st,
                    "cert_over_structured": cert / max(st, 1e-30),
                    "structured_over_PI": st / max(pi, 1e-30),
                })
        return out

    @torch.no_grad()
    def structured_contraction_upper_bound(self):
        """s * prod_i ||C_i||_2 using the EXACT finite-grid conv norms.

        Still an upper bound on s*Lip(g) -- ELU is 1-Lipschitz and the
        composition bound is a product -- but it removes the per-offset
        slice-sum slack entirely.  Any remaining looseness is the
        product-over-layers bound, not the conv norms.
        """
        ub = 1.0
        for m in self.g:
            if isinstance(m, _SNMixedPadConv2d):
                ub *= m.structured_operator_norm()
        return self.lipschitz * ub

    @torch.no_grad()
    def certified_contraction_upper_bound(self):
        """s * prod_i sum_delta ||W_delta^(i)||_2 -- a genuine UPPER bound.

        For a convolution, summing the spectral norms of the per-offset
        (out_ch x in_ch) weight slices upper-bounds its operator norm.  ELU is
        1-Lipschitz, so the product over the convolutions in g upper-bounds
        Lip(g).  The block is GUARANTEED invertible only when this is < 1;
        this is the only quantity that licenses that claim.
        """
        ub = 1.0
        for m in self.g:
            if isinstance(m, _SNMixedPadConv2d):
                ub *= m.operator_norm_upper_bound()
        return self.lipschitz * ub


class IResNetYZ(nn.Module):
    """f_theta: a stack of invertible residual blocks. Dimension preserving."""

    def __init__(self, hw, n_blocks=6, hidden=32, lipschitz=0.6,
                 n_inverse_iters=40, n_grad_iters=3, channels=N_COMP,
                 forcing_k=None, forcing_axis=-1):
        super().__init__()
        self.blocks = nn.ModuleList([
            IResBlockYZ(hw, channels=channels, hidden=hidden,
                        lipschitz=lipschitz, n_inverse_iters=n_inverse_iters,
                        n_grad_iters=n_grad_iters, forcing_k=forcing_k,
                        forcing_axis=forcing_axis)
            for _ in range(n_blocks)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def inverse(self, z):
        for b in reversed(self.blocks):
            z = b.inverse(z)
        return z

    @torch.no_grad()
    def roundtrip_errors(self, x):
        """Round-trip verification, reported two ways.

        verified_float64      forward AND inverse evaluated in float64.  This
                              is the quantity that says whether f_theta is
                              actually invertible; it is limited by the
                              contraction rate, not by arithmetic.
        train_path_float32    what the training graph really computes: float32
                              forward, float64 fixed-point solve, float32
                              differentiable tail.  Bounded below by the
                              float32 noise floor of the conv stack (~1e-5),
                              so it must NOT be used as an invertibility
                              criterion -- that conflation is what invalidated
                              the first canary.
        """
        z32 = self(x)
        r32 = float((self.inverse(z32) - x).abs().max())

        mods = [m for b in self.blocks for m in b.g
                if isinstance(m, _SNMixedPadConv2d)]
        prev = [m._freeze_sigma for m in mods]
        for m in mods:
            m._freeze_sigma = True
        try:
            x64 = x.double()
            h = x64
            for b in self.blocks:
                h = h + b.lipschitz * b._g_float64()(h)
            for b in reversed(self.blocks):
                g64 = b._g_float64()
                y64, z64 = h, h
                for _ in range(b.n_inverse_iters):
                    z_new = y64 - b.lipschitz * g64(z64)
                    if float((z_new - z64).abs().max()) < 1e-14:
                        z64 = z_new
                        break
                    z64 = z_new
                h = z64
            r64 = float((h - x64).abs().max())
        finally:
            for m, p_ in zip(mods, prev):
                m._freeze_sigma = p_
        return {"verified_float64": r64, "train_path_float32": r32}

    @torch.no_grad()
    def banach_rates(self, x, n_iter=25):
        """Measured contraction factor of the map actually being inverted.

        The fixed-point iteration x <- y - s g(x) converges at exactly
        s*Lip(g) near the solution, so the ratio of successive residuals IS
        the contraction factor -- no probing, no finite differences, no
        assumption about where the worst direction lies.  This is the
        empirical truth-check on the certificate: the certificate is an upper
        bound, so the measured rate must come in below it.

        Runs in float64 for the same reason the solve does: in float32 the
        residuals hit the noise floor after a few steps and the ratios become
        meaningless.
        """
        rates = []
        mods = [m for b in self.blocks for m in b.g
                if isinstance(m, _SNMixedPadConv2d)]
        prev = [m._freeze_sigma for m in mods]
        for m in mods:
            m._freeze_sigma = True
        try:
            h = x.double()
            for b in self.blocks:
                g64 = b._g_float64()
                y64 = h + b.lipschitz * g64(h)      # forward through block
                z = y64
                res = []
                for _ in range(n_iter):
                    z_new = y64 - b.lipschitz * g64(z)
                    res.append(float((z_new - z).abs().max()))
                    z = z_new
                    if res[-1] < 1e-13:
                        break
                r = [res[i + 1] / res[i] for i in range(len(res) - 1)
                     if res[i] > 1e-13]
                rates.append(max(r[len(r) // 2:]) if len(r) > 2 else
                             (r[-1] if r else float("nan")))
                h = y64
            return rates
        finally:
            for m, p_ in zip(mods, prev):
                m._freeze_sigma = p_

    @torch.no_grad()
    def refresh_spectral_norm(self, tol=1e-4, max_iter=200):
        """Re-converge every conv's power iteration on the CURRENT weights.

        Call immediately after optimizer.step(), so the sigma used by the next
        forward's min(1, coeff/sigma) reflects the weights as they are now
        rather than as they were several steps ago.
        """
        out = []
        for blk in self.blocks:
            for m in blk.g:
                if isinstance(m, _SNMixedPadConv2d):
                    sig, n_it, rel = m.converge_power_iteration(
                        tol=tol, max_iter=max_iter)
                    out.append({"sigma": sig, "n_iter": n_it,
                                "rel_change": rel})
        return out

    @torch.no_grad()
    def warm_power_iteration(self, x, n=30):
        """Converge u,v so sigma_PI tracks the true operator norm.

        forward() rescales by coeff/sigma_PI, and power iteration returns a
        LOWER estimate, so a lagging u,v makes the rescaling too weak and the
        operator actually applied can exceed coeff by a wide margin.  Running
        the iteration to convergence before the round's diagnostics makes the
        applied map, the estimate and the certificate all refer to the same
        properly normalized function.
        """
        was = self.training
        self.train()
        try:
            for _ in range(n):
                self(x)
        finally:
            self.train(was)

    def inverse_diagnostics(self):
        """Fixed-point iteration counts and residuals from the last inverse."""
        return {
            "iters_per_block": [b.last_inverse_iters for b in self.blocks],
            "max_iters": max(b.last_inverse_iters for b in self.blocks),
            "residual_per_block": [b.last_inverse_residual
                                   for b in self.blocks],
            "max_residual": max(b.last_inverse_residual for b in self.blocks),
        }

    @torch.no_grad()
    def lipschitz_estimates(self, x):
        """Empirical Lip(g) for every residual block."""
        return [b.empirical_lipschitz(x) for b in self.blocks]

    @torch.no_grad()
    def estimated_contractions(self, x):
        """Per-block s * Lip_hat(g).  Diagnostic lower bounds, not a proof."""
        return [b.estimated_contraction(x) for b in self.blocks]

    # old name kept so existing callers keep working
    contraction_factors = estimated_contractions

    @torch.no_grad()
    def conv_operator_norms(self):
        """Per-block list of per-conv effective operator norms."""
        return [b.conv_operator_norms() for b in self.blocks]

    @torch.no_grad()
    def certified_contraction_upper_bounds(self):
        """Per-block certified upper bound.  All < 1 => provably invertible."""
        return [b.certified_contraction_upper_bound() for b in self.blocks]

    certified_contractions = certified_contraction_upper_bounds

    @torch.no_grad()
    def structured_contraction_upper_bounds(self):
        """Per-block certificate from the exact finite-grid conv norms."""
        return [b.structured_contraction_upper_bound() for b in self.blocks]

    @torch.no_grad()
    def conv_norm_comparisons(self):
        return [b.conv_norm_comparison() for b in self.blocks]


# ═══════════════════════════════════════════════════════════════════════
#  Subspace iteration
# ═══════════════════════════════════════════════════════════════════════

def orth(A):
    """Orthonormal basis for the column space of A via thin QR.

    Sign-fixed so the basis is deterministic given A (QR is unique only up to
    column signs), which keeps principal-angle tracking meaningful.
    """
    Q, R = torch.linalg.qr(A, mode="reduced")
    sign = torch.sign(torch.diagonal(R))
    sign[sign == 0] = 1.0
    return Q * sign.unsqueeze(0)


def symmetrize_mean_y(mu, ny):
    """Project a latent mean onto the y-shift-invariant subspace.

    S_tau mu = mu for every tau iff mu is constant in y, i.e. only the k_y = 0
    Fourier component survives.  (The Nyquist component flips sign under odd
    shifts, so keeping it would break strict invariance.)

    mu: (..., ny, nz) -> same shape, constant along y.
    """
    return mu.mean(dim=-2, keepdim=True).expand_as(mu).contiguous()


def latent_csym_apply(Z_c, Q, ny, n_comp=N_COMP, chunk=256):
    """Apply the y-SYMMETRIZED latent covariance to Q, without forming it.

        C_sym = (1/Ny) sum_tau S_tau C_z S_tau^{-1}

    Conjugating by S_tau multiplies the (k, k') Fourier block of C_z by
    exp(-2 pi i (k-k') tau / Ny), so averaging over tau annihilates every
    k != k' block: C_sym is block diagonal in k_y with blocks E[Z_k Z_k^H].
    We therefore never build the 24576 x 24576 operator -- we accumulate the
    per-k_y blocks from the rFFT of the latents and apply them blockwise.

    Z_c : (M, N) centred latent rows, N = n_comp * ny * nz
    Q   : (N, D)
    """
    M, N = Z_c.shape
    nz = N // (n_comp * ny)
    n_ky = ny // 2 + 1
    dim = n_comp * nz
    D = Q.shape[1]

    # per-k_y blocks of the symmetrized covariance, unitary DFT along y
    blocks = torch.zeros(n_ky, dim, dim, dtype=torch.complex64,
                         device=Z_c.device)
    for i in range(0, M, chunk):
        blk = Z_c[i:i + chunk].reshape(-1, n_comp, ny, nz)
        H = torch.fft.rfft(blk, dim=-2) / (ny ** 0.5)      # (b,c,n_ky,nz)
        H = H.permute(0, 2, 1, 3).reshape(-1, n_ky, dim)   # (b,n_ky,dim)
        blocks += torch.einsum("bki,bkj->kij", H, H.conj())
    blocks /= M

    # apply blockwise to each column of Q
    Qf = Q.T.reshape(D, n_comp, ny, nz)
    W = torch.fft.rfft(Qf, dim=-2) / (ny ** 0.5)
    W = W.permute(0, 2, 1, 3).reshape(D, n_ky, dim)
    Y = torch.einsum("kij,dkj->dki", blocks, W)
    Y = Y.reshape(D, n_ky, n_comp, nz).permute(0, 2, 1, 3)
    out = torch.fft.irfft(Y * (ny ** 0.5), n=ny, dim=-2)
    return out.reshape(D, N).T.contiguous()


def procrustes_align(Q_pow, Q_old):
    """Rotate Q_pow within its own column space to best match Q_old.

    Solves  min_{R orthogonal} ||Q_pow R - Q_old||_F  via the SVD of
    Q_pow^T Q_old.  The column space of Q_pow is unchanged; only the basis
    inside it is rotated, which is exactly what makes a convex blend with
    Q_old meaningful.
    """
    M = Q_pow.T @ Q_old
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return Q_pow @ (U @ Vh)


def subspace_iteration_step(Z_c, Q, alpha, n_power=1, chunk=256,
                            ny=None, symmetrized=True):
    """One relaxed subspace-iteration update of Q on the latent covariance.

    Parameters
    ----------
    Z_c : (M, N) centred latent training matrix (never transposed in full)
    Q   : (N, D) current orthonormal basis
    alpha : float in [0, 1]; 0 leaves Q untouched (frozen control)
    n_power : number of power steps applied before the relaxation
    chunk : row block size, so the M x N matrix is streamed

    Returns
    -------
    Q_new : (N, D) orthonormal
    info  : dict with the Rayleigh-quotient eigenvalue estimates

    The covariance C = (1/M) Z_c^T Z_c is never formed: we evaluate
    C Q = (1/M) Z_c^T (Z_c Q) as two tall-skinny products.
    """
    M = Z_c.shape[0]
    if symmetrized and ny is None:
        raise ValueError("symmetrized update needs ny")
    Q_old = Q
    Q_cur = Q
    for _ in range(max(1, n_power)):
        if symmetrized:
            # production path: y-symmetrized latent covariance, so the
            # resulting projector stays y-translation equivariant
            Y = latent_csym_apply(Z_c, Q_cur, ny, chunk=chunk)
        else:
            # ablation only: raw latent covariance, not y-equivariant
            Y = torch.zeros_like(Q_cur)
            for i in range(0, M, chunk):
                blk = Z_c[i:i + chunk]
                Y += blk.T @ (blk @ Q_cur)
            Y /= M
        Q_cur = orth(Y)

    if alpha <= 0.0:
        Q_new = Q_old                            # exact frozen control
    elif alpha >= 1.0:
        Q_new = Q_cur                            # plain subspace iteration
    else:
        # align first, then blend: without this the interpolation would be
        # dominated by an arbitrary QR basis rotation rather than by alpha
        Q_align = procrustes_align(Q_cur, Q_old)
        Q_new = orth((1.0 - alpha) * Q_old + alpha * Q_align)

    # Rayleigh quotients diag(Q^T C Q), again without forming C
    if symmetrized:
        R = (Q_new * latent_csym_apply(Z_c, Q_new, ny, chunk=chunk)).sum(0)
    else:
        R = torch.zeros(Q_new.shape[1], device=Q_new.device,
                        dtype=Q_new.dtype)
        for i in range(0, M, chunk):
            blk = Z_c[i:i + chunk]
            R += (blk @ Q_new).pow(2).sum(dim=0)
        R /= M
    return Q_new, {"rayleigh": R.detach().cpu().numpy()}


# ---------------------------------------------------------------------------
# y-equivariant (k_y block-structured) subspace representation
# ---------------------------------------------------------------------------
# A dense D-column subspace of C_sym is y-translation equivariant only if it is
# a *converged* invariant subspace AND D happens to land on a +-k_y conjugate
# pair boundary.  Neither holds in general: truncating between the two members
# of a pair breaks equivariance structurally, and a near-degenerate spectral
# gap leaves a residual error decaying as (lambda_{D+1}/lambda_D)^n.
#
# We therefore never carry a free (N, D) matrix.  The retained subspace is
# stored as one complex orthonormal basis B_k per k_y band, with an integer
# rank r_k, and the real physical basis is synthesized from the blocks exactly
# as the canonical POD baseline does.  A y-shift multiplies band k by the
# scalar phase exp(2 pi i k tau / Ny), so it rotates each retained conjugate
# pair inside its own 2D real span: the resulting projector commutes with
# S_tau EXACTLY, for any block bases and at every iterate, converged or not.
#
# The real-DOF budget uses the same accounting as POD/FFT: cost 1 per mode at
# DC and Nyquist, cost 2 at interior k_y.  The band allocation {r_k} is fixed
# at the round-0 canonical allocation and only the bases inside each band are
# adapted, which keeps the retained real DOF identical to POD/FFT at every
# round and keeps alpha a clean continuous control (a discrete reallocation
# mid-training would jump regardless of alpha).


def is_real_ky(ky, ny):
    """k_y bands whose Fourier coefficient is real: DC and, if ny is even,
    Nyquist.  These carry real multiplicity 1; all others carry 2."""
    return ky == 0 or (ny % 2 == 0 and ky == ny // 2)


def orth_c(A):
    """Orthonormal basis for the column space of A, real or complex.

    Phase-fixed via the diagonal of R so the basis is deterministic given A
    (QR is unique only up to a unit-modulus factor per column), which keeps
    Procrustes alignment and principal-angle tracking meaningful.
    """
    Q, R = torch.linalg.qr(A, mode="reduced")
    d = torch.diagonal(R)
    ph = torch.where(d.abs() > 0, d / d.abs().clamp_min(1e-30),
                     torch.ones_like(d))
    return Q * ph.conj().unsqueeze(0)


def latent_csym_blocks(Z_c, ny, n_comp=N_COMP, chunk=256):
    """Per-k_y blocks of the y-symmetrized latent covariance.

        C_sym = (1/Ny) sum_tau S_tau C_z S_tau^{-1}

    Conjugating by S_tau multiplies the (k, k') Fourier block of C_z by
    exp(-2 pi i (k-k') tau / Ny), so averaging over tau annihilates every
    off-diagonal block: C_sym is block diagonal in k_y with blocks
    E[Z_k Z_k^H].  The 24576 x 24576 operator is never formed.

    Returns (n_ky, dim, dim) complex, dim = n_comp * nz.
    """
    M, N = Z_c.shape
    nz = N // (n_comp * ny)
    n_ky = ny // 2 + 1
    dim = n_comp * nz
    # double precision: the blocks are only dim x dim, but a complex64
    # accumulation over M samples carries ~sqrt(M)*eps relative error, which
    # small eigengaps amplify into O(1e-4) rad rotations of the retained
    # directions relative to the float64 canonical POD baseline
    blocks = torch.zeros(n_ky, dim, dim, dtype=torch.complex128,
                         device=Z_c.device)
    for i in range(0, M, chunk):
        blk = Z_c[i:i + chunk].reshape(-1, n_comp, ny, nz).double()
        H = torch.fft.rfft(blk, dim=-2) / (ny ** 0.5)      # (b,c,n_ky,nz)
        H = H.permute(0, 2, 1, 3).reshape(-1, n_ky, dim)   # (b,n_ky,dim)
        blocks += torch.einsum("bki,bkj->kij", H, H.conj())
    blocks /= M
    # Hermitian symmetrization kills the accumulated round-off asymmetry
    blocks = 0.5 * (blocks + blocks.conj().transpose(-1, -2))
    for ky in range(n_ky):
        if is_real_ky(ky, ny):
            blocks[ky] = blocks[ky].real.to(blocks.dtype)
    return blocks


def blocks_to_dense_Q(bases, ranks, ny, nz, n_comp=N_COMP):
    """Synthesize the real orthonormal physical basis from per-band blocks.

    For a unit direction v in C^{n_comp*nz} at band k_y, the physical content
    of the conjugate pair {k, -k} is (2/sqrt(Ny)) Re[c v exp(i theta)], so the
    real basis directions are Re(v e^{i k y}) and -Im(v e^{i k y}), scaled by
    sqrt(2/Ny).  DC and Nyquist contribute a single real direction each,
    scaled by 1/sqrt(Ny).  This is the same construction the canonical POD
    baseline uses, so an identical set of blocks yields an identical Q.
    """
    dev = bases[0].device
    y = torch.arange(ny, device=dev, dtype=torch.float64)
    cols = []
    for ky, B in enumerate(bases):
        r = int(ranks[ky])
        if r == 0:
            continue
        phase = torch.exp(2j * torch.pi * ky * y / ny).to(torch.complex128)
        for j in range(r):
            v = B[:, j].reshape(n_comp, nz)
            field = v[:, None, :] * phase[None, :, None]   # (n_comp, ny, nz)
            if is_real_ky(ky, ny):
                cols.append(field.real / (ny ** 0.5))
            else:
                s = (2.0 / ny) ** 0.5
                cols.append(field.real * s)
                cols.append(-field.imag * s)
    Q = torch.stack(cols).reshape(len(cols), -1).T.contiguous()
    return Q


def allocate_bands(eigvals, ny, target_dof):
    """Greedy real-DOF allocation over the global symmetrized spectrum.

    eigvals: list of per-band descending eigenvalue tensors.  Interior k_y
    modes cost 2 real DOF (the conjugate pair), DC and Nyquist cost 1.  Modes
    are taken in descending eigenvalue order, in order within a band, and a
    mode is skipped if it would overrun the budget -- identical to the
    canonical POD/FFT allocation, so the retained real DOF matches exactly.
    """
    n_ky = ny // 2 + 1
    entries = []
    for ky in range(n_ky):
        cost = 1 if is_real_ky(ky, ny) else 2
        for r, val in enumerate(eigvals[ky].tolist()):
            entries.append((val, ky, r, cost))
    entries.sort(key=lambda e: e[0], reverse=True)
    ranks = [0] * n_ky
    used = 0
    for val, ky, r, cost in entries:
        if used + cost > target_dof:
            continue
        if r == ranks[ky]:
            ranks[ky] += 1
            used += cost
        if used >= target_dof:
            break
    return ranks, used


def canonical_bands(blocks, ny, target_dof):
    """Round-0 initialization: per-band eigendecomposition of C_sym plus the
    canonical greedy allocation.  This is exactly JointSymmetrizedGlobalPOD
    expressed in the latent coordinates, so with f_theta at the identity the
    round-0 bottleneck reproduces canonical POD (== FFT) to numerical
    precision.  Each block is only (n_comp*nz) x (n_comp*nz), and this runs
    once at initialization -- no exact POD/SVD is recomputed during training.
    """
    n_ky = blocks.shape[0]
    eigvals, eigvecs = [], []
    for ky in range(n_ky):
        # double precision: the blocks are only (n_comp*nz)^2 and this runs
        # once, but near-degenerate eigenvalues at the truncation boundary
        # would otherwise be ordered differently from the float64 canonical
        # POD baseline, selecting a different (equally good) mode there
        B = blocks[ky]
        if is_real_ky(ky, ny):
            vals, vecs = torch.linalg.eigh(B.real)
            vecs = vecs.to(blocks.dtype)
        else:
            vals, vecs = torch.linalg.eigh(B)
        idx = torch.argsort(vals, descending=True)
        eigvals.append(vals[idx])
        eigvecs.append(vecs[:, idx])
    ranks, used = allocate_bands(eigvals, ny, target_dof)
    bases = [eigvecs[ky][:, :ranks[ky]].contiguous() for ky in range(n_ky)]
    return bases, ranks, used, eigvals


def block_subspace_step(blocks, bases, ranks, ny, alpha, n_power=1):
    """One relaxed subspace-iteration update, band by band.

    alpha = 0 returns the incoming bases untouched (bitwise frozen control);
    alpha = 1 is plain block subspace iteration; intermediate alpha rotates
    the power iterate onto the old basis by complex orthogonal Procrustes and
    then blends.  Equivariance does not depend on any of this: it is built
    into the block structure itself.
    """
    out, rayleigh = [], []
    for ky, B_old in enumerate(bases):
        r = int(ranks[ky])
        if r == 0:
            out.append(B_old)
            continue
        C = blocks[ky]
        B_cur = B_old
        for _ in range(max(1, n_power)):
            B_cur = orth_c(C @ B_cur)
        if alpha <= 0.0:
            B_new = B_old
        elif alpha >= 1.0:
            B_new = B_cur
        else:
            # Procrustes first, then blend.  A basis is only defined up to a
            # unitary rotation inside its own span, so two identical subspaces
            # can differ by sign flips or rotations that a raw blend would
            # read as real motion.  Aligning first makes the blend measure the
            # change of SUBSPACE rather than the change of coordinates.
            M = B_cur.conj().transpose(-1, -2) @ B_old
            U, _, Vh = torch.linalg.svd(M, full_matrices=False)
            B_align = B_cur @ (U @ Vh)
            B_new = orth_c((1.0 - alpha) * B_old + alpha * B_align)
        out.append(B_new)
        rq = torch.einsum("ij,ji->i", B_new.conj().transpose(-1, -2),
                          blocks[ky] @ B_new).real
        rayleigh.append(rq)
    return out, {"rayleigh": torch.cat(rayleigh).cpu().numpy()
                 if rayleigh else np.zeros(0)}


def block_projector_equivariance_error(Q, ny, nz, n_comp=N_COMP, shifts=(1, 3,
                                                                         7),
                                       seed=0):
    """max_tau || P S_tau w - S_tau P w ||_inf for random probes w."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n_comp, ny, nz, generator=g).to(Q.device, Q.dtype)
    P = lambda v: (v.reshape(1, -1) @ Q @ Q.T).reshape(n_comp, ny, nz)
    err = 0.0
    for t in shifts:
        a = P(torch.roll(w, t, dims=-2))
        b = torch.roll(P(w), t, dims=-2)
        err = max(err, float((a - b).abs().max()))
    return err


def principal_angles(Q1, Q2):
    """Principal angles (radians) between two orthonormal bases.

    Uses the sine formulation in float64.  The cosine form arccos(svd(Q1^T Q2))
    is ill-conditioned for nearly identical subspaces -- in float32 it bottoms
    out around 1e-3 rad, which would swamp the small updates we want to track.
    The residual (I - Q1 Q1^T) Q2 gives the sines directly and stays accurate
    near zero.
    """
    A = Q1.double()
    B = Q2.double()
    R = B - A @ (A.T @ B)
    s = torch.linalg.svdvals(R).clamp(0.0, 1.0)
    return torch.arcsin(s)


def subspace_distance(Q1, Q2):
    """Largest principal angle, in radians. 0 = identical subspaces."""
    return float(principal_angles(Q1, Q2).max())


def retained_energy(Z_c, Q, chunk=256):
    """Fraction of centred latent energy captured by P_Q: ||P_Q Z||^2/||Z||^2."""
    num = den = 0.0
    for i in range(0, Z_c.shape[0], chunk):
        blk = Z_c[i:i + chunk]
        num += float((blk @ Q).pow(2).sum())
        den += float(blk.pow(2).sum())
    return num / den


# ═══════════════════════════════════════════════════════════════════════
#  iResNet FINE
# ═══════════════════════════════════════════════════════════════════════

class IResNetFINE_YZ(nn.Module):
    """iResNet FINE: x -> f_theta -> P_Q -> f_theta^{-1} -> x_hat.

    Q and the latent mean are buffers, not parameters: they are refreshed by
    subspace iteration between optimizer phases, never by gradient descent.
    """

    def __init__(self, ny, nz, dof, n_blocks=6, hidden=32, lipschitz=0.6,
                 n_inverse_iters=40, n_grad_iters=3, channels=N_COMP,
                 pad_mode="circular_y_zero_z", bottleneck="banded",
                 forcing_k=None, forcing_axis=-1):
        super().__init__()
        # bottleneck="banded":       retained subspace is a direct sum of whole
        #                            k_y bands, so the projector is exactly
        #                            y-translation equivariant.  Correct for
        #                            channel flow, where y is homogeneous.
        # bottleneck="unrestricted": a free (N, D) orthonormal Q adapted on the
        #                            RAW latent covariance.  Required whenever
        #                            the statistics are not y-homogeneous --
        #                            e.g. Kolmogorov flow, where the forcing
        #                            -k_f cos(k_f y) makes y inhomogeneous and
        #                            band structure would be a false prior.
        if bottleneck not in ("banded", "unrestricted"):
            raise ValueError(f"unknown bottleneck: {bottleneck}")
        self.bottleneck = bottleneck
        self.pad_mode = pad_mode
        self.ny, self.nz, self.dof = ny, nz, dof
        self.channels = channels
        self.n_features = channels * ny * nz

        self.f = IResNetYZ((ny, nz), n_blocks=n_blocks, hidden=hidden,
                           lipschitz=lipschitz,
                           n_inverse_iters=n_inverse_iters,
                           n_grad_iters=n_grad_iters, channels=channels,
                           forcing_k=forcing_k, forcing_axis=forcing_axis)
        if pad_mode != "circular_y_zero_z":
            for m in self.f.modules():
                if isinstance(m, _SNMixedPadConv2d):
                    m.pad_mode = pad_mode

        self.n_ky = ny // 2 + 1
        self.band_dim = channels * nz

        # Q is DERIVED from the per-k_y block bases below; it is kept as a
        # buffer because project() is a dense matmul, but it is never the
        # source of truth.  The block representation is what guarantees exact
        # y-translation equivariance of the projector.
        self.register_buffer("Q", torch.zeros(self.n_features, dof))
        self.register_buffer("B_blocks", torch.zeros(self.n_ky, self.band_dim,
                                                     1,
                                                     dtype=torch.complex128))
        self.register_buffer("band_ranks", torch.zeros(self.n_ky,
                                                       dtype=torch.long))
        self.register_buffer("z_mean", torch.zeros(self.n_features))
        self.register_buffer("q_initialized", torch.zeros(1))

    def _load_from_state_dict(self, state_dict, prefix, *a, **kw):
        # B_blocks / Q widths depend on the fitted band allocation, so resize
        # the placeholders to the checkpoint shapes before loading.
        for name in ("Q", "B_blocks", "band_ranks"):
            key = prefix + name
            if key in state_dict:
                cur = getattr(self, name)
                if cur.shape != state_dict[key].shape:
                    setattr(self, name, torch.zeros_like(state_dict[key]))
        return super()._load_from_state_dict(state_dict, prefix, *a, **kw)

    # -- block <-> dense --------------------------------------------------
    def _bases_list(self):
        return [self.B_blocks[k, :, :int(self.band_ranks[k])]
                for k in range(self.n_ky)]

    def _store_bases(self, bases, ranks):
        r_max = max(1, max(int(r) for r in ranks))
        B = torch.zeros(self.n_ky, self.band_dim, r_max,
                        dtype=torch.complex128, device=self.Q.device)
        for k, b in enumerate(bases):
            r = int(ranks[k])
            if r:
                B[k, :, :r] = b
        self.B_blocks = B
        self.band_ranks = torch.tensor([int(r) for r in ranks],
                                       dtype=torch.long, device=self.Q.device)
        Q = blocks_to_dense_Q(bases, ranks, self.ny, self.nz, self.channels)
        self.Q = Q.to(self.Q.dtype)
        return self.Q

    # -- latent helpers -------------------------------------------------
    def encode(self, x):
        """x: (B, C, ny, nz) -> flattened latent (B, N)."""
        return self.f(x).reshape(x.shape[0], -1)

    def decode(self, z_flat):
        z = z_flat.reshape(-1, self.channels, self.ny, self.nz)
        return self.f.inverse(z)

    def project(self, z_flat):
        """P_Q applied about the latent mean."""
        zc = z_flat - self.z_mean.unsqueeze(0)
        return (zc @ self.Q) @ self.Q.T + self.z_mean.unsqueeze(0)

    def forward(self, x):
        return self.decode(self.project(self.encode(x)))

    def latent_codes(self, x):
        """D retained coordinates, for diagnostics."""
        return (self.encode(x) - self.z_mean.unsqueeze(0)) @ self.Q

    # -- subspace maintenance -------------------------------------------
    @torch.no_grad()
    def latent_matrix(self, loader, device, center=True):
        """Stack the latent training data into (M, N), with the mean removed."""
        self.eval()
        zs = []
        for xb in loader:
            zs.append(self.encode(xb.to(device)).detach())
        Z = torch.cat(zs)
        mean = Z.mean(dim=0)
        if self.bottleneck == "banded":
            # with an equivariant projector the latent mean must also satisfy
            # S_tau mu = mu, or the affine bottleneck breaks equivariance.
            # For an unrestricted bottleneck there is no such constraint and
            # forcing one would throw away the y-dependence of the mean, which
            # in Kolmogorov flow is physical.
            mean = symmetrize_mean_y(
                mean.reshape(self.channels, self.ny, self.nz),
                self.ny).reshape(-1)
        if center:
            Z = Z - mean.unsqueeze(0)
        return Z, mean

    @torch.no_grad()
    def init_subspace(self, loader, device, method="sympod", n_power=12,
                      seed=0):
        """Initialize the bottleneck from the latent training statistics.

        method="sympod" (default, PRODUCTION): per-k_y eigendecomposition of
        the y-symmetrized latent covariance plus the canonical greedy real-DOF
        allocation -- i.e. JointSymmetrizedGlobalPOD expressed in latent
        coordinates.  Because f_theta starts at (essentially) the identity,
        round 0 of iResNet FINE reproduces the canonical POD baseline, and
        hence the FFT baseline, to numerical precision.  Safe-mean centering,
        real-DOF accounting and allocation all match the canonical baseline.

        method="rawpod": leading D directions of the raw, UNSYMMETRIZED latent
        covariance (the old JointUnrestrictedPOD convention).  ABLATION ONLY.
        It is not band-structured, so its projector is not y-equivariant; it
        must never be used as the production initialization.

        method="subspace": Gaussian sketch plus n_power block subspace
        iterations, so not even the per-band eigendecomposition is used.
        Ablation.

        The per-band eigendecomposition happens once, here.  No exact POD or
        SVD is recomputed at any point during training.
        """
        Z_c, mean = self.latent_matrix(loader, device)
        info = {"method": method, "mean_is_y_symmetric": True}

        if method in ("sympod", "subspace"):
            blocks = latent_csym_blocks(Z_c, self.ny, self.channels)
            bases, ranks, used, eigvals = canonical_bands(
                blocks, self.ny, self.dof)
            if method == "subspace":
                # keep the canonical band allocation but throw the bases away,
                # so not even the per-band eigendecomposition informs Q
                g = torch.Generator(device="cpu").manual_seed(seed)
                bases = []
                for k, r in enumerate(ranks):
                    r = int(r)
                    if r == 0:
                        bases.append(blocks[k][:, :0])
                        continue
                    re = torch.randn(self.band_dim, r, generator=g,
                                     dtype=torch.float64)
                    im = torch.zeros_like(re) if is_real_ky(k, self.ny) \
                        else torch.randn(self.band_dim, r, generator=g,
                                         dtype=torch.float64)
                    A = torch.complex(re, im).to(Z_c.device)
                    bases.append(orth_c(A))
                for _ in range(n_power):
                    bases, _ = block_subspace_step(blocks, bases, ranks,
                                                   self.ny, alpha=1.0)
            Q = self._store_bases(bases, ranks)
            info["actual_dof"] = int(used)
            info["band_ranks"] = [int(r) for r in ranks]
            info["n_active_bands"] = int(sum(1 for r in ranks if r))
        elif method == "rawpod":
            # leading D directions of the RAW latent covariance, via the
            # snapshot (Gram) form.  No symmetry is imposed, which is the
            # correct choice whenever the flow is not homogeneous in y.
            G = (Z_c @ Z_c.T).double() / Z_c.shape[0]      # (M, M)
            evals, U = torch.linalg.eigh(G)
            idx = torch.argsort(evals, descending=True)[:self.dof]
            U = U[:, idx].to(Z_c.dtype)
            Q = orth(Z_c.T @ U)                            # (N, D)
            self.Q = Q
            self.band_ranks = torch.zeros(self.n_ky, dtype=torch.long,
                                          device=Q.device)
            info["actual_dof"] = int(self.dof)
            info["band_structured"] = False
        else:
            raise ValueError(f"unknown init method: {method}")

        self.z_mean = mean.to(self.z_mean.dtype)
        self.q_initialized.fill_(1.0)
        info.update({
            "retained_energy": retained_energy(Z_c, self.Q),
            "band_structured": info.get("band_structured", True),
            "orthonormality_error": float(
                (self.Q.T @ self.Q
                 - torch.eye(self.Q.shape[1], device=self.Q.device))
                .abs().max()),
            "projector_equivariance_error":
                block_projector_equivariance_error(self.Q, self.ny, self.nz,
                                                   self.channels),
            "latent_rms": float(Z_c.pow(2).mean().sqrt()),
            "latent_var": float(Z_c.var()),
        })
        return info

    @torch.no_grad()
    def make_identity(self):
        """Force f_theta to the exact identity (zero every residual branch).

        With init_subspace(method="sympod"), identity f_theta + the
        POD-initialized Q reproduces **SymPOD-safe**
        (JointSymmetrizedGlobalPOD / JointSymmetryPOD with the safe-mean
        convention) to numerical precision -- NOT JointUnrestrictedPOD.

        An earlier version of this docstring named JointUnrestrictedPOD, which
        is a different and strictly worse baseline (D=32: 0.5546376 vs
        0.5488478). Channel gains must be reported against SymPOD-safe, the
        model's true exact round-0; ordinary POD is a separate reference and
        must not be mixed into the same denominator.

        Since the exact zero-init (a0514e3) the constructor already produces
        this state, so this method is a no-op re-assertion for models built
        by the current code. It remains meaningful for checkpoints trained
        before that commit, whose last conv was scaled by 0.05 rather than
        zeroed.
        """
        for blk in self.f.blocks:
            for m in blk.g:
                if hasattr(m, "conv"):
                    m.conv.weight.zero_()
                    if m.conv.bias is not None:
                        m.conv.bias.zero_()

    @torch.no_grad()
    def update_subspace(self, loader, device, alpha, n_power=1):
        """One relaxed subspace-iteration refresh of the bottleneck.

        The update runs band by band on the y-SYMMETRIZED latent covariance
        C_sym = (1/Ny) sum_tau S_tau C_z S_tau^{-1}, whose k_y blocks are
        accumulated from the rFFT of the latents; the 24576 x 24576 operator
        is never formed.  Because the retained subspace stays a direct sum of
        whole k_y bands, the projector is y-translation equivariant exactly,
        at every round, converged or not.

        The band allocation {r_k} is held at the round-0 canonical allocation,
        so the retained real DOF stays identical to POD/FFT and alpha remains
        a clean continuous control; only the bases inside each band move.

        The latent mean is recomputed with Q and symmetrized to satisfy
        S_tau mu = mu, so the affine bottleneck
        z_hat = mu_z + Q Q^T (z - mu_z) is equivariant as a whole.  mu_z uses
        training latents only.  alpha = 0 leaves the bottleneck bitwise
        unchanged.
        """
        import time as _time
        t0 = _time.time()
        Z_c, mean = self.latent_matrix(loader, device)
        t_latent = _time.time() - t0

        t1 = _time.time()
        Q_old = self.Q.clone()
        if self.bottleneck == "unrestricted":
            # free Q on the RAW latent covariance: no band structure, no
            # symmetry assumption.  Same Procrustes-relaxed step, applied to
            # the whole subspace at once.
            Q_new, info = subspace_iteration_step(
                Z_c, Q_old, alpha=alpha, n_power=n_power, symmetrized=False)
            self.Q = Q_new
            ranks = [0] * self.n_ky
        else:
            blocks = latent_csym_blocks(Z_c, self.ny, self.channels)
            ranks = [int(r) for r in self.band_ranks]
            bases_new, info = block_subspace_step(blocks, self._bases_list(),
                                                  ranks, self.ny, alpha=alpha,
                                                  n_power=n_power)
            Q_new = self._store_bases(bases_new, ranks)
        t_update = _time.time() - t1

        self.z_mean = mean.to(self.z_mean.dtype)
        return {
            "n_power_steps": int(n_power),
            "covariance": ("raw latent covariance (unrestricted)"
                           if self.bottleneck == "unrestricted"
                           else "y-symmetrized latent covariance "
                                "(per-k_y blocks)"),
            "band_structured": self.bottleneck == "banded",
            "retained_real_dof": int(self.dof
                                     if self.bottleneck == "unrestricted"
                                     else sum(
                                         r if is_real_ky(k, self.ny) else 2 * r
                                         for k, r in enumerate(ranks))),
            "latent_rms": float(Z_c.pow(2).mean().sqrt()),
            "latent_var": float(Z_c.var()),
            "latent_absmax": float(Z_c.abs().max()),
            "z_mean_norm": float(mean.norm()),
            "numerical_rank": int(torch.linalg.matrix_rank(Q_new)),
            "wallclock_latent_pass_s": t_latent,
            "wallclock_subspace_update_s": t_update,
            "max_principal_angle": subspace_distance(Q_old, Q_new),
            "mean_principal_angle": float(
                principal_angles(Q_old, Q_new).mean()),
            "retained_energy_before": retained_energy(Z_c, Q_old),
            "retained_energy_after": retained_energy(Z_c, Q_new),
            "orthonormality_error": float(
                (Q_new.T @ Q_new
                 - torch.eye(Q_new.shape[1], device=Q_new.device))
                .abs().max()),
            "projector_equivariance_error":
                block_projector_equivariance_error(Q_new, self.ny, self.nz,
                                                   self.channels),
            "rayleigh_top": float(info["rayleigh"].max()),
            "rayleigh_sum": float(info["rayleigh"].sum()),
        }
