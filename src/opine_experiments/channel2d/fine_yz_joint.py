"""
Joint three-component FINE model for y-z slices.

Architecture (full variant):
    x -> K1 -> FFT_y -> A_{k_y} -> IFFT_y -> K2
    -> P_{32}(truncate) ->
    K2^{-1} -> FFT_y -> A_{k_y}^{-1} -> IFFT_y -> K1^{-1}

Architecture (latent_rank variant):
    x -> K1 -> FFT_y -> A_{k_y} -> P(keep first r_{k_y} latent dims)
    -> A_{k_y}^{-1} -> IFFT_y -> K1^{-1}

Key differences from per-component FINE_YZ:
- Input shape: (B, 3, ny, nz) -- three velocity components
- A_{k_y} in GL(3*nz): mixes BOTH across components AND z-profiles
- K1, K2: per-component RealLineRQSpline (shared spline per component)

Truncation modes:
- mask-based (legacy): binary (c, k_y, z_j) entry mask, 32 real DOF
- latent_rank: per-k_y prefix truncation on A's 192-dim latent vector
"""

import math

import torch
import torch.nn as nn
import torch.fft as fft

from opine_experiments.models.one_d.activations import make_kan_layer


def _make_dct_matrix(N):
    """Orthonormal DCT-II matrix of size N.

    D[k, n] = alpha_k * cos(pi * (2n+1) * k / (2N))
    alpha_0 = sqrt(1/N),  alpha_k = sqrt(2/N) for k > 0.

    D is orthogonal: D @ D^T = I, so IDCT = D^T.
    """
    n = torch.arange(N, dtype=torch.float64)
    k = torch.arange(N, dtype=torch.float64)
    D = torch.cos(math.pi * (2.0 * n.unsqueeze(0) + 1.0)
                  * k.unsqueeze(1) / (2.0 * N))
    D[0, :] *= math.sqrt(1.0 / N)
    D[1:, :] *= math.sqrt(2.0 / N)
    return D.float()


class InvertibleLinearZJoint(nn.Module):
    """Per-k_y invertible linear on concatenated 3-component z-profiles.

    For each k_y, operates on vectors of length n_comp * nz.
    A_{k_y} = L_{k_y} @ U_{k_y}, identity at init.
    Inverse via two triangular solves -- no explicit matrix inverse.
    """

    def __init__(self, n_ky, nz, n_comp=3):
        super().__init__()
        self.n_ky = n_ky
        self.nz = nz
        self.n_comp = n_comp
        self.dim = n_comp * nz

        n_off = self.dim * (self.dim - 1) // 2
        self.L_lower = nn.Parameter(torch.zeros(n_ky, n_off))
        self.U_upper = nn.Parameter(torch.zeros(n_ky, n_off))
        self.log_diag = nn.Parameter(torch.zeros(n_ky, self.dim))

        tril = torch.tril_indices(self.dim, self.dim, offset=-1)
        triu = torch.triu_indices(self.dim, self.dim, offset=1)
        self.register_buffer("_tril_r", tril[0])
        self.register_buffer("_tril_c", tril[1])
        self.register_buffer("_triu_r", triu[0])
        self.register_buffer("_triu_c", triu[1])

    def _get_LU(self):
        dev, dt = self.L_lower.device, self.L_lower.dtype
        diag = torch.arange(self.dim, device=dev)

        L = torch.zeros(self.n_ky, self.dim, self.dim, device=dev, dtype=dt)
        L[:, self._tril_r, self._tril_c] = self.L_lower
        L[:, diag, diag] = 1.0

        U = torch.zeros(self.n_ky, self.dim, self.dim, device=dev, dtype=dt)
        U[:, self._triu_r, self._triu_c] = self.U_upper
        U[:, diag, diag] = torch.exp(self.log_diag)
        return L, U

    def forward(self, h):
        """A_{k_y} @ h.  h: (B, n_ky, n_comp*nz) real or complex."""
        L, U = self._get_LU()
        A = L @ U
        if h.is_complex():
            return torch.complex(
                torch.einsum("knm,bkm->bkn", A, h.real),
                torch.einsum("knm,bkm->bkn", A, h.imag))
        return torch.einsum("knm,bkm->bkn", A, h)

    def inverse(self, h):
        """A_{k_y}^{-1} @ h via triangular solves."""
        L, U = self._get_LU()
        if h.is_complex():
            return torch.complex(
                self._tri_solve(L, U, h.real),
                self._tri_solve(L, U, h.imag))
        return self._tri_solve(L, U, h)

    def _tri_solve(self, L, U, h):
        B = h.shape[0]
        h_col = h.unsqueeze(-1)
        L_b = L.unsqueeze(0).expand(B, -1, -1, -1)
        U_b = U.unsqueeze(0).expand(B, -1, -1, -1)
        y = torch.linalg.solve_triangular(
            L_b, h_col, upper=False, unitriangular=True)
        x = torch.linalg.solve_triangular(U_b, y, upper=True)
        return x.squeeze(-1)


class JointFINE_YZ(nn.Module):
    """Joint three-component FINE for y-z slices.

    Variants
    --------
    rqs_only      : K -> P -> K^{-1}
    linear_c_only : FFT -> A -> P -> A^{-1} -> IFFT
    no_k2         : K -> FFT -> A -> P -> A^{-1} -> IFFT -> K^{-1}
    full          : K1 -> FFT -> A -> IFFT -> K2 -> P -> K2^{-1}
                    -> FFT -> A^{-1} -> IFFT -> K1^{-1}
    latent_rank   : K1 -> FFT_y -> A -> P(first r_{k_y} latent dims)
                    -> A^{-1} -> IFFT_y -> K1^{-1}
    """

    VARIANTS = ("rqs_only", "linear_c_only", "no_k2", "full", "latent_rank")
    N_COMP = 3

    def __init__(self, ny, nz, mask=None, variant="full", use_dct_z=False,
                 ranks=None, use_k1=True):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        self.ny = ny
        self.nz = nz
        self.n_ky = ny // 2 + 1
        self.variant = variant
        self.use_dct_z = use_dct_z
        self.use_k1 = use_k1

        if variant == "latent_rank":
            if ranks is None:
                raise ValueError("latent_rank variant requires ranks")
            self.register_buffer("ranks", ranks.long())
            dim = self.N_COMP * nz
            rank_mask = torch.zeros(self.n_ky, dim)
            for ky in range(self.n_ky):
                rank_mask[ky, :ranks[ky]] = 1.0
            self.register_buffer("_rank_mask", rank_mask)
        else:
            if mask is None:
                raise ValueError("non-latent_rank variants require mask")
            self.register_buffer("mask", mask.float())

        if use_dct_z:
            self.register_buffer("_dct_mat", _make_dct_matrix(nz))

        has_k1 = variant in ("rqs_only", "no_k2", "full", "latent_rank")
        if variant == "latent_rank" and not use_k1:
            has_k1 = False
        has_k2 = variant == "full"
        has_A = variant in ("linear_c_only", "no_k2", "full", "latent_rank")

        if has_k1:
            self.k1 = nn.ModuleList([
                make_kan_layer("realline_spline", num_points=20)
                for _ in range(self.N_COMP)])
        if has_k2:
            self.k2 = nn.ModuleList([
                make_kan_layer("realline_spline", num_points=20)
                for _ in range(self.N_COMP)])
        if has_A:
            self.A = InvertibleLinearZJoint(self.n_ky, nz, self.N_COMP)

    # -- helpers --

    def _apply_k(self, k_list, x, inverse=False):
        """Per-component K.  x: (B, 3, ny, nz)."""
        parts = []
        for c in range(self.N_COMP):
            xc = x[:, c]
            parts.append(k_list[c].inverse(xc) if inverse else k_list[c](xc))
        return torch.stack(parts, dim=1)

    def _to_joint(self, H):
        """(B, 3, n_ky, nz) -> (B, n_ky, 3*nz)"""
        B = H.shape[0]
        return H.permute(0, 2, 1, 3).reshape(B, self.n_ky,
                                               self.N_COMP * self.nz)

    def _from_joint(self, H):
        """(B, n_ky, 3*nz) -> (B, 3, n_ky, nz)"""
        B = H.shape[0]
        return (H.reshape(B, self.n_ky, self.N_COMP, self.nz)
                 .permute(0, 2, 1, 3))

    def _dct_z(self, H):
        """Orthonormal DCT-II along z (last dim).  H may be complex."""
        D = self._dct_mat  # (nz, nz)
        if H.is_complex():
            return torch.complex(
                torch.einsum("kn,...n->...k", D, H.real),
                torch.einsum("kn,...n->...k", D, H.imag))
        return torch.einsum("kn,...n->...k", D, H)

    def _idct_z(self, M):
        """IDCT = DCT^T along z (last dim).  M may be complex."""
        D = self._dct_mat  # (nz, nz)
        if M.is_complex():
            return torch.complex(
                torch.einsum("kn,...k->...n", D, M.real),
                torch.einsum("kn,...k->...n", D, M.imag))
        return torch.einsum("kn,...k->...n", D, M)

    def _trunc_spectral(self, H):
        """Mask in spectral space.  H: (B, 3, n_ky, nz).

        If use_dct_z: apply DCT along z before masking, IDCT after.
        """
        if self.use_dct_z:
            M = self._dct_z(H)
            M = M * self.mask.unsqueeze(0)
            return self._idct_z(M)
        return H * self.mask.unsqueeze(0)

    def _trunc_latent_rank(self, a):
        """Per-k_y prefix truncation.  a: (B, n_ky, dim), real or complex."""
        return a * self._rank_mask.unsqueeze(0)

    def _trunc_physical(self, x):
        """rfft_y -> [dct_z ->] mask [-> idct_z] -> irfft_y."""
        H = fft.rfft(x, dim=-2)
        H = self._trunc_spectral(H)
        return fft.irfft(H, n=self.ny, dim=-2)

    # -- forward --

    def forward(self, x):
        """x: (B, 3, ny, nz) -> recon: (B, 3, ny, nz)"""

        if self.variant == "rqs_only":
            h = self._apply_k(self.k1, x)
            h = self._trunc_physical(h)
            return self._apply_k(self.k1, h, inverse=True)

        if self.variant == "linear_c_only":
            H = fft.rfft(x, dim=-2)
            H = self._from_joint(self.A(self._to_joint(H)))
            H = self._trunc_spectral(H)
            H = self._from_joint(self.A.inverse(self._to_joint(H)))
            return fft.irfft(H, n=self.ny, dim=-2)

        if self.variant == "no_k2":
            h = self._apply_k(self.k1, x)
            H = fft.rfft(h, dim=-2)
            H = self._from_joint(self.A(self._to_joint(H)))
            H = self._trunc_spectral(H)
            H = self._from_joint(self.A.inverse(self._to_joint(H)))
            h = fft.irfft(H, n=self.ny, dim=-2)
            return self._apply_k(self.k1, h, inverse=True)

        if self.variant == "latent_rank":
            if self.use_k1:
                h = self._apply_k(self.k1, x)
            else:
                h = x
            H = fft.rfft(h, dim=-2)
            a = self.A(self._to_joint(H))          # (B, n_ky, dim)
            a = self._trunc_latent_rank(a)          # prefix truncation
            H = self._from_joint(self.A.inverse(a))
            h = fft.irfft(H, n=self.ny, dim=-2)
            if self.use_k1:
                return self._apply_k(self.k1, h, inverse=True)
            return h

        # full
        h = self._apply_k(self.k1, x)
        H = fft.rfft(h, dim=-2)
        H = self._from_joint(self.A(self._to_joint(H)))
        h = fft.irfft(H, n=self.ny, dim=-2)
        h = self._apply_k(self.k2, h)
        h = self._trunc_physical(h)
        h = self._apply_k(self.k2, h, inverse=True)
        H = fft.rfft(h, dim=-2)
        H = self._from_joint(self.A.inverse(self._to_joint(H)))
        h = fft.irfft(H, n=self.ny, dim=-2)
        return self._apply_k(self.k1, h, inverse=True)
