"""
Energy-ranked mode selection for joint three-component y-z slices.

Two representations are supported:

**Physical-z** (use_dct_z=False, legacy):
    rfft_y produces (3, n_ky, nz).  Each (c, k_y, z_j) is an entry.
    Problem: each entry affects only ONE z-height.

**Modal** (use_dct_z=True):
    T = FFT_y × DCT_z produces (3, n_ky, n_kz).  Each (c, k_y, k_z)
    is a true global modal coefficient with support across the full field.

DOF accounting (same for both):
    k_y = 0 or k_y = ny/2 (Nyquist, ny even):  1 real DOF
    interior k_y:                                2 real DOF
"""

import math

import numpy as np
import torch


def _make_dct_matrix_np(N):
    """Orthonormal DCT-II matrix (NumPy, float64)."""
    n = np.arange(N)
    k = np.arange(N)
    D = np.cos(math.pi * (2.0 * n[np.newaxis, :] + 1.0)
               * k[:, np.newaxis] / (2.0 * N))
    D[0, :] *= math.sqrt(1.0 / N)
    D[1:, :] *= math.sqrt(2.0 / N)
    return D


def compute_yz_joint_energy_spectrum(dataset, max_samples=500, use_dct_z=False):
    """Mean spectral energy over training samples.

    use_dct_z=False:  |rfft_y[c, k_y, z_j]|^2   (physical z)
    use_dct_z=True:   |FFT_y × DCT_z [c, k_y, k_z]|^2  (modal)

    Returns: (n_comp, n_ky, nz) float64 array.
    """
    n = min(len(dataset), max_samples)
    sample = dataset[0]
    n_comp, ny, nz = sample.shape
    n_ky = ny // 2 + 1

    D = _make_dct_matrix_np(nz) if use_dct_z else None

    accum = np.zeros((n_comp, n_ky, nz), dtype=np.float64)
    for i in range(n):
        x = dataset[i].numpy()  # (3, ny, nz)
        h = np.fft.rfft(x, axis=-2)  # (3, n_ky, nz)
        if use_dct_z:
            # DCT along z for real and imaginary parts
            m_real = np.einsum("kn,...n->...k", D, h.real)
            m_imag = np.einsum("kn,...n->...k", D, h.imag)
            accum += m_real ** 2 + m_imag ** 2
        else:
            accum += np.abs(h) ** 2
    return accum / n


def count_yz_joint_mask_dof(mask, ny):
    """Count total real DOF from a (n_comp, n_ky, nz) boolean mask."""
    n_comp, n_ky, nz = mask.shape
    dof = 0
    for c in range(n_comp):
        for ky in range(n_ky):
            n_entries = int(mask[c, ky].sum())
            if ky == 0 or (ny % 2 == 0 and ky == ny // 2):
                dof += n_entries
            else:
                dof += 2 * n_entries
    return dof


def select_yz_joint_energy_ranked(mean_energy, ny, nz, target_dof, n_comp=3):
    """Greedy energy-ranked selection over all (c, k_y, idx) entries.

    Works identically for physical-z or modal (DCT-z) energy spectra —
    the interpretation of the last axis depends on how mean_energy was
    computed.

    Returns
    -------
    mask : (n_comp, n_ky, nz) bool tensor
    actual_dof : int
    selected : list of (c, k_y, idx, dof_cost, energy)
    """
    n_ky = ny // 2 + 1

    entries = []
    for c in range(n_comp):
        for ky in range(n_ky):
            d = 1 if (ky == 0 or (ny % 2 == 0 and ky == ny // 2)) else 2
            for jz in range(nz):
                entries.append((c, ky, jz, d, float(mean_energy[c, ky, jz])))

    entries.sort(key=lambda e: e[4], reverse=True)

    mask = torch.zeros(n_comp, n_ky, nz, dtype=torch.bool)
    dof = 0
    selected = []

    for c, ky, jz, d, e in entries:
        if dof + d > target_dof:
            continue
        mask[c, ky, jz] = True
        dof += d
        selected.append((c, ky, jz, d, e))
        if dof >= target_dof:
            break

    return mask, dof, selected


def compute_latent_rank_dof(ranks, ny):
    """Total real DOF from per-k_y latent ranks.

    k_y=0 and k_y=ny/2 (Nyquist): each latent coord = 1 real DOF.
    Interior k_y: each latent coord = 2 real DOF (complex).
    """
    n_ky = ny // 2 + 1
    dof = 0
    for ky in range(n_ky):
        r = int(ranks[ky])
        if ky == 0 or (ny % 2 == 0 and ky == ny // 2):
            dof += r
        else:
            dof += 2 * r
    return dof


def uniform_latent_ranks(ny, target_dof):
    """Build uniform ranks (r=1 for all k_y).

    For ny=32: 1 + 2*15 + 1 = 32.
    """
    n_ky = ny // 2 + 1
    ranks = torch.ones(n_ky, dtype=torch.long)
    dof = compute_latent_rank_dof(ranks, ny)
    if dof != target_dof:
        raise ValueError(
            f"uniform r=1 gives {dof} DOF, target is {target_dof}")
    return ranks


def verify_yz_joint_dof_numerically(mask, ny, nz, n_comp=3, use_dct_z=False):
    """Verify real DOF by numerical rank of joint truncation map."""
    from opine_experiments.channel2d.fine_yz_joint import _make_dct_matrix

    dim = n_comp * ny * nz
    I = torch.eye(dim, dtype=torch.float64).reshape(dim, n_comp, ny, nz)
    H = torch.fft.rfft(I, dim=-2)  # (dim, n_comp, n_ky, nz)

    if use_dct_z:
        D = _make_dct_matrix(nz).to(torch.float64)
        H_real = torch.einsum("kn,...n->...k", D, H.real)
        H_imag = torch.einsum("kn,...n->...k", D, H.imag)
        M_modal = torch.complex(H_real, H_imag)
        M_masked = M_modal * mask.to(dtype=torch.complex128).unsqueeze(0)
        # IDCT: D^T
        H_back_real = torch.einsum("kn,...k->...n", D, M_masked.real)
        H_back_imag = torch.einsum("kn,...k->...n", D, M_masked.imag)
        H_masked = torch.complex(H_back_real, H_back_imag)
    else:
        H_masked = H * mask.to(dtype=torch.complex128).unsqueeze(0)

    recon = torch.fft.irfft(H_masked, n=ny, dim=-2)
    M = recon.reshape(dim, dim)
    return int(torch.linalg.matrix_rank(M).item())
