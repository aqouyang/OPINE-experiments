"""
Joint three-component linear baselines for y-z slices.

Official baselines (both allow full u-v-w component mixing):

    JointUnrestrictedPOD -- Official **POD** baseline.
                            Flatten (3,ny,nz)->R^{24576}, global joint PCA.
                            Unrestricted rank-D linear reconstruction.

    JointSymmetryPOD     -- Official **FFT** baseline.
                            FFT_y followed by per-k_y joint POD over
                            (u,v,w,z) in C^{3*nz=192}.  Cross-component
                            covariance blocks (C_uv, C_uw, C_vw) retained.
                            y-translation equivariant; same structural
                            constraint as FINE.
"""

import numpy as np


# =====================================================================
#  JointSymmetryPOD (per-k_y Hermitian covariance)
# =====================================================================

class JointSymmetryPOD:
    """Official FFT baseline: FFT_y + per-k_y joint POD.

    For each non-negative k_y, concatenate 3 components × N_z into C^{3*N_z},
    compute Sigma_{k_y} = (1/N) sum_n x_hat_ky^(n) x_hat_ky^(n)H,
    eigendecompose, and globally rank eigenvalues to allocate D real DOF.

    Cross-component covariance blocks (C_uv, C_uw, C_vw) are retained —
    eigenvectors can freely mix all three velocity components.

    DC/Nyquist: 1 real DOF per eigen-direction.
    Interior k_y: 2 real DOF per complex eigen-direction (re/im pair).

    No cross-k_y mixing.  Same equivariance constraint as FINE.

    Parameters
    ----------
    target_dof : int
        Total real DOF (default 32).
    """

    def __init__(self, target_dof=32, mean_convention="safe"):
        """
        mean_convention : {"safe", "raw"}
            "safe" (default) keeps only the DC and Nyquist per-k_y means and
            zeroes the interior ones.  By y-homogeneity the true interior
            expectation is zero, so the nonzero sample value is finite-sample
            noise phase-locked to the training snapshots -- keeping it breaks
            y-translation equivariance of the reconstruction operator.
            "raw" keeps the sample mean at every k_y (the pre-audit
            convention; NOT y-equivariant).
        """
        if mean_convention not in ("safe", "raw"):
            raise ValueError(f"unknown mean_convention: {mean_convention}")
        self.target_dof = target_dof
        self.mean_convention = mean_convention
        self.ny = None
        self.nz = None
        self.n_comp = 3
        self.n_ky = None
        self.dim = None

        self.mean_spec = None   # (n_ky, dim) complex, per-k_y spectral mean
        self.eigvecs = None     # list of (dim, dim) arrays, per k_y
        self.eigvals = None     # list of (dim,) arrays, per k_y, descending
        self.ranks = None       # dict {k_y: int}
        self.actual_dof = None
        self.shape = None       # (3, ny, nz)

    def _is_real_ky(self, ky):
        return ky == 0 or (self.ny % 2 == 0 and ky == self.ny // 2)

    def fit(self, snapshots):
        """Fit from training data.

        Parameters
        ----------
        snapshots : (N, 3, ny, nz) float64
        """
        N = snapshots.shape[0]
        self.shape = snapshots.shape[1:]
        self.ny = self.shape[1]
        self.nz = self.shape[2]
        self.n_ky = self.ny // 2 + 1
        self.dim = self.n_comp * self.nz

        # rfft along y -> (N, 3, n_ky, nz)
        H = np.fft.rfft(snapshots, axis=-2)
        # -> (N, n_ky, 3*nz)
        H = H.transpose(0, 2, 1, 3).reshape(N, self.n_ky, self.dim)

        # Per-k_y spectral mean (subtract before covariance, restore in recon)
        raw_mean = H.mean(axis=0)  # (n_ky, dim)
        if self.mean_convention == "raw":
            self.mean_spec = raw_mean
        else:  # "safe": zero the interior-k_y means to keep y-equivariance
            self.mean_spec = np.zeros_like(raw_mean)
            self.mean_spec[0] = raw_mean[0]
            if self.ny % 2 == 0:
                self.mean_spec[self.n_ky - 1] = raw_mean[self.n_ky - 1]
        H_c = H - self.mean_spec[np.newaxis]

        # Per-k_y covariance and eigendecomposition
        self.eigvecs = []
        self.eigvals = []
        for ky in range(self.n_ky):
            h = H_c[:, ky, :]  # (N, dim), centered
            if self._is_real_ky(ky):
                C = (h.real.T @ h.real) / N
                vals, vecs = np.linalg.eigh(C)
            else:
                # Full Hermitian covariance
                C = (h.conj().T @ h) / N
                vals, vecs = np.linalg.eigh(C)
            # Descending order
            idx = np.argsort(vals)[::-1]
            self.eigvals.append(vals[idx].copy())
            self.eigvecs.append(vecs[:, idx].copy())

        # Global eigenvalue ranking with DOF cost
        entries = []
        for ky in range(self.n_ky):
            dof_cost = 1 if self._is_real_ky(ky) else 2
            for r, val in enumerate(self.eigvals[ky]):
                entries.append((val, ky, r, dof_cost))
        entries.sort(key=lambda e: e[0], reverse=True)

        # Greedy allocation
        self.ranks = {ky: 0 for ky in range(self.n_ky)}
        dof_used = 0
        for val, ky, r, cost in entries:
            if dof_used + cost > self.target_dof:
                continue
            if r == self.ranks[ky]:
                self.ranks[ky] += 1
                dof_used += cost
            if dof_used >= self.target_dof:
                break
        self.actual_dof = dof_used

    def encode(self, snapshots):
        """Encode to real latent vector of shape (N, target_dof).

        DC/Nyquist coefficients stored as 1 real.
        Interior complex coefficients stored as (re, im) = 2 reals.
        """
        N = snapshots.shape[0]
        H = np.fft.rfft(snapshots, axis=-2)
        H = H.transpose(0, 2, 1, 3).reshape(N, self.n_ky, self.dim)
        H_c = H - self.mean_spec[np.newaxis]

        latent_parts = []
        for ky in range(self.n_ky):
            r = self.ranks[ky]
            if r == 0:
                continue
            V = self.eigvecs[ky][:, :r]
            h = H_c[:, ky]

            if self._is_real_ky(ky):
                coeff = h.real @ V  # (N, r) real
                latent_parts.append(coeff)
            else:
                coeff = h @ V  # (N, r) complex: project onto eigvec columns
                latent_parts.append(coeff.real)
                latent_parts.append(coeff.imag)

        return np.concatenate(latent_parts, axis=1)  # (N, actual_dof)

    def decode(self, latent):
        """Decode from real latent (N, target_dof) -> (N, 3, ny, nz)."""
        N = latent.shape[0]
        H_recon = np.zeros((N, self.n_ky, self.dim), dtype=np.complex128)

        col = 0
        for ky in range(self.n_ky):
            r = self.ranks[ky]
            if r == 0:
                # Restore mean even with no allocated DOF
                H_recon[:, ky] = self.mean_spec[ky]
                continue
            V = self.eigvecs[ky][:, :r]

            if self._is_real_ky(ky):
                coeff = latent[:, col:col + r]  # (N, r) real
                H_recon[:, ky] = coeff @ V.T + self.mean_spec[ky].real
                col += r
            else:
                coeff_re = latent[:, col:col + r]
                coeff_im = latent[:, col + r:col + 2 * r]
                coeff = coeff_re + 1j * coeff_im  # (N, r) complex
                H_recon[:, ky] = coeff @ V.conj().T + self.mean_spec[ky]
                col += 2 * r

        H_recon = H_recon.reshape(N, self.n_ky, self.n_comp, self.nz)
        H_recon = H_recon.transpose(0, 2, 1, 3)
        return np.fft.irfft(H_recon, n=self.ny, axis=-2)

    def reconstruct(self, snapshots):
        return self.decode(self.encode(snapshots))

    def mse(self, snapshots):
        recon = self.reconstruct(snapshots)
        return float(np.mean((snapshots - recon) ** 2))

    def per_component_mse(self, snapshots):
        recon = self.reconstruct(snapshots)
        labels = ("u", "v", "w")
        return {labels[c]: float(np.mean((snapshots[:, c] - recon[:, c]) ** 2))
                for c in range(self.n_comp)}


# =====================================================================
#  JointUnrestrictedPOD (flat SVD, no symmetry constraint)
# =====================================================================

class JointUnrestrictedPOD:
    """Official POD baseline: global joint PCA on full normalised state.

    Flatten (3,ny,nz)->R^D, top-r eigenvectors.
    No y-translation equivariance constraint; allows cross-k_y mixing.
    Unrestricted rank-D linear reconstruction.

    Parameters
    ----------
    dof : int
        Number of retained modes (total real DOF).
    """

    def __init__(self, dof=32):
        self.dof = dof
        self.mean = None
        self.modes = None
        self.eigvals = None
        self.shape = None

    def fit(self, snapshots):
        self.shape = snapshots.shape[1:]
        N = snapshots.shape[0]
        D = int(np.prod(self.shape))
        X = snapshots.reshape(N, D).astype(np.float64)
        self.mean = X.mean(axis=0)
        X_c = X - self.mean
        if N < D:
            # Gram matrix approach: O(N^2*D + N^3), fast when N << D.
            G = X_c @ X_c.T
            eigvals_g, U = np.linalg.eigh(G)
            idx = np.argsort(eigvals_g)[::-1]
            eigvals_g = eigvals_g[idx]
            U = U[:, idx]
            self.eigvals = (eigvals_g / N).copy()
            sigma = np.sqrt(np.maximum(eigvals_g[:self.dof], 0.0))
            modes = X_c.T @ U[:, :self.dof]
            for i in range(self.dof):
                if sigma[i] > 1e-12:
                    modes[:, i] /= sigma[i]
                else:
                    modes[:, i] = 0.0
            self.modes = modes
        else:
            C = (X_c.T @ X_c) / N
            eigvals, eigvecs = np.linalg.eigh(C)
            idx = np.argsort(eigvals)[::-1]
            self.eigvals = eigvals[idx].copy()
            self.modes = eigvecs[:, idx[:self.dof]].copy()

    def encode(self, X):
        shape_batch = X.shape[:-len(self.shape)]
        X_flat = X.reshape(*shape_batch, -1)
        return (X_flat - self.mean) @ self.modes

    def decode(self, Z):
        shape_batch = Z.shape[:-1]
        X_flat = Z @ self.modes.T + self.mean
        return X_flat.reshape(*shape_batch, *self.shape)

    def reconstruct(self, X):
        return self.decode(self.encode(X))

    def mse(self, snapshots):
        recon = self.reconstruct(snapshots)
        return float(np.mean((snapshots - recon) ** 2))

    def per_component_mse(self, snapshots):
        recon = self.reconstruct(snapshots)
        labels = ("u", "v", "w")
        return {labels[c]: float(np.mean((snapshots[:, c] - recon[:, c]) ** 2))
                for c in range(3)}


# Backward-compatible alias
JointPOD = JointUnrestrictedPOD


# ═══════════════════════════════════════════════════════════════════════
#  Canonical POD baseline: global joint POD of the y-symmetrized covariance
# ═══════════════════════════════════════════════════════════════════════

class JointSymmetrizedGlobalPOD:
    """Global joint POD over (u,v,w,y,z) of the y-translation-symmetrized
    empirical covariance.

        C_sym = (1/Ny) sum_tau S_tau C_hat S_tau^{-1}

    This is the canonical POD baseline.  It puts POD under exactly the same
    y-translation-symmetry assumption as the official FFT baseline, so the two
    differ only in conceptual role, not in the operator they realise:

        POD  global joint POD with the known y-translation symmetry enforced
        FFT  Fourier/per-k_y block implementation of the same symmetry-aware
             linear projection, and the fixed compression backbone inside FINE

    Commit f940e42 verified the two agree to machine precision.  The code path
    here is deliberately independent of JointSymmetryPOD: it forms explicit
    real physical modes in R^{3*ny*nz}, sorts one global real eigenvalue
    spectrum, and reconstructs with a single global projector U U^T, whereas
    JointSymmetryPOD stays in the complex per-k_y spectral domain throughout.
    Their agreement is therefore a genuine cross-check.

    Symmetrizing block-diagonalizes the covariance in k_y, so C_sym is built
    through its Fourier blocks rather than as a 24576 x 24576 matrix.  Nothing
    about the result depends on that shortcut.

    Centering uses the same safe-mean convention as the FFT baseline: the
    spectral mean is kept at DC and Nyquist and zeroed at interior k_y.

    The unrestricted/raw JointUnrestrictedPOD is preserved above for
    reference and ablation, but is no longer the canonical POD baseline.
    """

    N_COMP = 3

    def __init__(self, target_dof=32, mean_convention="safe"):
        if mean_convention not in ("safe", "raw"):
            raise ValueError(f"unknown mean_convention: {mean_convention}")
        self.target_dof = target_dof
        self.mean_convention = mean_convention
        self.ny = self.nz = self.n_ky = self.dim = None
        self.mean_phys = None      # (3, ny, nz) safe-mean offset
        self.modes = None          # (n_features, actual_dof) orthonormal
        self.eigvals_real = None   # global real spectrum, descending
        self.ranks = None          # per-k_y retained modes
        self.actual_dof = None
        self.shape = None

    # -- helpers --------------------------------------------------------
    def _is_real_ky(self, ky):
        return ky == 0 or (self.ny % 2 == 0 and ky == self.ny // 2)

    def _safe_mean(self, snapshots):
        H = np.fft.rfft(snapshots, axis=-2)
        raw = H.mean(axis=0)
        if self.mean_convention == "raw":
            m = raw
        else:
            m = np.zeros_like(raw)
            m[:, 0] = raw[:, 0]
            if self.ny % 2 == 0:
                m[:, self.n_ky - 1] = raw[:, self.n_ky - 1]
        return np.fft.irfft(m, n=self.ny, axis=-2)

    def _real_modes(self, ky, vecs, n_take):
        """Real physical modes for the leading n_take eigenvectors at k_y.

        For a unit eigen-direction v of E[X_k X_k^H], the physical content of
        the conjugate pair {k, -k} is (2/sqrt(Ny)) Re[c v exp(i theta)], so
        the real basis directions are Re(v e^{i theta}) and -Im(v e^{i theta}).
        DC and Nyquist contribute a single real direction each.
        """
        y = np.arange(self.ny)
        phase = np.exp(2j * np.pi * ky * y / self.ny)
        modes = []
        for r in range(n_take):
            v = vecs[:, r].reshape(self.N_COMP, self.nz)
            field = v[:, None, :] * phase[None, :, None]
            if self._is_real_ky(ky):
                modes.append(np.real(field) / np.sqrt(self.ny))
            else:
                s = np.sqrt(2.0 / self.ny)
                modes.append(np.real(field) * s)
                modes.append(-np.imag(field) * s)
        return modes

    # -- fit ------------------------------------------------------------
    def fit(self, snapshots):
        """snapshots: (N, 3, ny, nz) float64."""
        N = snapshots.shape[0]
        self.shape = snapshots.shape[1:]
        self.ny, self.nz = snapshots.shape[2], snapshots.shape[3]
        self.n_ky = self.ny // 2 + 1
        self.dim = self.N_COMP * self.nz

        self.mean_phys = self._safe_mean(snapshots)
        Xc = snapshots - self.mean_phys[np.newaxis]

        # Fourier blocks of C_sym under the unitary DFT along y.  The block
        # is E[X_k X_k^H] (column-vector convention), which is the actual
        # operator block; the row-vector Gram would be its conjugate.
        H = np.fft.rfft(Xc, axis=-2) / np.sqrt(self.ny)
        H = H.transpose(0, 2, 1, 3).reshape(N, self.n_ky, self.dim)

        eigvals, eigvecs = [], []
        for ky in range(self.n_ky):
            h = H[:, ky, :]
            B = (h.T @ h.conj()) / N
            B = 0.5 * (B + B.conj().T)
            if self._is_real_ky(ky):
                vals, vecs = np.linalg.eigh(B.real)
                vecs = vecs.astype(np.complex128)
            else:
                vals, vecs = np.linalg.eigh(B)
            idx = np.argsort(vals)[::-1]
            eigvals.append(vals[idx])
            eigvecs.append(vecs[:, idx])

        # one global real spectrum: interior k_y eigenvalues have real
        # multiplicity 2 (the conjugate pair), DC and Nyquist multiplicity 1
        spec = []
        for ky in range(self.n_ky):
            mult = 1 if self._is_real_ky(ky) else 2
            spec.extend(np.repeat(eigvals[ky], mult))
        self.eigvals_real = np.sort(np.asarray(spec))[::-1]

        # greedy real-DOF allocation over the global spectrum
        entries = []
        for ky in range(self.n_ky):
            cost = 1 if self._is_real_ky(ky) else 2
            for r, val in enumerate(eigvals[ky]):
                entries.append((val, ky, r, cost))
        entries.sort(key=lambda e: e[0], reverse=True)
        self.ranks = {ky: 0 for ky in range(self.n_ky)}
        used = 0
        for val, ky, r, cost in entries:
            if used + cost > self.target_dof:
                continue
            if r == self.ranks[ky]:
                self.ranks[ky] += 1
                used += cost
            if used >= self.target_dof:
                break
        self.actual_dof = used

        cols = []
        for ky in range(self.n_ky):
            r = self.ranks[ky]
            if r:
                cols.extend(self._real_modes(ky, eigvecs[ky], r))
        self.modes = np.stack(cols).reshape(
            len(cols), -1).T                        # (n_features, actual_dof)
        return self

    # -- transform ------------------------------------------------------
    def encode(self, snapshots):
        Xc = (snapshots - self.mean_phys[np.newaxis]).reshape(
            snapshots.shape[0], -1)
        return Xc @ self.modes

    def decode(self, latent):
        X = latent @ self.modes.T
        return X.reshape((-1,) + tuple(self.shape)) + self.mean_phys[np.newaxis]

    def reconstruct(self, snapshots):
        return self.decode(self.encode(snapshots))
