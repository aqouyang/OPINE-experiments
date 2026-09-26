"""
Fixed-z x-y plane dataset for Re_tau=180 turbulent channel flow.

Reduced-order modeling of homogeneous streamwise-spanwise planes extracted
from three-dimensional DNS.  One sample = u'(x, y; z_j) at a fixed
wall-normal location z_j.

Physical layout (from DNS README):
    DNS field shape: (Nx=256, Ny=128, Nz=64), Fortran order, float64 LE
    field[ix, iy, iz]:
        ix: streamwise x, periodic, Lx = 2*pi, 256 pts
        iy: spanwise   y, periodic, Ly = pi,   128 pts
        iz: wall-normal z, bounded, Lz = 1     (half-channel, wall at z=0)

    Half-channel: wall at z=0, centreline at z=Lz=1.
    LESGO uniform grid: z_uv[iz] = (iz + 0.5) * dz, dz = 1/64.
    z+ = z * Re_tau = z * 180.

One sample: u'(x, y; z_j) = u(x, y, z_j, t) - U(z_j)
    where U(z_j) is the mean streamwise velocity at z_j, computed from
    training timesteps only.

Normalization: zero-centred, per-plane RMS normalization.
    u_hat = u' / sigma   where sigma = std(u') over training planes at z_j.
    This preserves zero mean and unit variance.

Periodic Fourier downsampling applies only to x and y.
"""

import os
import json
import numpy as np
import torch
import torch.fft as fft
from torch.utils.data import Dataset

# ---- DNS grid constants ----
NX_FULL, NY_FULL, NZ = 256, 128, 64
FIELD_SHAPE = (NX_FULL, NY_FULL, NZ)
FIELD_BYTES = NX_FULL * NY_FULL * NZ * 8  # float64
RE_TAU = 180.0
DZ = 1.0 / NZ  # uniform grid spacing
LX, LY, LZ = 2.0 * np.pi, np.pi, 1.0


def z_coordinate(iz):
    """Physical z coordinate of wall-normal index iz (LESGO uv-grid)."""
    return (iz + 0.5) * DZ


def z_plus(iz):
    """Wall-normal coordinate in wall units."""
    return z_coordinate(iz) * RE_TAU


def all_z_coordinates():
    """Return arrays of z and z+ for all 64 wall-normal positions."""
    iz = np.arange(NZ)
    z = (iz + 0.5) * DZ
    zp = z * RE_TAU
    return z, zp


# ---- I/O ----

def step_path(db_root, component, step):
    return os.path.join(db_root, "fields", f"{component}_velocity.{step:08d}")


def load_field(db_root, component, step):
    """Load one full 3D snapshot; returns float64 array (256, 128, 64)."""
    path = step_path(db_root, component, step)
    return np.fromfile(path, dtype="<f8").reshape(FIELD_SHAPE, order="F")


def discover_timesteps(db_root, stride=10):
    """Return sorted timestep numbers, subsampled by stride."""
    fields_dir = os.path.join(db_root, "fields")
    steps = sorted(
        int(fn.split(".")[-1])
        for fn in os.listdir(fields_dir)
        if fn.startswith("u_velocity.")
    )
    if stride > 1:
        steps = steps[::stride]
    return np.array(steps, dtype=np.int64)


def split_timesteps(steps, train_frac=0.7, val_frac=0.15, seed=42):
    """Deterministic temporal split: shuffle by seed, then partition."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(steps))
    n_train = int(len(steps) * train_frac)
    n_val = int(len(steps) * val_frac)
    train = np.sort(steps[idx[:n_train]])
    val = np.sort(steps[idx[n_train:n_train + n_val]])
    test = np.sort(steps[idx[n_train + n_val:]])
    return train, val, test


# ---- Statistics (training data only) ----

def compute_mean_profile(db_root, steps, component="u"):
    """Mean profile U(z) averaged over x, y, and training timesteps.
    Returns float64 array of shape (Nz,)."""
    accum = np.zeros(NZ, dtype=np.float64)
    for step in steps:
        field = load_field(db_root, component, step)
        accum += field.mean(axis=(0, 1))
    return accum / len(steps)


def compute_plane_std(db_root, steps, mean_profile, iz, component="u",
                      max_steps=200):
    """Standard deviation of u'(x,y;z_j) over training planes at z_j.

    Used for zero-centred RMS normalization: u_hat = u' / sigma.
    """
    rng = np.random.RandomState(0)
    subset = steps if len(steps) <= max_steps else rng.choice(
        steps, max_steps, replace=False)
    sum_sq = 0.0
    count = 0
    for step in subset:
        field = load_field(db_root, component, step)
        plane = field[:, :, iz] - mean_profile[iz]
        sum_sq += float(np.sum(plane ** 2))
        count += plane.size
    return float(np.sqrt(sum_sq / count))


# ---- Fourier downsampling (periodic x, y only) ----

def fourier_downsample_xy(plane, nx_out, ny_out):
    """Downsample a 2D periodic plane by Fourier truncation.

    plane: (Nx, Ny) real array
    Returns: (nx_out, ny_out) real array

    Only low-frequency modes are retained. Exact for bandlimited signals.
    """
    Nx, Ny = plane.shape
    if nx_out == Nx and ny_out == Ny:
        return plane.copy()
    assert nx_out <= Nx and ny_out <= Ny

    F = np.fft.rfft2(plane)  # (Nx, Ny//2+1)
    ny_half_out = ny_out // 2 + 1
    F_trunc = np.zeros((nx_out, ny_half_out), dtype=F.dtype)

    kx_pos = nx_out // 2
    kx_neg = nx_out - kx_pos
    F_trunc[:kx_pos, :] = F[:kx_pos, :ny_half_out]
    F_trunc[kx_pos:, :] = F[Nx - kx_neg:, :ny_half_out]

    result = np.fft.irfft2(F_trunc, s=(nx_out, ny_out))
    scale = (nx_out * ny_out) / (Nx * Ny)
    return result * scale


# ---- Dataset ----

class ChannelFlowXYDataset(Dataset):
    """Fixed-z x-y plane dataset for channel flow DNS.

    Each sample is one x-y plane u'(x, y; z_j) at a fixed wall-normal
    position, from one timestep.

    Shape: (Nx_ds, Ny_ds) float32.
    Normalization: u_hat = u' / sigma  (zero-centred, unit RMS).

    Parameters
    ----------
    db_root : str
        Path to DNS database root.
    steps : array of int
        Timestep numbers to include.
    mean_profile : ndarray (Nz,)
        Mean U(z) from training timesteps.
    iz : int
        Wall-normal index (0..63).
    plane_std : float
        Standard deviation of u' at this z for normalization.
    nx_out, ny_out : int
        Downsampled periodic resolution.
    component : str
        Velocity component.
    preload : bool
        Load all planes into memory at init.
    """

    def __init__(self, db_root, steps, mean_profile, iz, plane_std,
                 nx_out=256, ny_out=128, component="u", preload=False):
        self.db_root = db_root
        self.steps = np.asarray(steps, dtype=np.int64)
        self.mean_profile = np.asarray(mean_profile, dtype=np.float64)
        self.iz = int(iz)
        self.plane_std = float(plane_std)
        self.component = component
        self.nx_out = nx_out
        self.ny_out = ny_out
        self.n_steps = len(self.steps)

        self._preloaded = None
        if preload:
            self._preload_all()

    def _process_field(self, field):
        """Extract plane, subtract mean, downsample, normalize."""
        plane = field[:, :, self.iz] - self.mean_profile[self.iz]
        if self.nx_out < NX_FULL or self.ny_out < NY_FULL:
            plane = fourier_downsample_xy(plane, self.nx_out, self.ny_out)
        return (plane / self.plane_std).astype(np.float32)

    def _preload_all(self):
        data = np.empty((self.n_steps, self.nx_out, self.ny_out),
                        dtype=np.float32)
        for i, step in enumerate(self.steps):
            field = load_field(self.db_root, self.component, step)
            data[i] = self._process_field(field)
        self._preloaded = data

    def __len__(self):
        return self.n_steps

    def __getitem__(self, idx):
        if self._preloaded is not None:
            return torch.from_numpy(self._preloaded[idx].copy())
        field = load_field(self.db_root, self.component,
                           int(self.steps[idx]))
        return torch.from_numpy(self._process_field(field))

    @property
    def sample_shape(self):
        return (self.nx_out, self.ny_out)

    def denormalize(self, x_hat):
        """Convert normalized x_hat back to physical fluctuation u'."""
        return x_hat * self.plane_std


# ---- Spectral mask and DOF counting ----

def build_xy_truncation_mask(nx, ny, kx_keep, ky_keep):
    """Boolean mask for rfft2 output selecting low-frequency x,y modes.

    Conjugate-consistent: retains kx in {0..kx_keep-1} (positive) and
    their conjugates {nx-kx_keep+1..nx-1} (negative), so that the mask
    respects F[kx,0] = conj(F[nx-kx,0]) for ky=0 and ky=ny//2.

    kx_keep: number of non-negative kx modes to retain (0..kx_keep-1).
             Total unique kx = min(2*kx_keep - 1, nx).
    ky_keep: number of ky modes in the half spectrum (0..ky_keep-1).

    Returns: (nx, ny//2+1) boolean tensor
    """
    ny_half = ny // 2 + 1
    mask = torch.zeros(nx, ny_half, dtype=torch.bool)
    kx_pos = min(kx_keep, nx)
    ky_actual = min(ky_keep, ny_half)
    # Positive kx: 0..kx_keep-1
    mask[:kx_pos, :ky_actual] = True
    # Negative kx: conjugates of 1..kx_keep-1 → nx-1, nx-2, ..., nx-kx_keep+1
    n_neg = min(kx_keep - 1, nx - kx_pos)  # exclude DC (already counted)
    if n_neg > 0:
        mask[nx - n_neg:, :ky_actual] = True
    return mask


def count_real_dof(mask, nx, ny):
    """Count independent real DOF from a periodic truncation mask on rfft2.

    For ky=0 and ky=ny//2, the constraint F[kx,ky]=conj(F[nx-kx,ky]) applies
    (both stored in the half-spectrum). Modes whose conjugate partner is also
    retained share 2 DOF per pair; a mode without its partner contributes 2
    independent DOF (unconstrained complex coefficient).

    For 0 < ky < ny//2, every retained mode contributes 2 DOF.
    """
    ny_half = ny // 2 + 1
    real_dof = 0
    for kx in range(nx):
        for ky in range(ny_half):
            if not mask[kx, ky]:
                continue
            if ky == 0 or (ny % 2 == 0 and ky == ny // 2):
                conj_kx = (nx - kx) % nx
                if conj_kx == kx:
                    real_dof += 1  # self-conjugate (real only)
                elif mask[conj_kx, ky]:
                    # Both mode and conjugate retained: count pair once
                    if conj_kx > kx:
                        real_dof += 2
                    # else: already counted by partner
                else:
                    # Conjugate not retained: free complex coefficient
                    real_dof += 2
            else:
                real_dof += 2  # interior ky: two real components
    return real_dof


def find_mask_for_target_dof(nx, ny, target_dof):
    """Search for (kx_keep, ky_keep) giving real DOF closest to target.

    When multiple masks have the same distance to target_dof, prefers
    masks that are more balanced between kx and ky (i.e., aspect ratio
    kx_keep/ky_keep closer to nx/ny).

    Returns: (kx_keep, ky_keep, mask, actual_dof)
    """
    candidates = []
    ny_half = ny // 2 + 1
    target_ratio = nx / ny  # ideal aspect ratio
    for kx in range(1, nx // 2 + 2):
        for ky in range(1, ny_half + 1):
            mask = build_xy_truncation_mask(nx, ny, kx, ky)
            dof = count_real_dof(mask, nx, ny)
            candidates.append((kx, ky, mask, dof))
            if dof > target_dof + max(target_dof // 2, 10):
                break

    # Sort by: (1) distance to target (bucketed to ±2 tolerance),
    #          (2) aspect ratio balance within the same bucket
    def sort_key(c):
        kx, ky, _, dof = c
        dist = abs(dof - target_dof)
        bucket = dist // 3  # group 0-2, 3-5, 6-8, etc.
        ratio = kx / max(ky, 1)
        balance = abs(ratio - target_ratio)
        return (bucket, balance)

    candidates.sort(key=sort_key)
    kx, ky, mask, dof = candidates[0]
    return kx, ky, mask, dof


def verify_dof_numerically(mask, nx, ny, device="cpu"):
    """Verify real DOF by computing numerical rank of the linear map.

    Constructs the matrix mapping real input to retained complex coefficients,
    then counts its rank.
    """
    dim = nx * ny
    ny_half = ny // 2 + 1
    retained = mask.nonzero(as_tuple=False)
    n_retained = len(retained)
    if n_retained == 0:
        return 0

    # Build the map: real input -> retained rfft2 coefficients (real+imag)
    I = torch.eye(dim, device=device, dtype=torch.float64)
    fields = I.reshape(dim, nx, ny)
    F = torch.fft.rfft2(fields, dim=(-2, -1))  # (dim, nx, ny_half)

    # Extract retained modes and stack real/imag
    kx_idx = retained[:, 0]
    ky_idx = retained[:, 1]
    F_retained = F[:, kx_idx, ky_idx]  # (dim, n_retained)
    A = torch.cat([F_retained.real, F_retained.imag], dim=-1)  # (dim, 2*n_retained)

    rank = int(torch.linalg.matrix_rank(A).item())
    return rank


# ---- Preprocessing save/load ----

def save_preprocessing(path, mean_profile, plane_stds, train_steps,
                       val_steps, test_steps, db_root, component, stride,
                       nx_out, ny_out, selected_iz):
    """Save all preprocessing metadata to JSON."""
    z_coords, zp_coords = all_z_coordinates()
    d = {
        "experiment": "channel_retau180_xy",
        "description": ("Reduced-order modeling of homogeneous "
                        "streamwise-spanwise planes from 3D DNS"),
        "coordinate_convention": {
            "x": "streamwise, periodic, Lx=2pi, 256 pts",
            "y": "spanwise, periodic, Ly=pi, 128 pts",
            "z": "wall-normal, bounded, Lz=1, half-channel (wall at z=0), "
                 "64 pts, uniform grid",
            "z_formula": "z[iz] = (iz + 0.5) * dz, dz = 1/64",
            "z_plus_formula": "z+ = z * Re_tau = z * 180",
        },
        "Re_tau": RE_TAU,
        "half_channel": True,
        "grid_type": "uniform",
        "dz": DZ,
        "mean_profile": mean_profile.tolist(),
        "normalization": {
            "type": "zero_centred_rms",
            "formula": "u_hat = (u - U(z)) / sigma(z)",
            "description": "sigma = std(u') over training planes at each z_j",
        },
        "plane_stds": {str(iz): std for iz, std in plane_stds.items()},
        "selected_iz": selected_iz,
        "selected_z": {str(iz): {"z": float(z_coords[iz]),
                                  "z_plus": float(zp_coords[iz])}
                       for iz in selected_iz},
        "train_steps": [int(s) for s in train_steps],
        "val_steps": [int(s) for s in val_steps],
        "test_steps": [int(s) for s in test_steps],
        "db_root": db_root,
        "component": component,
        "velocity_component": "streamwise fluctuation u'",
        "stride": stride,
        "nx_full": NX_FULL,
        "ny_full": NY_FULL,
        "nz": NZ,
        "nx_out": nx_out,
        "ny_out": ny_out,
        "periodic_axes": ["x (streamwise)", "y (spanwise)"],
    }
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def load_preprocessing(path):
    """Load preprocessing metadata from JSON."""
    with open(path) as f:
        d = json.load(f)
    d["mean_profile"] = np.array(d["mean_profile"], dtype=np.float64)
    d["train_steps"] = np.array(d["train_steps"], dtype=np.int64)
    d["val_steps"] = np.array(d["val_steps"], dtype=np.int64)
    d["test_steps"] = np.array(d["test_steps"], dtype=np.int64)
    # Convert plane_stds keys back to int
    d["plane_stds"] = {int(k): v for k, v in d["plane_stds"].items()}
    return d
