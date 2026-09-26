"""
Fixed-x y-z slice dataset for Re_tau=180 turbulent channel flow.

One sample = u'(y, z; x_i) at a fixed streamwise position x_i.

Physical layout:
    y: spanwise, PERIODIC, Ly=pi, 128 pts → Fourier-downsample to ny_out
    z: wall-normal, BOUNDED, Lz=1 (half-channel, wall at z=0), 64 pts

Normalization: per-z zero-centred RMS.
    u_hat(y, z_j) = (u(x_i, y, z_j, t) - U(z_j)) / sigma(z_j)
    sigma(z_j) computed from training data over all x, y, and training t.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from opine_experiments.channel2d.xy_planes.dataset import (
    NX_FULL, NY_FULL, NZ,
    load_field, discover_timesteps, split_timesteps,
    compute_mean_profile, z_coordinate, z_plus,
    load_preprocessing,
)


def fourier_downsample_y(plane, ny_out):
    """Fourier-downsample along y (axis 0, periodic).  plane: (ny, nz)."""
    ny = plane.shape[0]
    if ny_out == ny:
        return plane.copy()
    assert ny_out <= ny
    F = np.fft.rfft(plane, axis=0)  # (ny//2+1, nz)
    ny_half_out = ny_out // 2 + 1
    F_trunc = F[:ny_half_out, :]
    return np.fft.irfft(F_trunc, n=ny_out, axis=0) * (ny_out / ny)


def compute_all_plane_stds(db_root, steps, mean_profile, component="u",
                           max_steps=200):
    """Per-z standard deviation of fluctuations over training data.

    Returns: float64 array of shape (NZ,).
    """
    rng = np.random.RandomState(0)
    subset = (steps if len(steps) <= max_steps
              else rng.choice(steps, max_steps, replace=False))
    sum_sq = np.zeros(NZ, dtype=np.float64)
    count = 0
    for step in subset:
        field = load_field(db_root, component, step)
        for iz in range(NZ):
            plane = field[:, :, iz] - mean_profile[iz]
            sum_sq[iz] += np.sum(plane ** 2)
        count += NX_FULL * NY_FULL
    return np.sqrt(sum_sq / count)


class ChannelFlowYZDataset(Dataset):
    """Fixed-x y-z slice dataset.

    Each sample: u'(y, z; x_i) at fixed streamwise position.
    Shape: (ny_out, nz) float32.

    Parameters
    ----------
    db_root : str
        Path to DNS database root.
    steps : array of int
        Timestep numbers.
    mean_profile : ndarray (NZ,)
        Mean U(z) from training timesteps.
    ix : int
        Fixed streamwise index (0..255).
    plane_stds_z : ndarray (NZ,)
        Per-z std for normalization.
    ny_out : int
        Downsampled y resolution.
    component : str
        Velocity component (u, v, or w).
    preload : bool
        Load all samples into memory.
    """

    def __init__(self, db_root, steps, mean_profile, ix, plane_stds_z,
                 ny_out=128, component="u", preload=False):
        self.db_root = db_root
        self.steps = np.asarray(steps, dtype=np.int64)
        self.mean_profile = np.asarray(mean_profile, dtype=np.float64)
        self.ix = int(ix)
        self.plane_stds_z = np.asarray(plane_stds_z, dtype=np.float64)
        self.component = component
        self.ny_out = ny_out
        self.nz = NZ
        self._preloaded = None
        if preload:
            self._preload_all()

    def _process_field(self, field):
        plane = field[self.ix, :, :]  # (NY_FULL, NZ)
        plane = plane - self.mean_profile[np.newaxis, :]
        if self.ny_out < NY_FULL:
            plane = fourier_downsample_y(plane, self.ny_out)
        safe_std = np.maximum(self.plane_stds_z, 1e-8)
        plane = plane / safe_std[np.newaxis, :]
        return plane.astype(np.float32)

    def _preload_all(self):
        data = np.empty((len(self.steps), self.ny_out, self.nz),
                        dtype=np.float32)
        for i, step in enumerate(self.steps):
            field = load_field(self.db_root, self.component, step)
            data[i] = self._process_field(field)
        self._preloaded = data

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        if self._preloaded is not None:
            return torch.from_numpy(self._preloaded[idx].copy())
        field = load_field(self.db_root, self.component,
                           int(self.steps[idx]))
        return torch.from_numpy(self._process_field(field))

    @property
    def sample_shape(self):
        return (self.ny_out, self.nz)
