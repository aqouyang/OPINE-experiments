"""
Joint three-component fixed-x y-z slice dataset.

Each sample = [u'(y,z), v(y,z), w(y,z)] at a fixed streamwise position x_i.
Shape: (3, ny_out, nz) float32, per-component per-z RMS normalization.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from opine_experiments.channel2d.xy_planes.dataset import (
    NX_FULL, NY_FULL, NZ,
    load_field, compute_mean_profile,
)
from opine_experiments.channel2d.dataset import (
    fourier_downsample_y, compute_all_plane_stds,
)

COMPONENTS = ("u", "v", "w")


def compute_joint_preprocessing(db_root, train_steps):
    """Compute mean profiles and per-z stds for all 3 components.

    Returns (mean_profiles, plane_stds) each a dict {comp: ndarray(NZ,)}.
    """
    mean_profiles = {}
    plane_stds = {}
    for comp in COMPONENTS:
        mean_profiles[comp] = compute_mean_profile(
            db_root, train_steps, component=comp)
        plane_stds[comp] = compute_all_plane_stds(
            db_root, train_steps, mean_profiles[comp], component=comp)
    return mean_profiles, plane_stds


class ChannelFlowYZJointDataset(Dataset):
    """Joint three-component fixed-x y-z slice dataset.

    Parameters
    ----------
    db_root : str
    steps : array of int
    mean_profiles : dict  {comp: ndarray(NZ,)}
    ix : int
    plane_stds : dict  {comp: ndarray(NZ,)}
    ny_out : int
    preload : bool
    """

    def __init__(self, db_root, steps, mean_profiles, ix, plane_stds,
                 ny_out=NY_FULL, preload=False):
        self.db_root = db_root
        self.steps = np.asarray(steps, dtype=np.int64)
        self.mean_profiles = {c: np.asarray(v, dtype=np.float64)
                              for c, v in mean_profiles.items()}
        self.ix = int(ix)
        self.plane_stds = {c: np.maximum(np.asarray(v, dtype=np.float64), 1e-8)
                           for c, v in plane_stds.items()}
        self.ny_out = ny_out
        self.nz = NZ
        self._preloaded = None
        if preload:
            self._preload_all()

    def _process_component(self, field, comp):
        plane = field[self.ix, :, :]  # (NY_FULL, NZ)
        plane = plane - self.mean_profiles[comp][np.newaxis, :]
        if self.ny_out < NY_FULL:
            plane = fourier_downsample_y(plane, self.ny_out)
        plane = plane / self.plane_stds[comp][np.newaxis, :]
        return plane.astype(np.float32)

    def _load_sample(self, step):
        planes = []
        for comp in COMPONENTS:
            field = load_field(self.db_root, comp, step)
            planes.append(self._process_component(field, comp))
        return np.stack(planes, axis=0)  # (3, ny_out, nz)

    def _preload_all(self):
        data = np.empty((len(self.steps), 3, self.ny_out, self.nz),
                        dtype=np.float32)
        for i, step in enumerate(self.steps):
            data[i] = self._load_sample(step)
        self._preloaded = data

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        if self._preloaded is not None:
            return torch.from_numpy(self._preloaded[idx].copy())
        return torch.from_numpy(self._load_sample(int(self.steps[idx])))

    @property
    def sample_shape(self):
        return (3, self.ny_out, self.nz)


class ChannelFlowYZJointDatasetLocal(Dataset):
    """Dataset loaded from pre-extracted .npz files (no raw binary access).

    The .npz contains 'data' (N, 3, 128, 64) float64 raw ix=0 slices.
    Normalization (mean subtraction + RMS scaling) is applied on load.
    """

    def __init__(self, npz_path, mean_profiles, plane_stds, ny_out=NY_FULL):
        arc = np.load(npz_path)
        raw = arc["data"]  # (N, 3, 128, 64) float64
        self.steps = arc["steps"]
        self.ny_out = ny_out
        self.nz = raw.shape[-1]

        # Apply normalization: subtract mean, divide by std
        normed = np.empty((raw.shape[0], 3, ny_out, self.nz), dtype=np.float32)
        for c, comp in enumerate(COMPONENTS):
            mean = np.asarray(mean_profiles[comp], dtype=np.float64)
            std = np.maximum(np.asarray(plane_stds[comp], dtype=np.float64), 1e-8)
            plane = raw[:, c, :, :] - mean[np.newaxis, np.newaxis, :]
            if ny_out < raw.shape[2]:
                plane = np.stack([fourier_downsample_y(plane[i], ny_out)
                                  for i in range(len(plane))])
            normed[:, c] = (plane / std[np.newaxis, np.newaxis, :]).astype(np.float32)
        self._data = normed

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return torch.from_numpy(self._data[idx].copy())

    @property
    def sample_shape(self):
        return (3, self.ny_out, self.nz)
