#!/usr/bin/env python3
"""Common PyTorch interface for the Re=100 Kolmogorov vorticity dataset.

Every reconstruction method (POD, CNN, iResNet FINE) consumes this, on the
same trajectory-level split, so their NMSE numbers are directly comparable.

Sample shape: (1, 128, 128) -- scalar out-of-plane vorticity omega(x, y).
Axis order on disk is (x, y): dim -2 is x, dim -1 is y.

Symmetry, and why it is not exploited:
  x is a continuous translation-equivariant direction.
  y is NOT: the forcing -k_f cos(k_f y) depends explicitly on y, so the
  statistics are y-dependent even though the boundary is periodic.
  Convolutions may therefore use circular padding in both directions -- that
  is just the correct boundary condition -- but no method here imposes
  continuous y-translation symmetry on its modes or latent subspace.

Normalization uses TRAINING TRAJECTORIES ONLY and is a single global scalar,
    omega_hat = omega / sigma_train,
so the canonical NMSE = sum|omega_hat - omega|^2 / sum|omega|^2 is identical
whether evaluated on normalized or raw vorticity -- the scale cancels.  No
mean field is removed here; methods that want an affine offset (POD) subtract
their own training mean internally and report it.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

N = 128


class KolmogorovVorticityDataset(Dataset):
    """Snapshots from a set of trajectories, as (1, 128, 128) float32."""

    def __init__(self, raw_dir, traj_ids, sigma=None, mmap=True):
        self.raw_dir = raw_dir
        self.traj_ids = list(traj_ids)
        self.files = [os.path.join(raw_dir, f"traj_{i:03d}.npy")
                      for i in self.traj_ids]
        for f in self.files:
            if not os.path.isfile(f):
                raise FileNotFoundError(f)
        self.arrays = [np.load(f, mmap_mode="r" if mmap else None)
                       for f in self.files]
        self.per = self.arrays[0].shape[0]
        for a in self.arrays:
            assert a.shape == (self.per, N, N), a.shape
        self.sigma = sigma

    def __len__(self):
        return len(self.arrays) * self.per

    def __getitem__(self, i):
        a, k = divmod(i, self.per)
        w = np.asarray(self.arrays[a][k], dtype=np.float32)
        if self.sigma is not None:
            w = w / self.sigma
        return torch.from_numpy(w).unsqueeze(0)      # (1, 128, 128)

    def as_matrix(self, dtype=np.float64):
        """(n_snapshots, 128*128), for POD.  Materialises the split."""
        out = np.empty((len(self), N * N), dtype=dtype)
        j = 0
        for a in self.arrays:
            blk = np.asarray(a, dtype=dtype)
            if self.sigma is not None:
                blk = blk / self.sigma
            out[j:j + blk.shape[0]] = blk.reshape(blk.shape[0], -1)
            j += blk.shape[0]
        return out


def load_splits(splits_file):
    with open(splits_file) as fh:
        s = json.load(fh)
    return s["train"], s["val"], s["test"]


def training_sigma(raw_dir, train_ids):
    """Global RMS vorticity over the training trajectories only."""
    tot = 0.0
    cnt = 0
    for i in train_ids:
        a = np.load(os.path.join(raw_dir, f"traj_{i:03d}.npy"),
                    mmap_mode="r")
        blk = np.asarray(a, dtype=np.float64)
        tot += float((blk ** 2).sum())
        cnt += blk.size
    return float(np.sqrt(tot / cnt))


def make_datasets(raw_dir, splits_file, normalize=True):
    tr, va, te = load_splits(splits_file)
    sigma = training_sigma(raw_dir, tr) if normalize else None
    return (KolmogorovVorticityDataset(raw_dir, tr, sigma),
            KolmogorovVorticityDataset(raw_dir, va, sigma),
            KolmogorovVorticityDataset(raw_dir, te, sigma),
            sigma)


def canonical_nmse(recon, truth):
    """sum|recon - truth|^2 / sum|truth|^2, the project-wide convention."""
    num = float(np.sum((recon - truth) ** 2))
    den = float(np.sum(truth ** 2))
    return num / den
