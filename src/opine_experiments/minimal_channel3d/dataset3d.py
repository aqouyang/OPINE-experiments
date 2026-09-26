#!/usr/bin/env python3
"""3-D minimal-channel dataset: chronological split + the project's own
normalization convention.

Fields are (N, 3, 64, 32, 32) = (sample, component, z, y, x), float32, memory
mapped -- the file is 15.7 GB and must not be read whole.

NORMALIZATION follows ChannelFlowYZJointDatasetLocal exactly: for each
component and each wall-normal level, subtract a mean profile and divide by an
rms profile,

    x_hat[c, k, :, :] = ( x[c, k, :, :] - mu[c, k] ) / sigma[c, k]

with mu and sigma computed over the TRAINING samples and the two homogeneous
directions only. Per-z scaling is not cosmetic here: <u> runs from 1.4 at the
first node to 19.7 at the lid, so without it an L2 metric would be dominated
by the outer layer and the near-wall region would be invisible.

Snapshots are never normalized individually.

CHRONOLOGICAL SPLIT. `step` is strictly increasing across the file, so index
order is acquisition order and a contiguous split is a time split. (`t` has 6
stale duplicate stamps at the two segment starts, indices 0-2 and 10000-10002;
they do not affect ordering.) Adjacent snapshots are correlated at 0.37
h/u_tau spacing, so a random split would leak.

    train  [0, 16000)      val  [16000, 18000)      test  [18000, 20000)

WALL VALUE. w is stored on the w-grid whose first node is the wall, so
w[:, 0, :, :] is identically zero. Its training sigma is therefore 0 and is
floored at 1e-8, leaving that slab exactly zero after normalization. That is
correct -- it carries no variance and no method can or should reconstruct
anything there.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Location of the dataset, given by the environment so one code revision
# runs anywhere -- the path is the only thing that differs between machines,
# and it is data, not behaviour.
# Location of the 3-D minimal-channel dataset.  It is 15.7 GB and is not
# distributed with this repository; see docs/DATASETS.md.  Set the
# environment variable to the directory holding channel_retau180_nz64.npy
# and channel_retau180_nz64_meta.npz.
ROOT = os.environ.get("CHANNEL3D_ROOT")
if ROOT is None:                       # keep the import side-effect-free
    ROOT = ""
NPY = os.path.join(ROOT, "channel_retau180_nz64.npy")


def require_root():
    """Fail with an actionable message rather than a confusing FileNotFound."""
    if not ROOT:
        raise SystemExit(
            "CHANNEL3D_ROOT is not set.  Point it at the directory holding "
            "channel_retau180_nz64.npy; see docs/DATASETS.md.")
    return ROOT
META = os.path.join(ROOT, "channel_retau180_nz64_meta.npz")
NC, NZ, NY, NX = 3, 64, 32, 32
N_FEAT = NC * NZ * NY * NX                 # 196608
SPLITS = {"train": (0, 16000), "val": (16000, 18000), "test": (18000, 20000)}


def compute_stats(cache, chunk=500):
    """Per-(component, z) mean and rms over the TRAINING split only."""
    if os.path.isfile(cache):
        d = np.load(cache)
        return d["mean"], d["std"]
    x = np.load(NPY, mmap_mode="r")
    a, b = SPLITS["train"]
    n = b - a
    s1 = np.zeros((NC, NZ), dtype=np.float64)
    s2 = np.zeros((NC, NZ), dtype=np.float64)
    for i in range(a, b, chunk):
        blk = np.asarray(x[i:min(i + chunk, b)], dtype=np.float64)
        s1 += blk.sum(axis=(0, 3, 4))
        s2 += (blk ** 2).sum(axis=(0, 3, 4))
    cnt = n * NY * NX
    mean = s1 / cnt
    var = np.maximum(s2 / cnt - mean ** 2, 0.0)
    std = np.sqrt(var)
    np.savez(cache, mean=mean, std=std, n_train=n,
             _note="per (component, z); training split only; homogeneous "
                   "directions x,y averaged out")
    return mean, std


class Channel3DDataset(Dataset):
    def __init__(self, split, mean, std, preload=False):
        self.a, self.b = SPLITS[split]
        self.x = np.load(NPY, mmap_mode="r")
        self.mean = np.asarray(mean, dtype=np.float32)[:, :, None, None]
        self.std = np.maximum(np.asarray(std, dtype=np.float32),
                              1e-8)[:, :, None, None]
        self._cache = None
        if preload:
            self._cache = ((np.asarray(self.x[self.a:self.b],
                                       dtype=np.float32) - self.mean)
                           / self.std)

    def __len__(self):
        return self.b - self.a

    def __getitem__(self, i):
        if self._cache is not None:
            return torch.from_numpy(self._cache[i].copy())
        v = np.asarray(self.x[self.a + i], dtype=np.float32)
        return torch.from_numpy((v - self.mean) / self.std)

    @property
    def sample_shape(self):
        return (NC, NZ, NY, NX)


def make_datasets(cache, preload=False):
    mean, std = compute_stats(cache)
    return (Channel3DDataset("train", mean, std, preload),
            Channel3DDataset("val", mean, std, preload),
            Channel3DDataset("test", mean, std, preload), mean, std)
