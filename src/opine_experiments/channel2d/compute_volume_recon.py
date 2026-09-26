#!/usr/bin/env python3
"""Reconstruct a FULL 3-D snapshot by running the y-z models plane by plane.

The y-z methods see one wall-normal plane at a time: they were fitted on the
ix = 0 slice and know nothing about the streamwise direction.  This takes one
DNS snapshot, feeds every one of its 256 x-planes through POD, the CNN and
OPINE independently, and stacks the results back into a volume.

WHY THIS IS LEGITIMATE.  x is a periodic, statistically homogeneous direction
of the channel, so a plane at any ix is a draw from the same distribution the
models were fitted on -- applying them across x is resampling, not
extrapolation.

WHAT IT DELIBERATELY EXPOSES.  Nothing in these models couples neighbouring
x-planes.  Whatever streamwise coherence survives in the stacked volume is
there because each plane was reconstructed well, not because the method knew
the planes were adjacent.  That is the point of drawing the isosurfaces.

Normalization is the training one, per component and per z:
    x_hat = (u - mean_profile(z)) / plane_std(z),
both read from preprocessing_yz_joint.json.  Fields are written back out as
u'/u_tau = x_hat * plane_std(z), with u_tau = 1 in these LESGO units.

POD is not reimplemented: a CouplingFINE_YZ has its transform replaced by the
exact identity BEFORE init_subspace, so the canonical sympod initialization
makes it literally the JointSymmetryPOD projector -- same band allocation,
same safe-mean centring, same code as the benchmark.
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NX, NY, NZ, NCOMP = 256, 128, 64, 3
COMPONENTS = ("u", "v", "w")
LX, LY, LZ = 2.0 * np.pi, np.pi, 1.0
RE_TAU = 180.0


class _Identity(nn.Module):
    def forward(self, x):
        return x

    def inverse(self, y):
        return y


def load_volume(volume_dir, step):
    """(3, NX, NY, NZ) raw DNS velocity, same reader as extract_yz_slices."""
    out = np.empty((NCOMP, NX, NY, NZ), dtype=np.float64)
    for c, comp in enumerate(COMPONENTS):
        p = os.path.join(volume_dir, f"{comp}_velocity.{step:08d}")
        out[c] = np.fromfile(p, dtype="<f8").reshape((NX, NY, NZ), order="F")
    return out


@torch.no_grad()
def run_planes(model, X, device, batch=64):
    """X: (NX, 3, NY, NZ) normalized planes -> same shape, reconstructed."""
    out = torch.empty_like(X)
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch].to(device)
        r = model(xb)
        r = r[0] if isinstance(r, tuple) else r
        out[i:i + batch] = r.cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-dir", required=True)
    ap.add_argument("--preprocessing-file", required=True)
    ap.add_argument("--volume-dir", required=True,
                    help="directory holding [uvw]_velocity.<step>")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--dof", type=int, default=256)
    ap.add_argument("--checkpoint", required=True, help="OPINE best.pt")
    ap.add_argument("--cnn-checkpoint", required=True)
    ap.add_argument("--cnn-dof", type=int, default=None,
                    help="CNN latent width, when it differs from --dof.  POD "
                         "and OPINE retain in pairs, so their stored rank is "
                         "twice the dimension the figure means; the CNN "
                         "latent is that dimension itself")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    from opine_experiments.channel2d.dataset_joint import (
        ChannelFlowYZJointDatasetLocal)
    from opine_experiments.channel2d.train_latent_rank import (
        load_joint_preprocessing)
    from opine_experiments.channel2d.coupling.coupling_fine_yz import (
        CouplingFINE_YZ)
    from opine_experiments.channel2d.cnn_yz_joint import JointCNN_YZ

    dev = torch.device(a.device or
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_grad_enabled(False)
    mp, ps, _, _, _ = load_joint_preprocessing(a.preprocessing_file)
    mean = np.stack([np.asarray(mp[c], dtype=np.float64) for c in COMPONENTS])
    std = np.stack([np.maximum(np.asarray(ps[c], dtype=np.float64), 1e-8)
                    for c in COMPONENTS])
    print(f"device {dev}   dof {a.dof}   step {a.step}", flush=True)

    # ---- POD: identity transform + canonical sympod init on the train split
    tr = ChannelFlowYZJointDatasetLocal(
        os.path.join(a.extracted_dir, "train.npz"), mp, ps, NY)
    tr_ld = torch.utils.data.DataLoader(tr, batch_size=64)
    torch.manual_seed(42); np.random.seed(42)
    pod = CouplingFINE_YZ(NY, NZ, a.dof, channels=NCOMP).to(dev)
    pod.f = _Identity().to(dev)
    t0 = time.time()
    pod.init_subspace(tr_ld, dev, method="sympod", n_power=12, seed=42)
    pod.eval()
    print(f"POD sympod init in {time.time()-t0:.0f}s", flush=True)

    # ---- OPINE
    torch.manual_seed(42); np.random.seed(42)
    opine = CouplingFINE_YZ(NY, NZ, a.dof, channels=NCOMP, n_blocks=8,
                            hidden=64).to(dev)
    st = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    opine.load_state_dict(st["best_state"]); opine.eval()

    # ---- CNN
    cnn_dof = a.cnn_dof or a.dof
    cnn = JointCNN_YZ(input_hw=(NY, NZ), latent_dim=cnn_dof).to(dev)
    cnn.load_state_dict(torch.load(a.cnn_checkpoint, map_location=dev,
                                   weights_only=True))
    cnn.eval()

    # ---- the snapshot, normalized plane by plane
    vol = load_volume(a.volume_dir, a.step)                 # (3, NX, NY, NZ)
    m4 = mean.reshape(NCOMP, 1, 1, NZ)          # broadcast over x and y
    s4 = std.reshape(NCOMP, 1, 1, NZ)
    hat = (vol - m4) / s4
    X = torch.from_numpy(np.ascontiguousarray(
        hat.transpose(1, 0, 2, 3))).float()                 # (NX, 3, NY, NZ)

    fields, nmse = {"truth": X}, {}
    for name, model in (("pod", pod), ("cnn", cnn), ("opine", opine)):
        t0 = time.time()
        r = run_planes(model, X, dev)
        fields[name] = r
        num = float((r.double() - X.double()).pow(2).sum())
        den = float(X.double().pow(2).sum())
        nmse[name] = num / den
        print(f"{name:6s} volume NMSE {nmse[name]:.6f}   "
              f"({time.time()-t0:.0f}s)", flush=True)

    # back to u'/u_tau; u_tau = 1 in these units, so this is just the
    # per-z rescaling that the normalization removed
    s = torch.from_numpy(std.reshape(1, NCOMP, 1, NZ)).float()
    out = {k: (v * s).permute(1, 0, 2, 3).contiguous().numpy().astype(
        np.float32) for k, v in fields.items()}            # (3, NX, NY, NZ)

    meta = {"step": a.step, "dof": a.dof, "re_tau": RE_TAU,
            "stored_rank": {"POD": a.dof, "OPINE": a.dof, "CNN": cnn_dof},
            "retained_D": {"POD": a.dof // 2, "OPINE": a.dof // 2,
                           "CNN": cnn_dof},
            "shape": "(component, x, y, z)", "components": list(COMPONENTS),
            "Lx": LX, "Ly": LY, "Lz": LZ, "nx": NX, "ny": NY, "nz": NZ,
            "units": "u'/u_tau (fluctuation about the training mean profile)",
            "volume_nmse": nmse,
            "_nmse_is": "sum|xhat-x|^2/sum|x|^2 over the whole volume in the "
                        "NORMALIZED variable, the same metric the benchmark "
                        "table uses, here aggregated over all 256 x-planes "
                        "of one snapshot rather than over the test split",
            "_reconstruction": "each x-plane reconstructed INDEPENDENTLY; no "
                               "method sees more than one plane at a time",
            "_why_x_is_fair": "x is periodic and statistically homogeneous, "
                              "so a plane at any ix is a draw from the "
                              "distribution the models were fitted on",
            "opine_checkpoint": a.checkpoint, "cnn_checkpoint": a.cnn_checkpoint,
            "preprocessing": a.preprocessing_file}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez_compressed(f"{a.out}.npz", meta=json.dumps(meta), **out)
    json.dump(meta, open(f"{a.out}.json", "w"), indent=1)
    print("wrote", f"{a.out}.npz", flush=True)


if __name__ == "__main__":
    main()
