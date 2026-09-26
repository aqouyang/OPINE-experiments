#!/usr/bin/env python3
"""Pick a representative channel test plane for the POD vs Coupling-FINE figure.

The POD arm is not reimplemented. A CouplingFINE_YZ is built, its transform is
replaced by the exact identity BEFORE `init_subspace`, and the canonical
sympod initialization then makes the model literally the JointSymmetryPOD
projector -- same band allocation, same safe-mean centring, same code. Its
test NMSE is printed so it can be checked against the canonical table.

The snapshot is chosen by rule: per-sample relative improvement (pod-cf)/pod
closest to the MEDIAN over the whole test split.
"""
import argparse, json, os, sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
NY, NZ, NCOMP = 128, 64, 3


class _Identity(nn.Module):
    def forward(self, x):
        return x

    def inverse(self, y):
        return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-dir", required=True)
    ap.add_argument("--preprocessing-file", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dof", type=int, required=True)
    ap.add_argument("--cnn-checkpoint", default=None)
    ap.add_argument("--cnn-dof", type=int, default=None,
                    help="CNN latent width, when it differs from --dof.  POD "
                         "and OPINE retain in pairs, so their stored rank is "
                         "twice the dimension the summary figure plots; the "
                         "CNN latent is that dimension itself")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    from opine_experiments.channel2d.dataset_joint import (
        ChannelFlowYZJointDatasetLocal)
    from opine_experiments.channel2d.train_latent_rank import (
        load_joint_preprocessing)
    from opine_experiments.channel2d.coupling.coupling_fine_yz import (
        CouplingFINE_YZ)
    from opine_experiments.channel2d.cnn_yz_joint import JointCNN_YZ

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp, ps, _, _, _ = load_joint_preprocessing(a.preprocessing_file)
    ds = lambda s: ChannelFlowYZJointDatasetLocal(
        os.path.join(a.extracted_dir, f"{s}.npz"), mp, ps, NY)
    tr, te = ds("train"), ds("test")
    tr_ld = torch.utils.data.DataLoader(tr, batch_size=64)
    te_ld = torch.utils.data.DataLoader(te, batch_size=64)

    torch.manual_seed(42); np.random.seed(42)
    pod = CouplingFINE_YZ(NY, NZ, a.dof, channels=NCOMP).to(dev)
    pod.f = _Identity().to(dev)                 # exact identity BEFORE init
    pod.init_subspace(tr_ld, dev, method="sympod", n_power=12, seed=42)
    pod.eval()

    torch.manual_seed(42); np.random.seed(42)
    cf = CouplingFINE_YZ(NY, NZ, a.dof, channels=NCOMP, n_blocks=8,
                         hidden=64).to(dev)
    st = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    cf.load_state_dict(st["best_state"]); cf.eval()

    # CNN joins the figure but NOT the selection rule: the plane stays the
    # median of the POD-vs-Coupling improvement, so the index cannot shift to
    # one that happens to flatter or punish the CNN.
    cnn = None
    if a.cnn_checkpoint and os.path.isfile(a.cnn_checkpoint):
        cnn_dof = a.cnn_dof or a.dof
        cnn = JointCNN_YZ(input_hw=(NY, NZ), latent_dim=cnn_dof).to(dev)
        cnn.load_state_dict(torch.load(a.cnn_checkpoint, map_location=dev,
                                       weights_only=True))
        cnn.eval()

    n_p, n_c = [], []
    with torch.no_grad():
        for xb in te_ld:
            xb = xb.to(dev)
            den = xb.double().pow(2).sum(dim=(1, 2, 3))
            n_p.append(((pod(xb) - xb).double().pow(2).sum(dim=(1, 2, 3))
                        / den).cpu())
            n_c.append(((cf(xb) - xb).double().pow(2).sum(dim=(1, 2, 3))
                        / den).cpu())
    p = torch.cat(n_p).numpy(); c = torch.cat(n_c).numpy()
    pod_agg = float(p.sum() / len(p))
    print(f"POD (identity+sympod) aggregate per-sample mean NMSE {pod_agg:.6f}",
          flush=True)

    rel = (p - c) / p
    med = float(np.median(rel))
    idx = int(np.argmin(np.abs(rel - med)))
    pct = float((rel < rel[idx]).mean() * 100.0)
    x = torch.stack([te[idx]]).to(dev)
    extra = {}
    with torch.no_grad():
        pr = pod(x)[0].cpu().numpy()
        cr = cf(x)[0].cpu().numpy()
        if cnn is not None:
            r = cnn(x)
            r = r[0] if isinstance(r, tuple) else r
            extra["cnn"] = r[0].cpu().numpy()
            num = den = 0.0
            for xb in te_ld:
                xb = xb.to(dev)
                rr = cnn(xb)
                rr = rr[0] if isinstance(rr, tuple) else rr
                num += float((rr - xb).double().pow(2).sum())
                den += float(xb.double().pow(2).sum())
    truth = x[0].cpu().numpy()
    meta = {"dof": a.dof, "sample_index": idx,
            "stored_rank": {"POD": a.dof, "OPINE": a.dof,
                            "CNN": a.cnn_dof or a.dof},
            "retained_D": {"POD": a.dof // 2, "OPINE": a.dof // 2,
                           "CNN": a.cnn_dof or a.dof},
            "selection_rule": "per-sample relative improvement (pod-cf)/pod "
                              "closest to the median over the test split",
            "n_test": int(len(p)),
            "sample_percentile_of_relative_improvement": pct,
            "median_relative_improvement": med,
            "pod_nmse_this_sample": float(p[idx]),
            "coupling_nmse_this_sample": float(c[idx]),
            "pod_per_sample_mean_nmse": pod_agg,
            "coupling_per_sample_mean_nmse": float(c.sum() / len(c)),
            "components": ["u", "v", "w"],
            "vmax_per_component": [float(np.abs(truth[k]).max())
                                   for k in range(NCOMP)]}
    if "cnn" in extra:
        meta["cnn_test_set_nmse"] = num / den
        meta["cnn_nmse_this_sample"] = float(
            np.sum((extra["cnn"] - truth) ** 2) / np.sum(truth ** 2))
    np.savez_compressed(a.output, truth=truth, pod=pr, coupling=cr,
                        meta=json.dumps(meta), **extra)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
