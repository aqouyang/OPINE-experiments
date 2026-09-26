#!/usr/bin/env python3
"""Pick a representative test snapshot for the POD vs Coupling-FINE figure.

Cherry-picking is the obvious failure mode here, so the sample is chosen by a
stated rule rather than by eye: compute per-sample NMSE for both methods over
the whole canonical test split, form the per-sample relative improvement
(pod - cf)/pod, and take the snapshot whose improvement is closest to the
MEDIAN. Its percentile is recorded so the choice is auditable.

Evaluation only -- existing checkpoints, canonical split, canonical NMSE.
Nothing is retrained.
"""
import argparse, json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
N = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dof", type=int, default=128,
                    help="stored rank of POD and OPINE, which retain in "
                         "pairs, so the dimension the figure means is half "
                         "of it")
    ap.add_argument("--cnn-dof", type=int, default=None,
                    help="CNN latent width, when it differs from --dof; set "
                         "it to dof/2 to put all three arms at the same "
                         "retained dimension")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    from opine_experiments.kolmogorov.canonical.dataset import make_datasets
    from opine_experiments.kolmogorov.canonical.models.cnn import KolmogorovCNN
    from opine_experiments.kolmogorov.coupling.coupling_flow import CouplingFINE

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = os.path.join(a.base, "results")
    _, _, te, _ = make_datasets(os.path.join(a.base, "raw"),
                                os.path.join(a.base, "splits.json"))
    ld = torch.utils.data.DataLoader(te, batch_size=128, shuffle=False,
                                     num_workers=8)
    pb = np.load(os.path.join(res, "pod", "pod_basis.npz"))
    Q = torch.from_numpy(pb["modes"].astype(np.float32)[:, :a.dof]).to(dev)
    mu = torch.from_numpy(pb["mean"].astype(np.float32)).to(dev)

    cf = CouplingFINE(N, N, a.dof, channels=1, n_blocks=8, hidden=64,
                      pad_mode="circular_both",
                      bottleneck="unrestricted").to(dev)
    st = torch.load(os.path.join(res, f"coupling_fine_D{a.dof}",
                                 f"coupling_D{a.dof}_best.pt"),
                    map_location=dev, weights_only=False)
    cf.load_state_dict(st["best_state"]); cf.eval()

    n_p, n_c = [], []
    with torch.no_grad():
        for xb in ld:
            xb = xb.to(dev)
            den = xb.double().pow(2).sum(dim=(1, 2, 3))
            xf = xb.reshape(xb.shape[0], -1)
            rp = ((xf - mu) @ Q) @ Q.T + mu
            n_p.append(((rp - xf).double().pow(2).sum(dim=1) / den).cpu())
            n_c.append(((cf(xb) - xb).double().pow(2).sum(dim=(1, 2, 3))
                        / den).cpu())
    p = torch.cat(n_p).numpy(); c = torch.cat(n_c).numpy()
    rel = (p - c) / p
    med = float(np.median(rel))
    idx = int(np.argmin(np.abs(rel - med)))
    pct = float((rel < rel[idx]).mean() * 100.0)

    # CNN is added to the figure but NOT to the selection rule: the plane is
    # still the median of the POD-vs-Coupling improvement, unchanged, so the
    # sample index does not move when this arm is included.
    cnn_rec = None
    cnn_dof = a.cnn_dof or a.dof
    cnn_ck = os.path.join(res, "cnn", f"CNN{cnn_dof}_best.pt")
    if os.path.isfile(cnn_ck):
        cnn = KolmogorovCNN(input_hw=(N, N), latent_dim=cnn_dof).to(dev)
        cnn.load_state_dict(torch.load(cnn_ck, map_location=dev,
                                       weights_only=True))
        cnn.eval()

    x = te[idx].unsqueeze(0).to(dev)
    with torch.no_grad():
        xf = x.reshape(1, -1)
        pod_rec = (((xf - mu) @ Q) @ Q.T + mu).reshape(N, N).cpu().numpy()
        cf_rec = cf(x)[0, 0].cpu().numpy()
        if cnn_rec is None and os.path.isfile(cnn_ck):
            r = cnn(x)
            r = r[0] if isinstance(r, tuple) else r
            cnn_rec = r[0, 0].cpu().numpy()
    truth = x[0, 0].cpu().numpy()
    vmax = float(np.abs(truth).max())

    meta = {
        "dof": a.dof, "sample_index": idx,
        "stored_rank": {"POD": a.dof, "OPINE": a.dof, "CNN": cnn_dof},
        "retained_D": {"POD": a.dof // 2, "OPINE": a.dof // 2,
                       "CNN": cnn_dof},
        "selection_rule": "per-sample relative improvement (pod-cf)/pod "
                          "closest to the median over the full test split",
        "n_test": int(len(p)),
        "sample_percentile_of_relative_improvement": pct,
        "median_relative_improvement": med,
        "sample_relative_improvement": float(rel[idx]),
        "pod_nmse_this_sample": float(p[idx]),
        "coupling_nmse_this_sample": float(c[idx]),
        "pod_nmse_test_set": float(p.sum() / len(p)),
        "coupling_nmse_test_set": float(c.sum() / len(c)),
        "vmin": -vmax, "vmax": vmax,
        "_vlim_note": "symmetric, from the reference field only; both "
                      "reconstructions use these same limits",
    }
    if cnn_rec is not None:
        meta["cnn_nmse_this_sample"] = float(
            np.sum((cnn_rec - truth) ** 2) / np.sum(truth ** 2))
        np.savez_compressed(a.output, truth=truth, pod=pod_rec,
                            coupling=cf_rec, cnn=cnn_rec,
                            meta=json.dumps(meta))
    else:
        np.savez_compressed(a.output, truth=truth, pod=pod_rec,
                            coupling=cf_rec, meta=json.dumps(meta))
    json.dump(meta, open(a.output.replace(".npz", ".json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
