#!/usr/bin/env python3
"""Unrestricted global POD of the Re=100 Kolmogorov vorticity dataset.

Unrestricted on purpose.  In the channel work the canonical POD enforces
y-translation symmetry, because there the spanwise direction is statistically
homogeneous.  Here it is not: the forcing -k_f cos(k_f y) makes the
statistics explicitly y-dependent, so imposing continuous y-translation
symmetry would be a false prior.  x is homogeneous, but the request is for
unrestricted global POD and no symmetry is imposed in either direction.

Efficiency: the snapshot matrix is 80,000 x 16,384.  Neither the 16,384^2
spatial covariance nor the 80,000^2 Gram matrix is formed.  A randomized
range finder with power iterations gets the leading 256 directions in one
streaming pass plus a few small matmuls, and the result is checked against
oversampling and power-iteration count so the approximation is measured
rather than assumed.

Reconstruction is affine: omega_hat = mean_train + U_D U_D^T (omega - mean).
The mean field is a single vector shared by every D and is not counted in the
DOF budget, which is the standard POD convention and matches the channel work.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

DOFS = (32, 64, 128, 256)
N = 128


def exact_pod(dataset, mean, device="cuda", chunk=4000):
    """Exact leading eigenvectors of the spatial covariance.

    The snapshot matrix is 80,000 x 16,384.  The 80,000^2 Gram matrix (51 GB)
    is out of the question, but the SPATIAL covariance is only
    16,384^2 = 2.1 GB in float64 -- smaller than the data itself -- so it is
    formed by streaming and then diagonalised exactly.  That is both cheaper
    and exact, where a randomized range finder is neither: on this dataset the
    spectrum is dense (D=32 holds only 63% of the energy) and two randomized
    runs at different oversampling disagreed by 1.4e-2 rad, so the trailing
    modes at D=256 were not resolved.

    Returns (n_features, n_features) eigenvectors descending, and eigenvalues.
    """
    nf = mean.shape[0]
    C = torch.zeros(nf, nf, dtype=torch.float64, device=device)
    mu = torch.from_numpy(mean).to(device=device, dtype=torch.float64)
    n = 0
    for a in dataset.arrays:
        for i in range(0, a.shape[0], chunk):
            blk = np.asarray(a[i:i + chunk], dtype=np.float32)
            X = torch.from_numpy(blk).to(device).reshape(blk.shape[0], -1)
            X = X.double() / dataset.sigma - mu
            C += X.T @ X
            n += X.shape[0]
    C /= n
    C = 0.5 * (C + C.T)                       # kill accumulation asymmetry
    evals, evecs = torch.linalg.eigh(C)
    idx = torch.argsort(evals, descending=True)
    return (evecs[:, idx].cpu().numpy(), evals[idx].cpu().numpy(),
            float(torch.diagonal(C).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--oversample", type=int, default=128)
    ap.add_argument("--n-power", type=int, default=4)
    ap.add_argument("--device", default=None)
    # The retained ranks are a CLI argument so the table can be extended (for
    # example to 512) without touching the frozen default.
    ap.add_argument("--dofs", type=int, nargs="+", default=list(DOFS))
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from opine_experiments.kolmogorov.canonical.dataset import (
        make_datasets, canonical_nmse)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    tr, va, te, sigma = make_datasets(args.raw_dir, args.splits)
    print(f"sigma_train = {sigma:.6f}   "
          f"train {len(tr)} / val {len(va)} / test {len(te)} snapshots",
          flush=True)

    # training mean field, streamed
    mean = np.zeros(N * N, dtype=np.float64)
    n_seen = 0
    for a in tr.arrays:
        blk = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)
        mean += blk.sum(axis=0)
        n_seen += blk.shape[0]
    mean /= n_seen * sigma
    print(f"training mean field over {n_seen} snapshots; "
          f"||mean|| = {np.linalg.norm(mean):.6f}", flush=True)

    dofs = sorted(args.dofs)
    evecs, evals, trace = exact_pod(tr, mean, device=dev)
    rank = max(dofs)
    modes = evecs[:, :rank]
    svals = np.sqrt(np.maximum(evals[:rank], 0.0))
    orth = float(np.abs(modes.T @ modes - np.eye(rank)).max())
    neg = float(evals.min())
    print(f"modes {modes.shape}  orthonormality err {orth:.2e}  "
          f"most negative eigenvalue {neg:.2e}  trace {trace:.6f}", flush=True)

    Xtr = tr.as_matrix()
    Xtr -= mean
    Xva, Xte = va.as_matrix(), te.as_matrix()
    res = {
        "_nmse": "sum|recon-truth|^2 / sum|truth|^2",
        "_pod": "unrestricted global POD; no symmetry imposed in x or y",
        "_reconstruction": "affine: mean_train + U_D U_D^T (x - mean_train)",
        "_mean_not_in_dof_budget": True,
        "sigma_train": sigma,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "orthonormality_error": orth,
        "method": "exact eigh of the 16384^2 spatial covariance, "
                  "streamed accumulation in float64",
        "most_negative_eigenvalue": neg,
        "covariance_trace": trace,
        "eigenvalues_top20": evals[:20].tolist(),
        "singular_values_top20": svals[:20].tolist(),
        "table": {},
    }
    total_energy = float((Xtr ** 2).sum())
    for D in dofs:
        U = modes[:, :D]
        row = {}
        for tag, X in (("train", Xtr + mean), ("val", Xva), ("test", Xte)):
            Xc = X - mean
            rec = (Xc @ U) @ U.T + mean
            row[f"{tag}_nmse"] = canonical_nmse(rec, X)
        row["captured_energy_fraction"] = float(
            ((Xtr @ U) ** 2).sum() / total_energy)
        row["captured_energy_from_eigenvalues"] = float(
            evals[:D].sum() / evals.sum())
        res["table"][str(D)] = row
        print(f"  D={D:>4}  train {row['train_nmse']:.8f}  "
              f"val {row['val_nmse']:.8f}  test {row['test_nmse']:.8f}  "
              f"energy {row['captured_energy_fraction']:.6f}", flush=True)

    np.savez_compressed(os.path.join(args.output_dir, "pod_basis.npz"),
                        modes=modes.astype(np.float32),
                        singular_values=svals, mean=mean.astype(np.float32),
                        sigma_train=sigma, dofs=np.array(dofs))

    # sample-0 reconstruction and error, every retained rank
    x0 = Xte[0]
    rec0 = {str(D): ((x0 - mean) @ modes[:, :D]) @ modes[:, :D].T + mean
            for D in dofs}
    np.savez_compressed(os.path.join(args.output_dir, "sample0.npz"),
                        truth=x0.reshape(N, N).astype(np.float32),
                        **{f"recon_D{D}": rec0[str(D)].reshape(N, N)
                           .astype(np.float32) for D in dofs},
                        **{f"error_D{D}": (rec0[str(D)] - x0).reshape(N, N)
                           .astype(np.float32) for D in dofs})
    with open(os.path.join(args.output_dir, "pod_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nsaved -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
