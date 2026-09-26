#!/usr/bin/env python3
"""Decode one-hot latent directions for POD, CNN and OPINE (Kolmogorov).

What a reduced coordinate of each method looks like as a physical field:
activate exactly ONE reduced coordinate, at an amplitude of one training
standard deviation, and map it back to the vorticity plane.

    POD      dx_j = alpha_j q_j                       (q_j the POD mode)
    OPINE    dx_j = f^-1(mu_z + alpha_j q_j) - f^-1(mu_z),   a = Q^T(z - mu_z)
    CNN      dx_j = g(a0 + alpha_j e_j) - g(a0),       a0 the latent mean

alpha_j is the standard deviation of that coordinate over the TRAINING split,
so every panel is the response to a one-sigma excursion and the amplitudes
are comparable across methods and across j.  For POD that standard deviation
is exact -- sqrt of the covariance eigenvalue, already stored in the basis
file.  For OPINE and the CNN it is measured on the same 20,000-snapshot
training subsample (seed 42) that the OPINE trainer used to fit Q, so the
statistic and the subspace come from the same sample.

ORDERING OF THE OPINE COORDINATES.  Q is maintained by a Procrustes-relaxed
subspace iteration, which keeps the SUBSPACE but not any particular basis of
it: the stored columns are not the latent-variance eigenvectors and "the
j-th column" carries no meaning on its own.  The retained latent covariance
C_a = cov(Q^T(z - mu_z)) is therefore diagonalized and Q is rotated by its
eigenvectors, giving orthonormal directions ordered by latent energy -- the
exact analogue of the POD ordering, spanning the identical subspace, so the
model is untouched.  Both the raw and the rotated coordinates' variances are
recorded.

ORDERING OF THE CNN COORDINATES.  A plain autoencoder latent has no ordering
at all, and the stored index order is an artefact of initialization.  The
four coordinates with the largest training standard deviation are taken, and
their stored indices are recorded so the choice is checkable.

Writes one npz with every field plus a json of the diagnostics.  Nothing is
plotted here -- plot_latent_directions.py reads the npz.
"""
import argparse, json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

N = 128
N_DIR = 4          # directions plotted per method


@torch.no_grad()
def latent_stats_cnn(model, loader, device):
    A = []
    for xb in loader:
        A.append(model.encode(xb.to(device)).cpu())
    return torch.cat(A).double()


@torch.no_grad()
def latent_stats_opine(model, loader, device):
    Q, mu = model.Q, model.z_mean
    A = []
    for xb in loader:
        z = model.encode(xb.to(device))
        A.append(((z - mu.unsqueeze(0)) @ Q).cpu())
    return torch.cat(A).double()


def pair_split(v):
    """Relative eigenvalue split inside each adjacent pair, as a fraction."""
    out = []
    for k in range(0, len(v) - 1, 2):
        a, b = float(v[k]), float(v[k + 1])
        out.append(abs(a - b) / (0.5 * (a + b)) if a + b else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="root holding raw/, splits.json, results/")
    ap.add_argument("--dof", type=int, default=64,
                    help="retained dimension: the CNN latent width, and half "
                         "the OPINE stored rank once --pair-divisor is 2")
    ap.add_argument("--pair-divisor", type=int, default=1, choices=(1, 2),
                    help="2 gives OPINE stored rank 2*dof, because its modes "
                         "come in pairs and the summary figure plots pairs.  "
                         "POD needs no such flag: only its leading directions "
                         "are drawn and they do not depend on where the exact "
                         "basis is truncated")
    ap.add_argument("--opine-ckpt", default=None)
    ap.add_argument("--cnn-ckpt", default=None)
    ap.add_argument("--pod-basis", default=None)
    ap.add_argument("--out", required=True, help="output stem")
    ap.add_argument("--subsample", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    from opine_experiments.kolmogorov.canonical.config import dataset_paths
    from opine_experiments.kolmogorov.canonical.dataset import make_datasets
    from opine_experiments.kolmogorov.canonical.models.cnn import KolmogorovCNN
    from opine_experiments.kolmogorov.coupling.coupling_flow import CouplingFINE

    dev = torch.device(a.device or
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_grad_enabled(False)      # pure inference, start to finish
    paths = dataset_paths(a.dataset)
    R = os.path.join(a.dataset, "results")
    pod_path = a.pod_basis or paths["pod_basis"]
    opine_dof = a.dof * a.pair_divisor       # OPINE's STORED rank
    opine_path = a.opine_ckpt or os.path.join(
        R, f"coupling_fine_D{opine_dof}", f"coupling_D{opine_dof}_best.pt")
    cnn_path = a.cnn_ckpt or os.path.join(R, "cnn", f"CNN{a.dof}_best.pt")

    tr, _, _, sigma = make_datasets(paths["raw"], paths["splits"])
    rng = np.random.default_rng(a.seed)         # same recipe as the trainer
    idx = np.sort(rng.choice(len(tr), size=min(a.subsample, len(tr)),
                             replace=False))
    sub = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr, idx.tolist()), batch_size=a.batch_size,
        shuffle=False, num_workers=a.num_workers)
    print(f"device {dev}   sigma_train {sigma:.6f}   "
          f"subsample {len(idx)} of {len(tr)}", flush=True)

    fields, info = {}, {"dof": a.dof,
                        "_dof_is": "the retained dimension; stored_rank "
                                   "below says what each method holds",
                        "stored_rank": {"OPINE": opine_dof, "CNN": a.dof,
                                        "POD": "leading directions only, "
                                               "rank-independent"},
                        "pair_divisor": a.pair_divisor,
                        "sigma_train": sigma,
                        "n_subsample": int(len(idx)), "seed": a.seed,
                        "field": "delta omega / sigma_train",
                        "alpha_rule": "one training standard deviation of "
                                      "that coordinate"}

    # ------------------------------------------------------------ POD
    pb = np.load(pod_path)
    modes, svals = pb["modes"], pb["singular_values"]
    # singular_values are sqrt of the covariance eigenvalues, i.e. exactly the
    # standard deviation of each POD coefficient over the training split
    alpha_pod = svals[:N_DIR].astype(np.float64)
    for j in range(N_DIR):
        fields[f"pod_{j}"] = (alpha_pod[j] * modes[:, j]).reshape(N, N)
    info["POD"] = {
        "directions": list(range(1, N_DIR + 1)),
        "alpha": alpha_pod.tolist(),
        "alpha_source": "sqrt of the spatial-covariance eigenvalue, from "
                        "pod_basis.npz (exact, not sampled)",
        "coefficient_variance": (svals[:20] ** 2).tolist(),
        "adjacent_pair_relative_split": pair_split(svals[:8] ** 2),
        "basis": pod_path}
    print("POD alpha", np.round(alpha_pod, 4), flush=True)

    # ------------------------------------------------------------ CNN
    cnn = KolmogorovCNN(input_hw=(N, N), latent_dim=a.dof).to(dev)
    cnn.load_state_dict(torch.load(cnn_path, map_location=dev))
    cnn.eval()
    A = latent_stats_cnn(cnn, sub, dev)
    a0, std = A.mean(0), A.std(0, unbiased=True)
    base = cnn.decode(a0.float().unsqueeze(0).to(dev))[0, 0].cpu().numpy()

    # EVERY coordinate is decoded, not just the four that get plotted, so the
    # selection can be checked against a second, independent criterion: the
    # rms of the decoded response.  Selection itself is on the training
    # standard deviation -- that is the same quantity POD and OPINE are
    # ordered by, which is what makes the three rows comparable -- and the
    # response ranking is recorded so the agreement is visible rather than
    # asserted.  64 decodes, seconds.
    resp = np.empty(a.dof)
    for j in range(a.dof):
        zj = a0.clone()
        zj[j] += std[j]
        d = cnn.decode(zj.float().unsqueeze(0).to(dev))[0, 0].cpu().numpy()
        resp[j] = float(np.sqrt(((d - base) ** 2).mean()))

    order = torch.argsort(std, descending=True)[:N_DIR]
    by_resp = np.argsort(-resp)[:N_DIR]
    for k, j in enumerate(order.tolist()):
        z = a0.clone()
        z[j] += std[j]
        out = cnn.decode(z.float().unsqueeze(0).to(dev))[0, 0].cpu().numpy()
        fields[f"cnn_{k}"] = out - base
    info["CNN"] = {
        "directions": [int(j) + 1 for j in order.tolist()],
        "_directions_are": "1-based STORED latent indices, picked as the four "
                           "largest training standard deviations; the stored "
                           "order itself is arbitrary",
        "selection_criterion": "largest standard deviation over the training "
                               "subsample -- the same quantity POD and OPINE "
                               "are ordered by",
        "alpha": [float(std[j]) for j in order.tolist()],
        "latent_std_all": std.tolist(),
        "decoded_response_rms_all": resp.tolist(),
        "top4_by_decoded_response_rms": [int(j) + 1 for j in by_resp],
        "criteria_agree_on": sorted(set(int(j) + 1 for j in order.tolist())
                                    & set(int(j) + 1 for j in by_resp)),
        "base_point": "training latent mean a0",
        "checkpoint": cnn_path}
    print("CNN coords", [int(j) for j in order], "alpha",
          np.round([float(std[j]) for j in order], 4), flush=True)

    # ---------------------------------------------------------- OPINE
    ck = torch.load(opine_path, map_location="cpu", weights_only=False)
    op = CouplingFINE(N, N, opine_dof, channels=1, n_blocks=8, hidden=64,
                      pad_mode="circular_both", init_scale=0.05, s_max=2.0,
                      bottleneck="unrestricted").to(dev)
    op.load_state_dict({k: v.to(dev) for k, v in ck["best_state"].items()})
    op.eval()
    q_sel = ck["Q_selected"].to(dev)
    dq = float((op.Q - q_sel).abs().max())
    if dq > 1e-5:
        raise SystemExit(f"Q in best_state disagrees with Q_selected by {dq}")

    A = latent_stats_opine(op, sub, dev)
    C = torch.cov(A.T)                                   # (D, D)
    evals, V = torch.linalg.eigh(C)
    srt = torch.argsort(evals, descending=True)
    evals, V = evals[srt], V[:, srt]
    Qr = op.Q.double() @ V.to(dev)                       # variance-ordered
    alpha_op = torch.sqrt(torch.clamp(evals, min=0.0)).numpy()

    mu = op.z_mean.double()
    base = op.f.inverse(mu.reshape(1, 1, N, N).float())[0, 0].cpu().numpy()
    for j in range(N_DIR):
        zj = mu + float(alpha_op[j]) * Qr[:, j]
        out = op.f.inverse(zj.reshape(1, 1, N, N).float())[0, 0].cpu().numpy()
        fields[f"opine_{j}"] = out - base
    info["OPINE"] = {
        "directions": list(range(1, N_DIR + 1)),
        "alpha": alpha_op[:N_DIR].tolist(),
        "alpha_source": "sqrt of the eigenvalue of cov(Q^T(z - mu_z)) on the "
                        "training subsample",
        "coefficient_variance": evals[:20].tolist(),
        "adjacent_pair_relative_split": pair_split(evals[:8]),
        "stored_column_variance": A.var(0, unbiased=True).tolist(),
        "rotation_max_abs_offdiag_of_V": float(
            (V - torch.diag(torch.diagonal(V))).abs().max()),
        "Q_rotated": "Q <- Q V, V the eigenvectors of the retained latent "
                     "covariance; same subspace, variance-ordered basis",
        "checkpoint": opine_path}
    print("OPINE alpha", np.round(alpha_op[:N_DIR], 4), flush=True)

    # amplitude of each decoded response, for the colour-scale decision
    info["response_rms"] = {k: float(np.sqrt((v ** 2).mean()))
                            for k, v in fields.items()}
    info["response_absmax"] = {k: float(np.abs(v).max())
                               for k, v in fields.items()}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez_compressed(f"{a.out}.npz",
                        **{k: v.astype(np.float32) for k, v in fields.items()},
                        meta=json.dumps(info))
    json.dump(info, open(f"{a.out}.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}.npz", "and", f"{a.out}.json", flush=True)


if __name__ == "__main__":
    main()
