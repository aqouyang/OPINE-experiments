#!/usr/bin/env python3
"""CANONICAL 1-D toy result: iid Uniform phase split, multiple seeds.

    f_omega(x) = tanh(sin(x + omega)),   omega ~ Uniform[0, 2pi)

omega is drawn iid from the generative distribution for train, val and test --
32 / 256 / 2000 draws, independent, no grid and no forced midpoints. Exact
overlap has probability zero, so no rejection sampling is needed or used.

This supersedes the deterministic 32-point grid with half-grid test midpoints,
which is a constructed interpolation experiment and is retained only as the
"uniform-grid interpolation diagnostic".

DISTANCE-STRATIFIED ERROR. With 32 iid phases the circle is covered unevenly:
some test phases land next to a training phase, others fall in a gap. Test
samples are therefore binned by circular distance to the nearest training
phase (closest / middle / farthest third) and NMSE is reported per bin as
sum/sum within the bin. If the model were interpolating rather than learning
a representation, its error would climb steeply with that distance the way a
nearest-phase copy's does.
"""
import argparse, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from toy_canonical import fields, oracle, train_coupling, latent_rank, DELTAS
from toy_tanh import pod_fit, pod_project, nmse, te_error


def circ_gap(w_te, w_tr):
    d = np.abs(w_te[:, None] - w_tr[None, :])
    d = np.minimum(d, 2 * np.pi - d)
    return d.min(1), d.argmin(1)


def coverage_max_gap(w_tr):
    """Largest empty arc between consecutive training phases."""
    s = np.sort(np.asarray(w_tr))
    return float(max(np.max(np.diff(s)), 2 * np.pi - (s[-1] - s[0])))


def binned(rec, truth, d, n_bins=3):
    q = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        m = (d >= q[i]) & (d <= q[i + 1] if i == n_bins - 1 else d < q[i + 1])
        num = float(((rec[m] - truth[m]) ** 2).sum())
        den = float((truth[m] ** 2).sum())
        out.append({"bin": i, "d_lo": float(q[i]), "d_hi": float(q[i + 1]),
                    "n": int(m.sum()), "nmse": num / den})
    return out


def one_seed(seed, a, dev):
    rng = np.random.default_rng(1000 + seed)
    w_tr = rng.uniform(0, 2 * np.pi, a.n_train)
    w_va = rng.uniform(0, 2 * np.pi, a.n_val)
    w_te = rng.uniform(0, 2 * np.pi, a.n_test)
    tr, va, te = fields(w_tr), fields(w_va), fields(w_te)
    Ftr, Fte = tr.reshape(len(tr), -1), te.reshape(len(te), -1)
    m_raw, mu_raw, s_raw = pod_fit(Ftr)
    model, c_tr, c_va, c_te = train_coupling(tr, va, te, a, dev)

    def cpl(Z):
        with torch.no_grad():
            return model(torch.tensor(Z, dtype=torch.float32,
                                      device=dev)).cpu().numpy()
    cpl_rec = cpl(te)[:, 0]
    raw_rec = pod_project(Fte, m_raw, mu_raw, 2)
    orc_rec = oracle(Fte)
    d, j = circ_gap(w_te, w_tr)
    nn_rec = Ftr[j]
    r = {"seed": seed,
         "coupling_train": c_tr, "coupling_val": c_va, "coupling_test": c_te,
         "coupling_train_test_gap": c_te - c_tr,
         "raw_pod_test": nmse(raw_rec, Fte),
         "oracle_test": nmse(orc_rec, Fte),
         "nearest_copy_test": nmse(nn_rec, Fte),
         "coverage_max_gap_rad": coverage_max_gap(w_tr),
         "nearest_phase_distance": {"median": float(np.median(d)),
                                    "max": float(d.max())},
         "binned": {"coupling": binned(cpl_rec, Fte, d),
                    "raw_pod": binned(raw_rec, Fte, d),
                    "nearest_copy": binned(nn_rec, Fte, d)},
         **latent_rank(model, te, dev)}
    r["raw_over_coupling"] = r["raw_pod_test"] / c_te
    r["copy_over_coupling"] = r["nearest_copy_test"] / c_te
    r["te_error"] = {
        "raw_pod": te_error(lambda Z: pod_project(
            Z.reshape(len(Z), -1), m_raw, mu_raw, 2).reshape(Z.shape),
            te[:256], DELTAS),
        "coupling": te_error(cpl, te[:256], DELTAS)}
    print(f"  seed {seed:<3} cov_gap {r['coverage_max_gap_rad']:.4f}  "
          f"cpl {c_te:.3e} (tr {c_tr:.3e})  raw {r['raw_pod_test']:.3e}  "
          f"copy {r['nearest_copy_test']:.3e}  orc {r['oracle_test']:.1e}",
          flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"iid Uniform split: n_train {a.n_train}  n_val {a.n_val}  "
          f"n_test {a.n_test}   {a.seeds} seeds\n")
    runs = [one_seed(s, a, dev) for s in range(a.seeds)]

    def ms(key):
        v = np.array([r[key] for r in runs])
        return float(v.mean()), float(v.std())

    summ = {k: {"mean": ms(k)[0], "std": ms(k)[1]} for k in
            ("coupling_train", "coupling_val", "coupling_test",
             "raw_pod_test", "oracle_test", "nearest_copy_test",
             "raw_over_coupling", "copy_over_coupling",
             "coverage_max_gap_rad", "latent_top2_energy_fraction",
             "latent_projection_residual")}
    bins = {m: [{"mean": float(np.mean([r["binned"][m][i]["nmse"]
                                        for r in runs])),
                 "std": float(np.std([r["binned"][m][i]["nmse"]
                                      for r in runs]))} for i in range(3)]
            for m in ("coupling", "raw_pod", "nearest_copy")}
    te = {m: {d: {"mean": float(np.mean([r["te_error"][m][d] for r in runs])),
                  "std": float(np.std([r["te_error"][m][d] for r in runs]))}
              for d in DELTAS} for m in ("raw_pod", "coupling")}
    json.dump({"config": vars(a), "runs": runs, "summary": summ,
               "binned": bins, "te_error": te},
              open(os.path.join(HERE, "iid_results.json"), "w"), indent=2)

    print("\n=== mean +/- std over seeds ===")
    for k in ("coupling_train", "coupling_val", "coupling_test",
              "raw_pod_test", "nearest_copy_test", "oracle_test",
              "raw_over_coupling", "copy_over_coupling",
              "coverage_max_gap_rad"):
        print(f"  {k:<28}{summ[k]['mean']:.4e}  +/- {summ[k]['std']:.2e}")
    print("\n=== NMSE by distance to nearest training phase ===")
    print(f"  {'bin':<16}{'Coupling-FINE':<26}{'Raw POD':<26}"
          f"{'nearest-copy'}")
    for i, lab in enumerate(("closest third", "middle third",
                             "farthest third")):
        print(f"  {lab:<16}" + "".join(
            f"{bins[m][i]['mean']:.3e} +/- {bins[m][i]['std']:.1e}   "
            for m in ("coupling", "raw_pod", "nearest_copy")))
    print("\n=== E_TE, mean over seeds ===")
    print("  delta:  " + "  ".join(f"{d}{'*' if d % 2 else ' '}" for d in DELTAS))
    for m in ("coupling", "raw_pod"):
        print(f"  {m:<9}" + "  ".join(f"{te[m][d]['mean']:.0e}" for d in DELTAS))
    print("\n[wrote] iid_results.json")


if __name__ == "__main__":
    main()
