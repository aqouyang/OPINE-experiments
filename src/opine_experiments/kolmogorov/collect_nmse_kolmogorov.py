#!/usr/bin/env python3
"""Collect the Kolmogorov train/test NMSE for POD, CNN and OPINE.

Nothing is retrained and no NMSE is recomputed -- every number already exists
under the one canonical metric, sum|xhat-x|^2/sum|x|^2 over the whole split,
and is only gathered here so the summary figure and any later convention
change read from one place.

  POD    Unrestricted global POD.  This is the ONE POD baseline for
         Kolmogorov, for two reasons that both point the same way:
         results/canonical/README.md marks the PODxTE / PODxC4 symmetry line
         "discontinued, do not cite" while POD_test_nmse is current, and
         run_pod.py documents the choice on physical grounds -- the
         -k_f cos(k_f y) forcing makes the statistics explicitly y-dependent,
         so a symmetry-constrained POD of the channel SymPOD kind has no
         y-homogeneity to exploit here.  It is also exactly the OPINE
         round-0 bottleneck, so the two curves share a starting point.
         Shown on test only, as in the 2-D and 3-D figures.
  CNN    canonical/train_cnn.py, validation-selected checkpoint.  The table
         stores test and generalization_gap = test - train, so train is
         recovered exactly as test - gap.  Ranks trained after the table was
         frozen come from --cnn-results, train_cnn.py's own output file,
         which carries the same quantities under their own names.
  OPINE  coupling_D*_results.json, which already stores selected_train_nmse
         and selected_test_nmse under the same metric.

Original FINE and iResNet FINE are deliberately excluded.
"""
import argparse, csv, json, os

PAIR = {"POD": 1, "OPINE": 1, "CNN": 1}
POD_KEY = "POD"

# OPINE result files: D64 lives in its own directory, the rest under
# coupling_all/.  Both are the same alpha=0.03, 61760-parameter configuration.
OPINE_PATHS = [
    "{root}/diagnostics/coupling_all/coupling_fine_D{D}/coupling_D{D}_results.json",
    "{root}/diagnostics/coupling_D{D}/coupling_D{D}_results.json",
    "{root}/diagnostics/coupling_fine_D{D}/coupling_D{D}_results.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True,
                    help="the results directory")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--dofs", type=int, nargs="+",
                    default=[32, 64, 128, 256, 512])
    ap.add_argument("--cnn-results", default=None,
                    help="cnn_results.json written by canonical/train_cnn.py; "
                         "supplies CNN at ranks that postdate the frozen "
                         "comparison table")
    ap.add_argument("--pod-ext", default=None,
                    help="pod_ext/pod_results.json from the extended-rank "
                         "rerun of run_pod.py; when given it supplies POD at "
                         "every rank from ONE exact decomposition and the "
                         "frozen comparison table is used only as a check")
    a = ap.parse_args()

    tbl = json.load(open(os.path.join(a.results_root, "canonical",
                                      "comparison_table.json")))
    t = tbl["table"]
    ext = json.load(open(a.pod_ext))["table"] if a.pod_ext else {}
    live = (json.load(open(a.cnn_results))["dofs"] if a.cnn_results else {})

    rows, flat = [], []
    for D in a.dofs:
        k = str(D)
        row = {"dof": D}
        if k in ext:
            e = ext[k]
            row["POD"] = {"train": e["train_nmse"], "val": e["val_nmse"],
                          "test": e["test_nmse"], "n_params": 0,
                          "variant": "unrestricted global POD (the POD "
                                     "baseline); exact eigh of the 16384^2 "
                                     "spatial covariance",
                          "captured_energy": e.get(
                              "captured_energy_fraction"),
                          "source": a.pod_ext}
            if k in t and "POD_test_nmse" in t[k]:
                # cross-check against the frozen table where it overlaps
                row["POD"]["frozen_table_test"] = t[k]["POD_test_nmse"]
                row["POD"]["frozen_table_delta"] = (
                    e["test_nmse"] - t[k]["POD_test_nmse"])
        elif k in t and f"{POD_KEY}_test_nmse" in t[k]:
            row["POD"] = {
                # The frozen table stored test only for POD.
                "train": None, "val": None,
                "test": t[k][f"{POD_KEY}_test_nmse"],
                "n_params": t[k].get(f"{POD_KEY}_params", 0),
                "variant": "unrestricted global POD (the POD baseline)",
                "captured_energy": t[k].get("POD_captured_energy"),
                "source": "canonical/comparison_table.json"}
        if k in t and "CNN_test_nmse" in t[k]:
            te, gap = t[k]["CNN_test_nmse"], t[k]["CNN_gen_gap"]
            row["CNN"] = {"train": te - gap, "val": None, "test": te,
                          "n_params": t[k]["CNN_params"],
                          "_train_note": "test - generalization_gap, which "
                                         "train_cnn.py defines as test-train"}
        elif k in live:
            c = live[k]
            row["CNN"] = {"train": c["selected_train_nmse"],
                          "val": None, "test": c["selected_test_nmse"],
                          "n_params": c["n_params"],
                          "best_epoch": c["best_epoch"],
                          "source": a.cnn_results}
        for pat in OPINE_PATHS:
            p = pat.format(root=a.results_root, D=D)
            if os.path.isfile(p):
                o = json.load(open(p))
                row["OPINE"] = {"train": o["selected_train_nmse"],
                                "val": o["best_val_nmse"],
                                "test": o["selected_test_nmse"],
                                "n_params": o["n_trainable_params"],
                                "best_round": o["best_round"],
                                "alpha": o["alpha"], "source": p}
                break
        rows.append(row)
        for meth in ("POD", "CNN", "OPINE"):
            if meth not in row:
                continue
            r = row[meth]
            tr, te = r["train"], r["test"]
            flat.append({"dataset": "kolmogorov_2d", "method": meth,
                         "stored_rank": D, "plotted_D": D // PAIR[meth],
                         "pair_divisor": PAIR[meth],
                         "train_nmse": "" if tr is None else tr,
                         "val_nmse": "" if r["val"] is None else r["val"],
                         "test_nmse": te,
                         "gap_test_minus_train": "" if tr is None else te - tr,
                         "n_params": r.get("n_params", ""),
                         "note": r.get("variant", "")})

    out = {"_metric": tbl["_metric"],
           "_split": tbl["_split"],
           "_sigma_train": tbl["sigma_train"],
           "_grid": "128 x 128 vorticity, periodic in x and y",
           "_excluded": ["Original FINE", "iResNet FINE",
                         "PODxTE / PODxC4 (discontinued symmetry line, "
                         "flagged do-not-cite in canonical/README.md)"],
           "_pod_definition":
               "POD means the unrestricted global POD, which is the live "
               "Kolmogorov POD and the OPINE round-0 bottleneck.  The "
               "symmetry-constrained PODxTE / PODxC4 line is discontinued and "
               "is not used.  POD is plotted on test only.",
           "_dimension_convention": {
               "POD": "reported D = retained real rank",
               "OPINE": "reported D = retained real rank",
               "CNN": "reported D = latent dimension",
               "_why": "the retained real dimension is reported directly for "
                       "every method.  The earlier D = stored rank / 2 "
                       "mode-pair convention is withdrawn: the pairing is "
                       "enforced by construction only in the banded y-z "
                       "bottleneck, and elsewhere it is a finite-sample "
                       "property of the empirical covariance rather than a "
                       "guaranteed one, which is not a sufficient basis for "
                       "halving a reported dimension."},
           "_sources": {"POD": a.pod_ext or "canonical/comparison_table.json",
                        "CNN": "canonical/comparison_table.json",
                        "OPINE": OPINE_PATHS},
           "rows": rows}
    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    json.dump(out, open(a.out_json, "w"), indent=1)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)
    print("wrote", a.out_json)
    print("wrote", a.out_csv)
    print(f"{'method':<7}{'rank':>6}{'D':>6}{'train':>11}{'test':>11}"
          f"{'gap':>10}{'params':>10}")
    for r in flat:
        tr = f"{r['train_nmse']:.6f}" if r["train_nmse"] != "" else "--"
        gp = f"{r['gap_test_minus_train']:+.4f}" if r["gap_test_minus_train"] != "" else "--"
        print(f"{r['method']:<7}{r['stored_rank']:>6}{r['plotted_D']:>6}"
              f"{tr:>11}{r['test_nmse']:>11.6f}{gp:>10}{r['n_params']:>10}")


if __name__ == "__main__":
    main()
