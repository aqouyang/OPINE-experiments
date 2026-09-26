#!/usr/bin/env python3
"""Collect the 2-D y-z channel train/val/test NMSE for POD, CNN and OPINE.

Nothing is retrained and no NMSE is recomputed -- every number already exists
under the one canonical metric and is only gathered here into a single table
so the summary figure and any later convention change read from one place.

  POD    SymPOD-safe.  This is the ONE POD baseline for the project;
         unrestricted POD is not a baseline and is not carried here.  From
         cnn_diagnosis/cnn_d256_diagnosis.json, which fitted it on the full
         training split and evaluated train/val/test with
         sum((xhat-x)^2)/sum(x^2).
  CNN    JointCNN_YZ, validation-selected checkpoint, same file.
  OPINE  coupling_fine/D*/couplingyz_D*_results.json, which already stores
         selected_train_nmse and selected_test_nmse under the same metric.

Original FINE and iResNet FINE are deliberately excluded.
"""
import argparse, csv, json, os

PAIR = {"POD": 1, "OPINE": 1, "CNN": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-diagnosis", required=True)
    ap.add_argument("--opine-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--dofs", type=int, nargs="+",
                    default=[32, 64, 128, 256, 512])
    ap.add_argument("--extracted-dir", default=None,
                    help="needed only for ranks absent from the diagnosis "
                         "file, where SymPOD is refitted here")
    ap.add_argument("--preprocessing-file", default=None)
    a = ap.parse_args()

    d = json.load(open(a.cnn_diagnosis))
    cnn = {r["dof"]: r for r in d["s1_cnn_train_val_test"]}
    pod = {r["dof"]: r for r in d["s1_pod_train_val_test"]}

    extra = {}
    missing = [D for D in a.dofs if D not in pod]
    if missing and a.extracted_dir:
        # Ranks the diagnosis file never covered.  SymPOD is refitted on the
        # same full training split with the same safe-mean convention, and
        # evaluated with the same metric, so the column stays homogeneous.
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        import numpy as np
        from opine_experiments.channel2d.dataset_joint import (
            ChannelFlowYZJointDatasetLocal)
        from opine_experiments.channel2d.train_latent_rank import (
            load_joint_preprocessing)
        from opine_experiments.channel2d.pod_yz_joint import (
            JointSymmetryPOD)
        mp, ps, _, _, _ = load_joint_preprocessing(a.preprocessing_file)
        X = {}
        for s_ in ("train", "val", "test"):
            ds = ChannelFlowYZJointDatasetLocal(
                os.path.join(a.extracted_dir, f"{s_}.npz"), mp, ps, 128)
            X[s_] = np.stack([ds[i].numpy() for i in range(len(ds))]
                             ).astype(np.float64)
        nm_ = lambda r, x: float(np.sum((r - x) ** 2) / np.sum(x ** 2))
        for D in missing:
            sy = JointSymmetryPOD(target_dof=D, mean_convention="safe")
            sy.fit(X["train"])
            extra[D] = {
                "SymPOD_safe": {k: nm_(sy.reconstruct(X[k]), X[k])
                                for k in X},
                "actual_dof": int(sy.actual_dof)}
            print(f"  refitted POD at rank {D}: SymPOD test "
                  f"{extra[D]['SymPOD_safe']['test']:.6f} "
                  f"(actual_dof {extra[D]['actual_dof']})")

    rows, flat = [], []
    for D in a.dofs:
        row = {"dof": D}
        src = pod.get(D) or extra.get(D)
        if src:
            s = src["SymPOD_safe"]
            row["POD"] = {"train": s["train"], "val": s["val"],
                          "test": s["test"], "n_params": 0,
                          "variant": "SymPOD-safe (the POD baseline)"}
        if D in cnn:
            c = cnn[D]["nmse"]
            row["CNN"] = {"train": c["train"], "val": c["val"],
                          "test": c["test"], "n_params": cnn[D]["n_params"],
                          "checkpoint": cnn[D]["checkpoint"]}
        p = os.path.join(a.opine_dir, f"D{D}", f"couplingyz_D{D}_results.json")
        if os.path.isfile(p):
            o = json.load(open(p))
            row["OPINE"] = {"train": o["selected_train_nmse"],
                            "val": o["best_val_nmse"],
                            "test": o["selected_test_nmse"],
                            "n_params": o["n_trainable_params"],
                            "best_round": o["best_round"]}
        rows.append(row)
        for meth in ("POD", "CNN", "OPINE"):
            if meth not in row:
                continue
            r = row[meth]
            flat.append({"dataset": "channel_yz_2d", "method": meth,
                         "stored_rank": D, "plotted_D": D // PAIR[meth],
                         "pair_divisor": PAIR[meth],
                         "train_nmse": r["train"], "val_nmse": r["val"],
                         "test_nmse": r["test"],
                         "gap_test_minus_train": r["test"] - r["train"],
                         "n_params": r.get("n_params", ""),
                         "note": r.get("variant", "")})

    out = {"_metric": "sum((xhat-x)^2)/sum(x^2), whole split at once",
           "_splits": {"train": 2940, "val": 630, "test": 631},
           "_excluded": ["Original FINE", "iResNet FINE",
                         "unrestricted POD"],
           "_pod_definition": "POD means SymPOD-safe throughout; the "
                              "unrestricted variant is not a baseline",
           "_sources": {"POD/CNN": a.cnn_diagnosis, "OPINE": a.opine_dir},
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
           "rows": rows}
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(out, open(a.out_json, "w"), indent=1)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)
    print("wrote", a.out_json)
    print("wrote", a.out_csv)
    print(f"{'method':<7}{'rank':>6}{'D':>6}{'train':>10}{'val':>10}"
          f"{'test':>10}{'gap':>9}")
    for r in flat:
        print(f"{r['method']:<7}{r['stored_rank']:>6}{r['plotted_D']:>6}"
              f"{r['train_nmse']:>10.6f}{r['val_nmse']:>10.6f}"
              f"{r['test_nmse']:>10.6f}{r['gap_test_minus_train']:>+9.4f}")


if __name__ == "__main__":
    main()
