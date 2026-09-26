#!/usr/bin/env python3
"""Diagnose the y-z channel CNN-AE baseline, with emphasis on D=256.

Question: is the poor D=256 result a bug, an optimization failure, or
overfitting?  Nothing about the model or the scientific setup is changed --
this script only measures.

Sections
  1  train / val / test NMSE for D=32/64/128/256 under ONE metric
  3  tiny-set memorisation test at D=256 (implementation correctness)
  4  evaluation-pipeline audit, executed as runtime assertions
  5  component-wise (u,v,w) test NMSE at D=256, CNN and POD

Section 2 (learning curves) is read off the existing per-epoch histories and
plotted locally; nothing needs re-running for it.
"""
import argparse, json, os, sys, time

import numpy as np
import torch

NY, NZ = 128, 64
COMPONENTS = ("u", "v", "w")


# --------------------------------------------------------------- metrics
def nmse_global(rec, x):
    """Canonical project metric: sum((xhat-x)^2) / sum(x^2), ONE ratio."""
    return float(np.sum((rec - x) ** 2) / np.sum(x ** 2))


def nmse_per_sample_mean(rec, x):
    """Mean over samples of the per-snapshot ratio.  NOT the project metric."""
    ax = tuple(range(1, x.ndim))
    return float(np.mean(np.sum((rec - x) ** 2, axis=ax)
                         / np.sum(x ** 2, axis=ax)))


def mean_pixel_mse(rec, x):
    """Per-pixel mean squared error.  This is what the old audit called
    'val_normalized_mse' and used for checkpoint selection."""
    return float(np.mean((rec - x) ** 2))


def nmse_per_component(rec, x):
    """Component-resolved global NMSE; axis 1 is (u, v, w)."""
    return {name: nmse_global(rec[:, c], x[:, c])
            for c, name in enumerate(COMPONENTS)}


@torch.no_grad()
def cnn_reconstruct(model, X, device, batch=64):
    model.eval()
    assert not model.training, "model.eval() did not take effect"
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().to(device)
        outs.append(model(xb)[0].double().cpu().numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-dir", required=True)
    ap.add_argument("--preprocessing-file", required=True)
    ap.add_argument("--ckpt-dir", required=True,
                    help="dir holding CNN32/64/128_best.pt")
    ap.add_argument("--ckpt-d256", required=True,
                    help="dir holding CNN256_best.pt")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dofs", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--tiny-n", type=int, default=32)
    ap.add_argument("--tiny-epochs", type=int, default=3000)
    ap.add_argument("--tiny-dof", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    from opine_experiments.channel2d.dataset_joint import (
        ChannelFlowYZJointDatasetLocal)
    from opine_experiments.channel2d.train_latent_rank import (
        load_joint_preprocessing)
    from opine_experiments.channel2d.cnn_yz_joint import JointCNN_YZ
    from opine_experiments.channel2d.pod_yz_joint import (
        JointSymmetryPOD, JointUnrestrictedPOD)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.output_dir, exist_ok=True)
    mp, ps, _, _, _ = load_joint_preprocessing(a.preprocessing_file)

    def ds(split):
        return ChannelFlowYZJointDatasetLocal(
            os.path.join(a.extracted_dir, f"{split}.npz"), mp, ps, NY)

    splits = {}
    for s in ("train", "val", "test"):
        d = ds(s)
        splits[s] = np.stack([d[i].numpy() for i in range(len(d))]
                             ).astype(np.float64)
        print(f"{s:5s} {splits[s].shape}", flush=True)
    rep = {"_device": str(dev),
           "_nmse_formula": "sum((xhat-x)^2) / sum(x^2), summed over the whole "
                            "split at once (single global ratio)",
           "_split_sizes": {k: int(v.shape[0]) for k, v in splits.items()},
           "_normalization": "x'_c(y,z) = (u_c(y,z) - mean_c(z)) / rms_c(z), "
                             "both from the TRAINING split only; identical "
                             "object for train/val/test and for every method",
           "_component_order": list(COMPONENTS)}

    # =================================================== 4  pipeline audit
    print("\n=== 4  evaluation pipeline audit ===", flush=True)
    d1, d2 = ds("test"), ds("test")
    audit = {"dataset_deterministic": bool(
        np.array_equal(d1[0].numpy(), d2[0].numpy())
        and np.array_equal(d1[len(d1) - 1].numpy(), d2[len(d2) - 1].numpy())),
        "mean_square_per_split": {k: float(np.mean(v ** 2))
                                  for k, v in splits.items()},
        "_why_denominator_matters":
            "the old audit selected and reported train/val with "
            "mean_pixel_mse and test with sum/sum NMSE; they differ by "
            "exactly mean(x^2) on that split, which is 1 only by "
            "construction on train"}

    # =================================================== 1  train/val/test
    print("\n=== 1  train / val / test NMSE, one metric ===", flush=True)
    rows = []
    r256_cnn = None
    for D in a.dofs:
        cdir = a.ckpt_d256 if D == 256 else a.ckpt_dir
        cp = os.path.join(cdir, f"CNN{D}_best.pt")
        sd = torch.load(cp, map_location="cpu")
        lat = sd["enc_linear.weight"].shape[0]
        assert lat == D, f"checkpoint {cp} has latent {lat}, expected {D}"
        model = JointCNN_YZ(input_hw=(NY, NZ), latent_dim=D).to(dev)
        model.load_state_dict(sd, strict=True)
        model.eval()
        bns = [m for m in model.modules()
               if isinstance(m, torch.nn.BatchNorm2d)]
        bn_ok = all((not m.training) and m.running_mean is not None
                    for m in bns)
        r = {s: cnn_reconstruct(model, splits[s], dev) for s in splits}
        row = {"dof": D,
               "n_params": sum(p.numel() for p in model.parameters()),
               "checkpoint": cp, "checkpoint_latent_dim": int(lat),
               "n_batchnorm": len(bns), "batchnorm_eval_mode": bool(bn_ok),
               "nmse": {s: nmse_global(r[s], splits[s]) for s in splits},
               "nmse_per_sample_mean": {
                   s: nmse_per_sample_mean(r[s], splits[s]) for s in splits},
               "mean_pixel_mse": {s: mean_pixel_mse(r[s], splits[s])
                                  for s in splits}}
        row["gap_test_minus_train"] = (row["nmse"]["test"]
                                       - row["nmse"]["train"])
        rows.append(row)
        print(f"  D={D:<4} train {row['nmse']['train']:.6f}  val "
              f"{row['nmse']['val']:.6f}  test {row['nmse']['test']:.6f}  "
              f"gap {row['gap_test_minus_train']:+.4f}  bn_eval={bn_ok}",
              flush=True)
        if D == 256:
            r256_cnn = r["test"]
        del model, r
        torch.cuda.empty_cache()
    rep["s1_cnn_train_val_test"] = rows

    print("\n  POD baselines, same metric:", flush=True)
    pod_rows = []
    r256_pod = r256_sym = None
    for D in a.dofs:
        pod = JointUnrestrictedPOD(dof=D); pod.fit(splits["train"])
        sym = JointSymmetryPOD(target_dof=D, mean_convention="safe")
        sym.fit(splits["train"])
        rp = {s: pod.reconstruct(splits[s]) for s in splits}
        rs = {s: sym.reconstruct(splits[s]) for s in splits}
        pr = {"dof": D,
              "ordinary_POD": {s: nmse_global(rp[s], splits[s])
                               for s in splits},
              "SymPOD_safe": {s: nmse_global(rs[s], splits[s])
                              for s in splits}}
        pod_rows.append(pr)
        print(f"  D={D:<4} POD  train {pr['ordinary_POD']['train']:.6f}  val "
              f"{pr['ordinary_POD']['val']:.6f}  test "
              f"{pr['ordinary_POD']['test']:.6f}   | SymPOD test "
              f"{pr['SymPOD_safe']['test']:.6f}", flush=True)
        if D == 256:
            r256_pod, r256_sym = rp["test"], rs["test"]
    rep["s1_pod_train_val_test"] = pod_rows
    rep["s4_audit"] = audit

    # =================================================== 5  component-wise
    print("\n=== 5  component-wise test NMSE, D=256 ===", flush=True)
    comp = {"CNN": nmse_per_component(r256_cnn, splits["test"]),
            "ordinary_POD": nmse_per_component(r256_pod, splits["test"]),
            "SymPOD_safe": nmse_per_component(r256_sym, splits["test"]),
            "_energy_share_of_test": {
                c: float(np.sum(splits["test"][:, i] ** 2)
                         / np.sum(splits["test"] ** 2))
                for i, c in enumerate(COMPONENTS)}}
    for m in ("CNN", "ordinary_POD", "SymPOD_safe"):
        print(f"  {m:<14} " + "  ".join(
            f"{c} {comp[m][c]:.6f}" for c in COMPONENTS), flush=True)
    print("  energy share   " + "  ".join(
        f"{c} {comp['_energy_share_of_test'][c]:.4f}" for c in COMPONENTS),
        flush=True)
    zr = {}
    for name, rr in (("CNN", r256_cnn), ("ordinary_POD", r256_pod),
                     ("SymPOD_safe", r256_sym)):
        num = np.sum((rr - splits["test"]) ** 2, axis=(0, 1, 2))
        den = np.sum(splits["test"] ** 2, axis=(0, 1, 2))
        zr[name] = (num / den).tolist()
    comp["z_resolved_nmse"] = zr
    rep["s5_components_d256"] = comp

    # =================================================== 3  tiny-set memo
    print(f"\n=== 3  tiny-set memorisation, D={a.tiny_dof}, "
          f"n={a.tiny_n} ===", flush=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    Xt = splits["train"][:a.tiny_n]
    xb = torch.from_numpy(Xt).float().to(dev)
    tm = JointCNN_YZ(input_hw=(NY, NZ), latent_dim=a.tiny_dof).to(dev)
    opt = torch.optim.AdamW(tm.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.tiny_epochs)
    hist, t0 = [], time.time()
    for ep in range(1, a.tiny_epochs + 1):
        tm.train()
        opt.zero_grad(set_to_none=True)
        loss = (tm(xb)[0] - xb).pow(2).mean()
        loss.backward(); opt.step(); sch.step()
        if ep % 100 == 0 or ep == 1:
            with torch.no_grad():
                tm.eval()
                n_eval = nmse_global(tm(xb)[0].double().cpu().numpy(), Xt)
                tm.train()
                n_tr = nmse_global(tm(xb)[0].double().cpu().numpy(), Xt)
            hist.append({"epoch": ep, "loss": float(loss),
                         "nmse_eval_mode": n_eval, "nmse_train_mode": n_tr})
            if ep % 500 == 0 or ep == 1:
                print(f"  ep {ep:5d}  loss {float(loss):.3e}  NMSE(eval) "
                      f"{n_eval:.3e}  NMSE(train-mode) {n_tr:.3e}", flush=True)
    rep["s3_tiny_memorisation"] = {
        "n_samples": a.tiny_n, "dof": a.tiny_dof, "epochs": a.tiny_epochs,
        "final_nmse_eval_mode": hist[-1]["nmse_eval_mode"],
        "final_nmse_train_mode": hist[-1]["nmse_train_mode"],
        "final_loss": hist[-1]["loss"], "runtime_s": time.time() - t0,
        "_note": "eval mode uses BatchNorm running statistics; train mode "
                 "uses batch statistics.  A large gap between the two on the "
                 "SAME data means the BatchNorm running estimates, not the "
                 "weights, limit the reconstruction.",
        "history": hist}
    print(f"  final NMSE eval-mode {hist[-1]['nmse_eval_mode']:.4e}  "
          f"train-mode {hist[-1]['nmse_train_mode']:.4e}", flush=True)

    out = os.path.join(a.output_dir, "cnn_d256_diagnosis.json")
    json.dump(rep, open(out, "w"), indent=1)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
