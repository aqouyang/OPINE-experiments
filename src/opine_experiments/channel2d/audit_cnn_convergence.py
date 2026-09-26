#!/usr/bin/env python3
"""Controlled CNN convergence audit for D=32/64/128.

Question: is the CNN baseline (especially D=128) optimization-limited, or has
it genuinely converged and started overfitting?

Runs the same JointCNN_YZ architecture on the same data, split and
normalization as the production runs, with a longer budget and the full
train/validation history recorded every epoch.  Checkpoints are selected on
validation normalized-MSE only; the reported figure of merit is canonical
test NMSE = sum((xhat-x)^2)/sum(x^2) on the 631-sample test split.

The architecture is NOT modified.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

NY, NZ = 128, 64


def test_nmse(model, loader, device):
    """Canonical aggregate NMSE = sum((xhat-x)^2) / sum(x^2)."""
    num = den = 0.0
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            r = model(xb)[0]
            num += float((r - xb).double().pow(2).sum())
            den += float(xb.double().pow(2).sum())
    return num / den


def mean_mse(model, loader, device):
    """Per-pixel MSE on RMS-normalized data (used for selection only)."""
    tot = n = 0.0
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            tot += float((model(xb)[0] - xb).double().pow(2).mean()) * xb.shape[0]
            n += xb.shape[0]
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-dir", required=True)
    ap.add_argument("--preprocessing-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dofs", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="If set, save the validation-selected best CNN "
                         "state per DOF as CNN{dof}_best.pt")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    from opine_experiments.channel2d.dataset_joint import (
        ChannelFlowYZJointDatasetLocal)
    from opine_experiments.channel2d.train_latent_rank import (
        load_joint_preprocessing)
    from opine_experiments.channel2d.cnn_yz_joint import JointCNN_YZ

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    mean_profiles, plane_stds, _, _, _ = load_joint_preprocessing(
        args.preprocessing_file)

    def ds(split):
        return ChannelFlowYZJointDatasetLocal(
            os.path.join(args.extracted_dir, f"{split}.npz"),
            mean_profiles, plane_stds, NY)

    train_ds, val_ds, test_ds = ds("train"), ds("val"), ds("test")
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True)
    train_eval_ld = torch.utils.data.DataLoader(train_ds, batch_size=64)
    val_ld = torch.utils.data.DataLoader(val_ds, batch_size=64)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=64)
    print(f"Device={device}  train={len(train_ds)} val={len(val_ds)} "
          f"test={len(test_ds)}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "_test_metric": "sum((xhat-x)^2)/sum(x^2), 631-sample test split",
        "_selection": "validation normalized-MSE only",
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "seed": args.seed, "architecture": "JointCNN_YZ (unmodified)",
        "dofs": {},
    }

    for dof in args.dofs:
        print(f"\n{'='*66}\nCNN DOF={dof}\n{'='*66}", flush=True)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = JointCNN_YZ(input_hw=(NY, NZ), latent_dim=dof).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs)

        hist = []
        best_val, best_epoch, best_state = float("inf"), 0, None
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            run, nb = 0.0, 0
            for xb in train_ld:
                xb = xb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = (model(xb)[0] - xb).pow(2).mean()
                loss.backward()
                # record gradient health on the latent and decoder paths
                if nb == 0:
                    g_enc_lin = model.enc_linear.weight.grad
                    g_dec_lin = model.dec_linear.weight.grad
                    g_dec = [p.grad for p in model.decoder.parameters()
                             if p.grad is not None]
                    grad_info = {
                        "enc_linear_grad_norm": float(g_enc_lin.norm()),
                        "dec_linear_grad_norm": float(g_dec_lin.norm()),
                        "decoder_grad_norm": float(torch.cat(
                            [g.flatten() for g in g_dec]).norm()),
                    }
                opt.step()
                run += float(loss)
                nb += 1
            sched.step()

            tr = run / nb
            v = mean_mse(model, val_ld, device)
            if v < best_val:
                best_val, best_epoch = v, ep
                best_state = {k: t.detach().clone()
                              for k, t in model.state_dict().items()}
            row = {"epoch": ep, "train_mse_running": tr,
                   "val_normalized_mse": v,
                   "lr": opt.param_groups[0]["lr"], **grad_info}
            hist.append(row)
            if ep <= 5 or ep % 10 == 0 or ep == args.epochs:
                tr_full = mean_mse(model, train_eval_ld, device)
                row["train_normalized_mse_eval"] = tr_full
                print(f"E{ep:4d}: train={tr:.6f} train_eval={tr_full:.6f} "
                      f"val={v:.6f} gap={v-tr_full:+.5f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)

        final_test = test_nmse(model, test_ld, device)
        final_train = mean_mse(model, train_eval_ld, device)
        model.load_state_dict(best_state)
        if args.checkpoint_dir:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            cp = os.path.join(args.checkpoint_dir, f"CNN{dof}_best.pt")
            torch.save(best_state, cp)
            print(f"  saved checkpoint: {cp}", flush=True)
        sel_test = test_nmse(model, test_ld, device)
        sel_train = mean_mse(model, train_eval_ld, device)

        d = {
            "n_params": n_params,
            "best_epoch": best_epoch,
            "best_val_normalized_mse": best_val,
            "selected_test_nmse": sel_test,
            "selected_train_normalized_mse": sel_train,
            "final_epoch_test_nmse": final_test,
            "final_epoch_train_normalized_mse": final_train,
            "generalization_gap_at_selection": best_val - sel_train,
            "runtime_s": time.time() - t0,
            "history": hist,
        }
        # convergence verdict
        tail = [h["val_normalized_mse"] for h in hist[-20:]]
        d["val_improved_in_last_20_epochs"] = float(min(tail) < best_val * 0.999)
        d["epochs_since_best"] = args.epochs - best_epoch
        d["verdict"] = (
            "optimization-limited (validation still improving at the end)"
            if best_epoch > args.epochs * 0.9 else
            "converged then overfit (validation bottomed out early, "
            "train loss kept falling)")
        report["dofs"][str(dof)] = d
        print(f"  best_epoch={best_epoch}/{args.epochs}  "
              f"selected test NMSE={sel_test:.8f}  "
              f"train={sel_train:.6f} val={best_val:.6f}", flush=True)
        print(f"  verdict: {d['verdict']}", flush=True)

    out = os.path.join(args.output_dir, "cnn_convergence_audit.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
