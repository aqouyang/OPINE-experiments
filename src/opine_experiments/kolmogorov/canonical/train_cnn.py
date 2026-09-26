#!/usr/bin/env python3
"""Train the Kolmogorov CNN baseline at several DOF budgets.

Training conventions are carried over unchanged from the channel-flow CNN
audit (channel2d/audit_cnn_convergence.py): AdamW,
cosine-annealed LR, best-validation checkpoint selection, and the same
per-epoch gradient-path diagnostics (encoder-linear, decoder-linear and whole
decoder gradient norms) so the gradient-distribution analysis is comparable
across projects.

Canonical NMSE = sum|recon - truth|^2 / sum|truth|^2 on the test split.
Validation is normalized MSE, used ONLY to pick the checkpoint; it is never
mixed into a reported test number.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def nmse_over_loader(model, loader, device):
    num = den = 0.0
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            r = model(xb)[0]
            num += float((r - xb).double().pow(2).sum())
            den += float(xb.double().pow(2).sum())
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dofs", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from opine_experiments.kolmogorov.canonical.dataset import make_datasets
    from opine_experiments.kolmogorov.canonical.models.cnn import KolmogorovCNN

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)
    tr, va, te, sigma = make_datasets(args.raw_dir, args.splits)
    mk = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=sh,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    train_ld, val_ld, test_ld = mk(tr, True), mk(va, False), mk(te, False)
    print(f"device {device}  sigma_train {sigma:.6f}  "
          f"train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    out = {"_test_metric": "sum|recon-truth|^2/sum|truth|^2, test split",
           "_selection": "validation normalized MSE only",
           "architecture": "KolmogorovCNN (JointCNN_YZ adapted: 1 channel, "
                           "128x128, circular padding both directions)",
           "epochs": args.epochs, "lr": args.lr,
           "batch_size": args.batch_size, "seed": args.seed,
           "sigma_train": sigma, "dofs": {}}

    for dof in args.dofs:
        print(f"\n{'='*66}\nCNN  D={dof}\n{'='*66}", flush=True)
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        model = KolmogorovCNN(input_hw=(128, 128), latent_dim=dof).to(device)
        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs)

        hist, best_val, best_ep, best_state = [], float("inf"), 0, None
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            run, nb, grad_info = 0.0, 0, None
            for xb in train_ld:
                xb = xb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = (model(xb)[0] - xb).pow(2).mean()
                loss.backward()
                if nb == 0:                      # gradient-path diagnostics
                    g_dec = [p.grad for p in model.decoder.parameters()
                             if p.grad is not None]
                    grad_info = {
                        "enc_linear_grad_norm":
                            float(model.enc_linear.weight.grad.norm()),
                        "dec_linear_grad_norm":
                            float(model.dec_linear.weight.grad.norm()),
                        "decoder_grad_norm": float(torch.cat(
                            [g.flatten() for g in g_dec]).norm()),
                    }
                opt.step()
                run += float(loss); nb += 1
            sched.step()

            model.eval()
            vnum = vden = 0.0
            with torch.no_grad():
                for xb in val_ld:
                    xb = xb.to(device, non_blocking=True)
                    r = model(xb)[0]
                    vnum += float((r - xb).double().pow(2).sum())
                    vden += float(xb.double().pow(2).sum())
            v = vnum / vden
            if v < best_val:
                best_val, best_ep = v, ep
                best_state = {k: t.detach().clone()
                              for k, t in model.state_dict().items()}
            hist.append({"epoch": ep, "train_mse_running": run / nb,
                         "val_normalized_mse": v,
                         "lr": opt.param_groups[0]["lr"], **grad_info})
            print(f"  ep {ep:3d}/{args.epochs}  train {run/nb:.6f}  "
                  f"val {v:.6f}  lr {opt.param_groups[0]['lr']:.2e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        model.load_state_dict(best_state)
        ck = os.path.join(args.output_dir, f"CNN{dof}_best.pt")
        torch.save(best_state, ck)
        test_nmse = nmse_over_loader(model, test_ld, device)
        train_nmse = nmse_over_loader(model, train_ld, device)
        out["dofs"][str(dof)] = {
            "n_params": n_params, "best_epoch": best_ep,
            "best_val_normalized_mse": best_val,
            "selected_test_nmse": test_nmse,
            "selected_train_nmse": train_nmse,
            "generalization_gap": test_nmse - train_nmse,
            "runtime_s": time.time() - t0,
            "checkpoint": ck, "history": hist,
        }
        print(f"  -> test NMSE {test_nmse:.8f}  (train {train_nmse:.8f}, "
              f"gap {test_nmse-train_nmse:+.8f})  best epoch {best_ep}",
              flush=True)

        # sample-0 original / reconstruction / error
        x0 = te[0].unsqueeze(0).to(device)
        with torch.no_grad():
            r0 = model(x0)[0]
        np.savez_compressed(
            os.path.join(args.output_dir, f"sample0_D{dof}.npz"),
            truth=x0[0, 0].cpu().numpy(), recon=r0[0, 0].cpu().numpy(),
            error=(r0 - x0)[0, 0].cpu().numpy())

    with open(os.path.join(args.output_dir, "cnn_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
