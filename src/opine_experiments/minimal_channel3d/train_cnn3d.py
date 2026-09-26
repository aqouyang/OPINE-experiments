#!/usr/bin/env python3
"""CNN-AE baseline on the 3-D minimal channel.  One DOF per invocation.

Protocol is the established channel CNN protocol, unchanged: AdamW, lr 1e-3,
batch 32, 50 epochs, CosineAnnealingLR, seed 42, loss = mean((xhat-x)^2),
checkpoint selected on validation ONLY.

Reported figure of merit is the project metric, identical to POD and OPINE:

    NMSE = sum((xhat - x)^2) / sum(x^2)

summed over the whole split at once.  train, val and test are all reported
under that single metric -- the 2-D audit mixed per-pixel MSE into the
train/val columns and that is not repeated here.

The full per-epoch history is recorded so overfitting is visible directly.
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.append(ROOT)
sys.path.insert(0, HERE)
from dataset3d import make_datasets, N_FEAT, SPLITS
from cnn3d import JointCNN3D


def sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except OSError:
        return "unavailable"


@torch.no_grad()
def nmse(model, loader, device):
    """sum((xhat-x)^2)/sum(x^2), accumulated in float64 over the split."""
    model.eval()
    num = den = 0.0
    for xb in loader:
        x = xb.to(device, non_blocking=True)
        r = model(x)[0]
        num += float((r - x).double().pow(2).sum())
        den += float(x.double().pow(2).sum())
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dof", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--stats-cache", required=True)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--num-down", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--preload", type=int, default=1,
                    help="hold the normalized splits in RAM (~14 GB); the "
                         "npy is 15.7 GB on /scratch and 50 epochs of memmap "
                         "reads would be IO-bound")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stop-val-improvement", type=float, default=None,
                    help="stop once the best validation NMSE has improved by "
                         "less than this over --stop-patience epochs.  OFF by "
                         "default: the canonical protocol is a fixed 50-epoch "
                         "cosine run and every other DOF used it.  A run that "
                         "stops early is NOT on that protocol and records the "
                         "fact in its result file")
    ap.add_argument("--stop-patience", type=int, default=3)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)
    tag = f"cnn3d_D{a.dof}"

    t_load = time.time()
    tr, va, te, mean, std = make_datasets(a.stats_cache, preload=bool(a.preload))
    mk = lambda ds, sh, bs: torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=sh,
        num_workers=0 if a.preload else a.workers, pin_memory=True)
    train_ld = mk(tr, True, a.batch_size)
    train_eval_ld = mk(tr, False, a.eval_batch)
    val_ld = mk(va, False, a.eval_batch)
    test_ld = mk(te, False, a.eval_batch)
    print(f"[{tag}] data ready in {time.time()-t_load:.0f}s  "
          f"train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    model = JointCNN3D(input_shape=(64, 32, 32), latent_dim=a.dof,
                       base_channels=a.base_channels,
                       num_down=a.num_down).to(dev)
    groups = model.param_groups()
    print(f"[{tag}] N={N_FEAT}  params {groups['total']:,}  "
          f"flat={groups['flat_dim']}  latent-adjacent "
          f"{groups['latent_adjacent']:,} ({groups['latent_adjacent_frac']*100:.1f}%)"
          f"  params/train-snapshot {groups['total']/len(tr):.1f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    res = {"tag": tag, "dof": a.dof, "ambient_dim": N_FEAT,
           "n_params": groups["total"], "param_groups": groups,
           "_metric": "sum((xhat-x)^2)/sum(x^2), whole split at once",
           "_split": f"chronological {SPLITS}",
           "_normalization": "per (component, z) mean and rms from the "
                             "training split only; homogeneous x,y averaged",
           "_selection": "validation NMSE only; never mixed into test",
           "_padding": "x,y circular; z zero (architectural choice, not a "
                       "physical boundary condition)",
           "code_revision": {"cnn": sha(os.path.join(HERE, "cnn3d.py")),
                             "dataset": sha(os.path.join(HERE,
                                                         "dataset3d.py"))},
           "config": vars(a), "history": []}

    best_val, best_ep, best_state = float("inf"), 0, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for xb in train_ld:
            x = xb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = (model(x)[0] - x).pow(2).mean()
            loss.backward()
            opt.step()
            run += float(loss); nb += 1
        sched.step()
        v = nmse(model, val_ld, dev)
        row = {"epoch": ep, "train_loss_running": run / nb, "val_nmse": v,
               "lr": opt.param_groups[0]["lr"]}
        if ep <= 5 or ep % 5 == 0 or ep == a.epochs:
            row["train_nmse_eval"] = nmse(model, train_eval_ld, dev)
        res["history"].append(row)
        if v < best_val:
            best_val, best_ep = v, ep
            best_state = {k: t.detach().clone()
                          for k, t in model.state_dict().items()}
        print(f"[{tag}] E{ep:3d}/{a.epochs} train_loss {run/nb:.6f}  "
              f"val {v:.6f}" + (f"  train {row['train_nmse_eval']:.6f}"
                                if "train_nmse_eval" in row else "")
              + f"  ({time.time()-t0:.0f}s)", flush=True)

        # Optional early stop, measured on the BEST validation NMSE rather
        # than the last one, so a single noisy epoch cannot end the run.
        if a.stop_val_improvement is not None and ep > a.stop_patience:
            ref = min(r["val_nmse"] for r in
                      res["history"][:-a.stop_patience])
            gain = ref - best_val
            if gain < a.stop_val_improvement:
                stopped = {"epoch": ep, "patience": a.stop_patience,
                           "threshold": a.stop_val_improvement,
                           "best_val_gain_over_patience": gain,
                           "_protocol": "EARLY STOPPED -- not the canonical "
                                        "fixed 50-epoch cosine schedule used "
                                        "by every other DOF"}
                res["early_stop"] = stopped
                print(f"[{tag}] early stop at E{ep}: best val improved by "
                      f"{gain:.6f} over the last {a.stop_patience} epochs, "
                      f"under {a.stop_val_improvement}", flush=True)
                break

    res["final_epoch"] = {"train_nmse": nmse(model, train_eval_ld, dev),
                          "val_nmse": nmse(model, val_ld, dev),
                          "test_nmse": nmse(model, test_ld, dev)}
    model.load_state_dict(best_state)
    res.update({"best_epoch": best_ep, "best_val_nmse": best_val,
                "selected_train_nmse": nmse(model, train_eval_ld, dev),
                "selected_val_nmse": nmse(model, val_ld, dev),
                "selected_test_nmse": nmse(model, test_ld, dev),
                "runtime_s": time.time() - t0})
    res["generalization_gap_at_selection"] = (res["selected_val_nmse"]
                                              - res["selected_train_nmse"])
    ck = os.path.join(a.output_dir, f"CNN3D{a.dof}_best.pt")
    torch.save(best_state, ck)
    res["checkpoint"] = ck
    out = os.path.join(a.output_dir, f"{tag}_results.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[{tag}] best epoch {best_ep}/{a.epochs}  train "
          f"{res['selected_train_nmse']:.6f}  val {res['selected_val_nmse']:.6f}"
          f"  test {res['selected_test_nmse']:.6f}  "
          f"gap {res['generalization_gap_at_selection']:+.4f}", flush=True)
    print(f"[{tag}] final-epoch test {res['final_epoch']['test_nmse']:.6f}",
          flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
