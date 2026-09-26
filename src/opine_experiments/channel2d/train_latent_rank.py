#!/usr/bin/env python3
"""
Training script for latent-rank FINE on y-z slices.

Architecture:
    K1 -> FFT_y -> A_{k_y}(192x192) -> P(first r_{k_y} latent dims)
    -> A_{k_y}^{-1} -> IFFT_y -> K1^{-1}

Usage:
    python -u -m opine_experiments.channel2d.train_latent_rank \
        --db-root /path/to/db \
        --output-dir results/yz_latent_rank32/run001 \
        --allocation energy
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opine_experiments.channel2d.xy_planes.dataset import (
    NX_FULL, NZ,
    discover_timesteps, split_timesteps,
)
from opine_experiments.channel2d.dataset_joint import (
    ChannelFlowYZJointDataset, compute_joint_preprocessing,
)
from opine_experiments.channel2d.energy_truncation_yz_joint import (
    compute_yz_joint_energy_spectrum, compute_latent_rank_dof,
    uniform_latent_ranks,
)
from opine_experiments.channel2d.fine_yz_joint import JointFINE_YZ

N_COMP = 3
COMPONENTS = ("u", "v", "w")


def save_joint_preprocessing(path, mean_profiles, plane_stds,
                              train_steps, val_steps, test_steps):
    """Save joint y-z preprocessing to JSON for caching."""
    data = {
        "mean_profiles": {c: m.tolist() for c, m in mean_profiles.items()},
        "plane_stds": {c: s.tolist() for c, s in plane_stds.items()},
        "train_steps": [int(s) for s in train_steps],
        "val_steps": [int(s) for s in val_steps],
        "test_steps": [int(s) for s in test_steps],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_joint_preprocessing(path):
    """Load cached joint y-z preprocessing from JSON."""
    with open(path) as f:
        d = json.load(f)
    mean_profiles = {c: np.array(v, dtype=np.float64)
                     for c, v in d["mean_profiles"].items()}
    plane_stds = {c: np.array(v, dtype=np.float64)
                  for c, v in d["plane_stds"].items()}
    train_steps = np.array(d["train_steps"], dtype=np.int64)
    val_steps = np.array(d["val_steps"], dtype=np.int64)
    test_steps = np.array(d["test_steps"], dtype=np.int64)
    return mean_profiles, plane_stds, train_steps, val_steps, test_steps
RECON_INDICES = [0, 415]
RECON_EPOCHS = {1, 5}  # save recon arrays at these epochs too


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ---- FFT_y energy spectrum ----

def compute_ky_energy(train_ds, ny):
    """E(k_y) = sum over samples, components, z of |rfft_y|^2."""
    n_ky = ny // 2 + 1
    n = min(len(train_ds), 500)
    energy_per_ky = np.zeros(n_ky, dtype=np.float64)

    for i in range(n):
        x = train_ds[i].numpy()  # (3, ny, nz)
        h = np.fft.rfft(x, axis=-2)  # (3, n_ky, nz)
        for ky in range(n_ky):
            energy_per_ky[ky] += float(np.sum(np.abs(h[:, ky, :]) ** 2))

    energy_per_ky /= n
    return energy_per_ky


def save_ky_energy(energy, ny, path):
    """Save per-k_y energy spectrum as CSV."""
    n_ky = ny // 2 + 1
    cumulative = np.cumsum(energy)
    total = cumulative[-1]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ky", "energy", "fraction", "cumulative_fraction"])
        for ky in range(n_ky):
            w.writerow([ky, f"{energy[ky]:.10e}",
                        f"{energy[ky]/total:.8f}",
                        f"{cumulative[ky]/total:.8f}"])


# ---- Rank allocation ----

def energy_based_ranks(ky_energy, ny, target_dof, dim):
    """Greedy allocation: assign ranks proportional to energy / DOF cost."""
    n_ky = ny // 2 + 1
    costs = [1 if (ky == 0 or (ny % 2 == 0 and ky == ny // 2)) else 2
             for ky in range(n_ky)]

    ranks = [0] * n_ky
    dof = 0

    while dof < target_dof:
        best, best_score = -1, -1.0
        for ky in range(n_ky):
            if ranks[ky] < dim and dof + costs[ky] <= target_dof:
                score = ky_energy[ky] / (1 + ranks[ky]) / costs[ky]
                if score > best_score:
                    best_score = score
                    best = ky
        if best == -1:
            break
        ranks[best] += 1
        dof += costs[best]

    return torch.tensor(ranks, dtype=torch.long)


def save_rank_allocation(ranks_dict, ny, path):
    """Save rank allocations to JSON."""
    data = {}
    for name, ranks in ranks_dict.items():
        r = ranks.tolist()
        dof = compute_latent_rank_dof(ranks, ny)
        data[name] = {"ranks": r, "total_dof": dof, "n_ky": len(r)}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---- Gradient diagnostics ----

def grad_norms_by_group(model):
    """Return dict of gradient norms for A and K1 parameter groups."""
    norms = {}

    # A parameters
    a_norm = 0.0
    a_count = 0
    for name, p in model.named_parameters():
        if p.grad is not None and name.startswith("A."):
            a_norm += p.grad.data.norm(2).item() ** 2
            a_count += 1
    norms["A"] = a_norm ** 0.5 if a_count > 0 else 0.0

    # K1 parameters
    k_norm = 0.0
    k_count = 0
    for name, p in model.named_parameters():
        if p.grad is not None and name.startswith("k1."):
            k_norm += p.grad.data.norm(2).item() ** 2
            k_count += 1
    norms["K1"] = k_norm ** 0.5 if k_count > 0 else 0.0

    return norms


# ---- Evaluation ----

def eval_dataset(model, dataset, device, batch_size=32):
    """Evaluate on a dataset. Returns (joint_mse, {u,v,w: mse})."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mse_sum = 0.0
    comp_sum = np.zeros(3)
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            B = batch.shape[0]
            mse_sum += F.mse_loss(recon, batch, reduction="sum").item()
            for c in range(3):
                comp_sum[c] += F.mse_loss(
                    recon[:, c], batch[:, c], reduction="sum").item()
            n += B
    n_comp, ny, nz = dataset.sample_shape
    total_el = n * n_comp * ny * nz
    comp_el = n * ny * nz
    return mse_sum / total_el, {
        "u": comp_sum[0] / comp_el,
        "v": comp_sum[1] / comp_el,
        "w": comp_sum[2] / comp_el,
    }


def save_reconstructions(model, test_ds, device, indices, out_dir, tag):
    """Save original, reconstruction, and error arrays for given indices."""
    tag_dir = os.path.join(out_dir, tag)
    os.makedirs(tag_dir, exist_ok=True)
    model.eval()
    for idx in indices:
        if idx >= len(test_ds):
            continue
        x = test_ds[idx].unsqueeze(0).to(device)
        with torch.no_grad():
            r = model(x)
        x_np = x[0].cpu().numpy()
        r_np = r[0].cpu().numpy()
        e_np = x_np - r_np
        np.savez_compressed(
            os.path.join(tag_dir, f"sample_{idx}.npz"),
            original=x_np, reconstruction=r_np, error=e_np,
            index=idx)


# ---- Training loop ----

def train(model, train_ds, val_ds, test_ds, device, args, out_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    recon_dir = os.path.join(out_dir, "reconstructions")
    diag_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    # Save init checkpoint + reconstructions
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "init_model.pt"))
    save_reconstructions(model, test_ds, device, RECON_INDICES,
                         recon_dir, "init")

    # Init eval
    init_mse, init_comp = eval_dataset(model, test_ds, device)
    print(f"  [init] test={init_mse:.6e} u={init_comp['u']:.6e} "
          f"v={init_comp['v']:.6e} w={init_comp['w']:.6e}", flush=True)

    # CSV setup
    csv_path = os.path.join(out_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_joint", "val_joint", "test_joint",
                          "test_u", "test_v", "test_w", "lr",
                          "grad_norm_A", "grad_norm_K1", "wall_s"])

    # Gradient diagnostics file (per-batch, early epochs)
    grad_csv_path = os.path.join(diag_dir, "grad_norms.csv")
    grad_file = open(grad_csv_path, "w", newline="")
    grad_writer = csv.writer(grad_file)
    grad_writer.writerow(["epoch", "batch", "A", "K1"])

    best_val = float("inf")
    best_epoch = 0
    best_test_mse = float("inf")
    best_test_comp = {}
    t0 = time.time()
    instability_flag = False

    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        epoch_grad_A, epoch_grad_K1, grad_count = 0.0, 0.0, 0

        for bi, batch in enumerate(train_loader):
            batch = batch.to(device)
            recon = model(batch)
            loss = F.mse_loss(recon, batch)

            if not torch.isfinite(loss):
                print(f"  NON-FINITE LOSS at epoch {epoch+1} batch {bi}",
                      flush=True)
                instability_flag = True
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()

            # Per-layer gradient snapshot BEFORE clipping
            # at first batch of diagnostic epochs (early, middle, final)
            grad_diag_epochs = {1, max(1, args.epochs // 2), args.epochs}
            if (epoch + 1) in grad_diag_epochs and bi == 0:
                grad_snapshot = []
                for pname, p in model.named_parameters():
                    if p.grad is not None:
                        g = p.grad.data
                        grad_snapshot.append({
                            "name": pname,
                            "norm": float(g.norm(2)),
                            "mean": float(g.mean()),
                            "std": float(g.std()),
                            "max": float(g.abs().max()),
                            "median": float(g.abs().median()),
                            "zero_frac": float((g == 0).float().mean()),
                            "nan_count": int(torch.isnan(g).sum()),
                            "inf_count": int(torch.isinf(g).sum()),
                            "numel": int(g.numel()),
                        })
                snap_path = os.path.join(
                    diag_dir, f"grad_snapshot_epoch{epoch+1}.json")
                with open(snap_path, "w") as f:
                    json.dump(grad_snapshot, f, indent=2)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Accumulate grad norms for epoch-level reporting
            gnorms = grad_norms_by_group(model)
            epoch_grad_A += gnorms["A"]
            epoch_grad_K1 += gnorms["K1"]
            grad_count += 1

            # Per-batch diagnostics: first 3 batches of first 5 epochs
            if epoch < 5 and bi < 3:
                grad_writer.writerow([epoch + 1, bi + 1,
                                      f"{gnorms['A']:.6e}",
                                      f"{gnorms['K1']:.6e}"])
                grad_file.flush()
                if epoch == 0 and bi == 0:
                    print(f"  [grad check] A={gnorms['A']:.4e} "
                          f"K1={gnorms['K1']:.4e}", flush=True)
                    if gnorms["A"] == 0 or gnorms["K1"] == 0:
                        print("  WARNING: zero gradient detected!", flush=True)
                    if not (np.isfinite(gnorms["A"]) and
                            np.isfinite(gnorms["K1"])):
                        print("  WARNING: non-finite gradient!", flush=True)
                        instability_flag = True

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = total_loss / max(n_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        avg_grad_A = epoch_grad_A / max(grad_count, 1)
        avg_grad_K1 = epoch_grad_K1 / max(grad_count, 1)
        epoch_wall = time.time() - epoch_t0

        # Validation + test every epoch (val_every=1 for complete record)
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                vb = vb.to(device)
                vr = model(vb)
                val_loss += F.mse_loss(vr, vb).item()
                val_n += 1
        val_loss /= max(val_n, 1)

        test_mse, test_comp = eval_dataset(model, test_ds, device)

        csv_writer.writerow([
            epoch + 1, f"{avg_train:.8e}", f"{val_loss:.8e}",
            f"{test_mse:.8e}", f"{test_comp['u']:.8e}",
            f"{test_comp['v']:.8e}", f"{test_comp['w']:.8e}",
            f"{current_lr:.8e}", f"{avg_grad_A:.6e}",
            f"{avg_grad_K1:.6e}", f"{epoch_wall:.1f}"])
        csv_file.flush()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_test_mse = test_mse
            best_test_comp = dict(test_comp)
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, "best_model.pt"))
            save_reconstructions(model, test_ds, device, RECON_INDICES,
                                 recon_dir, "best")

        # Save reconstructions at key epochs
        if (epoch + 1) in RECON_EPOCHS:
            save_reconstructions(model, test_ds, device, RECON_INDICES,
                                 recon_dir, f"epoch_{epoch+1}")

        # Intermediate reconstruction at midpoint
        if (epoch + 1) == args.epochs // 2:
            save_reconstructions(model, test_ds, device, RECON_INDICES,
                                 recon_dir, f"epoch_{epoch+1}")

        print(f"  ep={epoch+1}/{args.epochs} train={avg_train:.6e} "
              f"val={val_loss:.6e} test={test_mse:.6e} "
              f"best_val={best_val:.6e}@{best_epoch} "
              f"wall={epoch_wall:.0f}s", flush=True)

    elapsed = time.time() - t0

    # Final checkpoint + reconstructions
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final_model.pt"))
    save_reconstructions(model, test_ds, device, RECON_INDICES,
                         recon_dir, "final")

    # Final eval
    final_mse, final_comp = eval_dataset(model, test_ds, device)

    csv_file.close()
    grad_file.close()

    return {
        "init_mse": init_mse,
        "init_comp": init_comp,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "best_test_mse": best_test_mse,
        "best_test_comp": best_test_comp,
        "final_mse": final_mse,
        "final_comp": final_comp,
        "elapsed_s": elapsed,
        "instability": instability_flag,
    }


# ---- Post-training plots ----

DZ = 1.0 / NZ
RE_TAU = 180.0
COMP_LABELS = ("u", "v", "w")


def _make_dct_matrix(N):
    """Type-II DCT basis, orthonormal."""
    n = np.arange(N)
    k = np.arange(N)
    D = np.cos(math.pi * (2.0 * n[np.newaxis, :] + 1.0)
               * k[:, np.newaxis] / (2.0 * N))
    D[0, :] *= np.sqrt(1.0 / N)
    D[1:, :] *= np.sqrt(2.0 / N)
    return D


def dct32_reconstruct(x, nz):
    """DCT truncation to 32 coefficients per (component, y-location)."""
    D = _make_dct_matrix(nz)  # (nz, nz)
    n_keep = min(32, nz)
    D_trunc = D[:n_keep, :]   # (n_keep, nz)
    coeffs = D_trunc @ x.reshape(-1, nz).T  # (n_keep, C*ny)
    recon = D_trunc.T @ coeffs               # (nz, C*ny)
    return recon.T.reshape(x.shape)


def generate_plots(out_dir, test_ds, device, ranks, ny, nz):
    """Generate all post-training plots and metrics table."""
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "metrics.csv")
    if not os.path.isfile(csv_path):
        print("  No metrics.csv found, skipping plots.", flush=True)
        return

    # Read metrics
    epochs, train_j, val_j, test_j = [], [], [], []
    test_u, test_v, test_w = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_j.append(float(row["train_joint"]))
            val_j.append(float(row["val_joint"]))
            test_j.append(float(row["test_joint"]))
            test_u.append(float(row["test_u"]))
            test_v.append(float(row["test_v"]))
            test_w.append(float(row["test_w"]))

    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, train_j, label="Train")
    ax.semilogy(epochs, val_j, label="Validation")
    ax.semilogy(epochs, test_j, label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Joint MSE")
    ax.set_title("Latent-rank FINE32: Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)

    # 2. u/v/w component curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, test_u, label="u (streamwise)")
    ax.semilogy(epochs, test_v, label="v (spanwise)")
    ax.semilogy(epochs, test_w, label="w (wall-normal)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Component MSE")
    ax.set_title("Latent-rank FINE32: Per-Component MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "component_curves.png"), dpi=150)
    plt.close(fig)

    # 3. Sample reconstruction figures for best checkpoint
    recon_dir = os.path.join(out_dir, "reconstructions", "best")
    z_plus = (np.arange(nz) + 0.5) * DZ * RE_TAU
    for idx in RECON_INDICES:
        npz_path = os.path.join(recon_dir, f"sample_{idx}.npz")
        if not os.path.isfile(npz_path):
            continue
        data = np.load(npz_path)
        orig, recon_arr = data["original"], data["reconstruction"]

        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        for c in range(3):
            axes[c, 0].pcolormesh(orig[c].T, cmap="RdBu_r")
            axes[c, 0].set_ylabel(f"{COMP_LABELS[c]}")
            if c == 0:
                axes[c, 0].set_title("Original")

            axes[c, 1].pcolormesh(recon_arr[c].T, cmap="RdBu_r")
            if c == 0:
                axes[c, 1].set_title("FINE32 Recon")

            err = orig[c] - recon_arr[c]
            axes[c, 2].pcolormesh(err.T, cmap="RdBu_r")
            if c == 0:
                axes[c, 2].set_title("Error")

        fig.suptitle(f"Sample #{idx} — Latent-rank FINE32 (best)")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"recon_sample_{idx}.png"),
                    dpi=150)
        plt.close(fig)

    # 4. DCT32 comparison (compute DCT32 baseline on same samples)
    for idx in RECON_INDICES:
        npz_path = os.path.join(recon_dir, f"sample_{idx}.npz")
        if not os.path.isfile(npz_path):
            continue
        data = np.load(npz_path)
        orig = data["original"]
        fine_recon = data["reconstruction"]
        dct_recon = dct32_reconstruct(orig, nz)

        fine_mse = float(np.mean((orig - fine_recon) ** 2))
        dct_mse = float(np.mean((orig - dct_recon) ** 2))

        fig, axes = plt.subplots(3, 4, figsize=(18, 10))
        for c in range(3):
            vmax = float(np.max(np.abs(orig[c])))
            axes[c, 0].pcolormesh(orig[c].T, cmap="RdBu_r",
                                  vmin=-vmax, vmax=vmax)
            axes[c, 0].set_ylabel(f"{COMP_LABELS[c]}")
            if c == 0:
                axes[c, 0].set_title("Original")

            axes[c, 1].pcolormesh(fine_recon[c].T, cmap="RdBu_r",
                                  vmin=-vmax, vmax=vmax)
            if c == 0:
                axes[c, 1].set_title(f"FINE32\nMSE={fine_mse:.4e}")

            axes[c, 2].pcolormesh(dct_recon[c].T, cmap="RdBu_r",
                                  vmin=-vmax, vmax=vmax)
            if c == 0:
                axes[c, 2].set_title(f"DCT32\nMSE={dct_mse:.4e}")

            err_fine = orig[c] - fine_recon[c]
            err_dct = orig[c] - dct_recon[c]
            emax = max(float(np.max(np.abs(err_fine))),
                       float(np.max(np.abs(err_dct))))
            axes[c, 3].pcolormesh(err_fine.T, cmap="RdBu_r",
                                  vmin=-emax, vmax=emax)
            if c == 0:
                axes[c, 3].set_title("FINE32 Error")

        fig.suptitle(f"Sample #{idx} — FINE32 vs DCT32")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"dct_comparison_{idx}.png"),
                    dpi=150)
        plt.close(fig)

    # 5. Compact metrics table
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        table_lines = [
            "Latent-rank FINE32 — Compact Metrics",
            "=" * 45,
            f"Allocation:    {summary.get('allocation', 'N/A')}",
            f"Ranks:         {summary.get('ranks', 'N/A')}",
            f"Total DOF:     {summary.get('total_dof', 'N/A')}",
            "",
            f"{'Stage':<12} {'Joint MSE':>12} {'u':>12} {'v':>12} {'w':>12}",
            "-" * 65,
            f"{'Init':<12} {summary['init_mse']:>12.6e} "
            f"{summary['init_u']:>12.6e} {summary['init_v']:>12.6e} "
            f"{summary['init_w']:>12.6e}",
            f"{'Best(ep{})'.format(summary['best_epoch']):<12} "
            f"{summary['best_mse']:>12.6e} "
            f"{summary.get('best_u', 0):>12.6e} "
            f"{summary.get('best_v', 0):>12.6e} "
            f"{summary.get('best_w', 0):>12.6e}",
            f"{'Final':<12} {summary['final_mse']:>12.6e} "
            f"{summary['final_u']:>12.6e} {summary['final_v']:>12.6e} "
            f"{summary['final_w']:>12.6e}",
            "",
            f"Training time: {summary['training_time_min']:.1f} min",
            f"Instability:   {summary.get('instability', False)}",
        ]
        with open(os.path.join(plot_dir, "metrics_table.txt"), "w") as f:
            f.write("\n".join(table_lines) + "\n")

    print(f"  Plots saved to {plot_dir}", flush=True)


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="Latent-rank FINE training for y-z slices")
    parser.add_argument("--db-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allocation", default="energy",
                        choices=["uniform", "energy"],
                        help="Rank allocation strategy")
    parser.add_argument("--target-dof", type=int, default=32)
    parser.add_argument("--ix", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--preprocessing-file", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--no-k1", action="store_true",
                        help="Disable K1 spline (C-only: FFT->A->P->A^-1->IFFT)")
    args = parser.parse_args()

    assert 0 <= args.ix < NX_FULL
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu"))

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"=== Latent-rank FINE y-z: {run_id} ===", flush=True)
    print(f"Device: {device}", flush=True)

    # ---- Data ----
    pp_cache = args.preprocessing_file
    pp_loaded = False
    if pp_cache and os.path.isfile(pp_cache):
        try:
            print(f"Loading cached preprocessing from {pp_cache}", flush=True)
            mean_profiles, plane_stds, train_steps, val_steps, test_steps = \
                load_joint_preprocessing(pp_cache)
            pp_loaded = True
        except (KeyError, ValueError) as e:
            print(f"  Cache format mismatch ({e}), recomputing...",
                  flush=True)
    if not pp_loaded:
        all_steps = discover_timesteps(args.db_root, stride=args.stride)
        train_steps, val_steps, test_steps = split_timesteps(
            all_steps, seed=args.seed)
        print("Computing joint preprocessing (may take ~15 min)...",
              flush=True)
        mean_profiles, plane_stds = compute_joint_preprocessing(
            args.db_root, train_steps)
        # Cache for future runs
        if pp_cache:
            save_joint_preprocessing(
                pp_cache, mean_profiles, plane_stds,
                train_steps, val_steps, test_steps)
            print(f"Saved preprocessing cache to {pp_cache}", flush=True)

    ny, nz = args.ny, NZ
    n_ky = ny // 2 + 1
    dim = N_COMP * nz  # 192

    mk_ds = lambda steps, pre=False: ChannelFlowYZJointDataset(
        args.db_root, steps, mean_profiles, args.ix, plane_stds,
        ny, preload=pre)
    train_ds = mk_ds(train_steps, args.preload)
    val_ds = mk_ds(val_steps)
    test_ds = mk_ds(test_steps)

    print(f"Data: {len(train_ds)} train, {len(val_ds)} val, "
          f"{len(test_ds)} test, shape={train_ds.sample_shape}", flush=True)

    # ---- FFT_y energy spectrum ----
    print("Computing per-k_y FFT_y energy spectrum...", flush=True)
    ky_energy = compute_ky_energy(train_ds, ny)
    save_ky_energy(ky_energy, ny, os.path.join(out_dir, "ky_energy.csv"))
    print(f"  E(k_y=0)={ky_energy[0]:.4e}, E(k_y=1)={ky_energy[1]:.4e}, "
          f"E(k_y=16)={ky_energy[-1]:.4e}", flush=True)

    # ---- Rank allocation ----
    ranks_energy = energy_based_ranks(ky_energy, ny, args.target_dof, dim)
    alloc_dict = {"energy": ranks_energy}

    # Uniform allocation only valid when target_dof matches r=1 base (32 for ny=32)
    try:
        ranks_uniform = uniform_latent_ranks(ny, args.target_dof)
        alloc_dict["uniform"] = ranks_uniform
    except ValueError:
        ranks_uniform = None

    save_rank_allocation(
        alloc_dict, ny, os.path.join(out_dir, "rank_allocation.json"))

    if args.allocation == "uniform":
        if ranks_uniform is None:
            print(f"ERROR: uniform allocation not possible for DOF={args.target_dof}",
                  flush=True)
            sys.exit(1)
        ranks = ranks_uniform
    else:
        ranks = ranks_energy

    actual_dof = compute_latent_rank_dof(ranks, ny)
    print(f"Allocation '{args.allocation}': ranks={ranks.tolist()}, "
          f"DOF={actual_dof}", flush=True)

    # ---- Model ----
    use_k1 = not args.no_k1
    model = JointFINE_YZ(ny, nz, variant="latent_rank", ranks=ranks,
                         use_k1=use_k1)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    variant_label = "c_only" if args.no_k1 else "latent_rank"
    print(f"Model: {variant_label}, use_k1={use_k1}, params={n_params:,}, "
          f"DOF={actual_dof}", flush=True)

    # ---- Config ----
    config = {
        "experiment": "channel_retau180_yz_latent_rank",
        "run_id": run_id,
        "variant": variant_label,
        "use_k1": use_k1,
        "allocation": args.allocation,
        "ranks": ranks.tolist(),
        "total_dof": actual_dof,
        "target_dof": args.target_dof,
        "n_params": n_params,
        "ix": args.ix,
        "components": "u,v,w",
        "input_shape": [N_COMP, ny, nz],
        "dim_per_ky": dim,
        "n_ky": n_ky,
        "periodic_axis": "y (spanwise)",
        "bounded_axis": "z (wall-normal)",
        "normalization": "per_component_per_z_rms",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "grad_clip": 1.0,
        "stride": args.stride,
        "ny": ny, "nz": nz,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "recon_indices": RECON_INDICES,
        "device": str(device),
        "git_hash": get_git_hash(),
        "timestamp": run_id,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- Train ----
    results = train(model, train_ds, val_ds, test_ds, device, args, out_dir)

    # ---- Summary ----
    summary = {
        "run_id": run_id,
        "allocation": args.allocation,
        "ranks": ranks.tolist(),
        "total_dof": actual_dof,
        "init_mse": results["init_mse"],
        "init_u": results["init_comp"]["u"],
        "init_v": results["init_comp"]["v"],
        "init_w": results["init_comp"]["w"],
        "best_mse": results["best_test_mse"],
        "best_epoch": results["best_epoch"],
        "best_u": results["best_test_comp"].get("u"),
        "best_v": results["best_test_comp"].get("v"),
        "best_w": results["best_test_comp"].get("w"),
        "final_mse": results["final_mse"],
        "final_u": results["final_comp"]["u"],
        "final_v": results["final_comp"]["v"],
        "final_w": results["final_comp"]["w"],
        "training_time_s": results["elapsed_s"],
        "training_time_min": results["elapsed_s"] / 60,
        "instability": results["instability"],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Update config with results
    config.update(summary)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY — {args.allocation} allocation", flush=True)
    print(f"  init  MSE = {results['init_mse']:.6e}", flush=True)
    print(f"  best  MSE = {results['best_test_mse']:.6e} "
          f"(epoch {results['best_epoch']})", flush=True)
    print(f"  final MSE = {results['final_mse']:.6e}", flush=True)
    print(f"  u/v/w = {results['best_test_comp'].get('u', 0):.6e} / "
          f"{results['best_test_comp'].get('v', 0):.6e} / "
          f"{results['best_test_comp'].get('w', 0):.6e}", flush=True)
    print(f"  r_k_y = {ranks.tolist()}", flush=True)
    print(f"  time  = {results['elapsed_s']/60:.1f} min", flush=True)
    print(f"  instability = {results['instability']}", flush=True)
    print(f"{'='*60}", flush=True)

    # ---- Post-training plots ----
    print("Generating post-training plots...", flush=True)
    generate_plots(out_dir, test_ds, device, ranks, ny, nz)
    print(f"Results saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
