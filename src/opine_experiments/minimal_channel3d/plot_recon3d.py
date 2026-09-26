#!/usr/bin/env python3
"""Qualitative 3-D reconstruction figures for the minimal channel.

Three orthogonal slices through ONE representative test snapshot, for
ground truth / POD / CNN / OPINE, plus a matching error figure.

Fair-evaluation protocol
  POD    exact snapshot POD of the full 16000-snapshot training split
  CNN    validation-selected best checkpoint
  OPINE  16k-trained f_theta with the projector carried in its checkpoint,
         the relaxed subspace iteration of the training loop.  That is the
         protocol behind every curve in the paper, so figure and table quote
         the same OPINE at every D.

Plotted quantity is the streamwise FLUCTUATION u' in physical units.  Because
preprocessing is  x_norm = (u - mean_c(z)) / rms_c(z)  with both profiles from
the training split, the fluctuation in u_tau units is simply

    u'(z,y,x) = x_norm(z,y,x) * rms_u(z)

so de-normalising and removing the mean profile are the same operation.  The
mean profile never enters the figure, which is the point -- it would otherwise
dominate everything (<u> runs 1.4 -> 19.7 across the channel).
"""
import argparse, json, os, sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from dataset3d import make_datasets, N_FEAT, NC, NZ, NY, NX
from coupling_flow_3d import OPINE3D, Identity3D
from cnn3d import JointCNN3D

LX, LY, LZ = np.pi, np.pi / 2, 1.0
RE_TAU = 180.0


# ------------------------------------------------------------------ linear
@torch.no_grad()
def cache_latents(model, loader, device):
    model.eval()
    out, i = None, 0
    for xb in loader:
        z = model.encode(xb.to(device, non_blocking=True)).float()
        if out is None:
            out = torch.empty(len(loader.dataset), z.shape[1],
                              dtype=torch.float32, device=device)
        out[i:i + z.shape[0]] = z
        i += z.shape[0]
    return out


@torch.no_grad()
def exact_snapshot_pod(Zc, dof, chunk=16384):
    """Exact rank-dof POD of centred (M, N) Zc.  Orthonormalised by QR in
    float64 -- column normalisation alone is not enough, see
    refit_subspace3d.py."""
    M, N = Zc.shape
    G = torch.zeros(M, M, dtype=torch.float64, device=Zc.device)
    for k in range(0, N, chunk):
        b = Zc[:, k:k + chunk].double()
        G += b @ b.T
    evals, evecs = torch.linalg.eigh(G)
    V = evecs.flip(1)[:, :dof]
    U = torch.empty(N, dof, dtype=torch.float64, device=Zc.device)
    for k in range(0, N, chunk):
        U[k:k + chunk] = Zc[:, k:k + chunk].double().T @ V
    Q, _ = torch.linalg.qr(U)
    err = float((Q.T @ Q - torch.eye(dof, dtype=Q.dtype,
                                     device=Q.device)).abs().max())
    return Q.float(), err


@torch.no_grad()
def recon_projected(model, X, device, Q, mu, batch=32):
    """Full reconstruction through f, P_Q, f^-1 for a stack of snapshots."""
    model.eval()
    qQ, qmu = model.Q, model.z_mean
    model.Q, model.z_mean = Q, mu
    out = np.empty_like(X)
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().to(device)
        out[i:i + batch] = model(xb).cpu().numpy()
    model.Q, model.z_mean = qQ, qmu
    return out


@torch.no_grad()
def recon_cnn(model, X, device, batch=32):
    model.eval()
    out = np.empty_like(X)
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().to(device)
        out[i:i + batch] = model(xb)[0].cpu().numpy()
    return out


def nmse_global(r, x):
    return float(np.sum((r - x) ** 2) / np.sum(x ** 2))


def nmse_per_sample(r, x):
    ax = tuple(range(1, x.ndim))
    return np.sum((r - x) ** 2, axis=ax) / np.sum(x ** 2, axis=ax)


# ------------------------------------------------------------------ plotting
def slices(field3d, iz, iy, ix):
    """field3d is (z, y, x).  Returns the three orthogonal slices, each
    oriented so the first named axis is horizontal."""
    xy = field3d[iz, :, :]          # (y, x) -> rows y, cols x
    xz = field3d[:, iy, :]          # (z, x) -> rows z, cols x
    yz = field3d[:, :, ix]          # (z, y) -> rows z, cols y
    return xy, xz, yz


PANEL = [("$x$--$y$", (0, LX, 0, LY), "$x/h$", "$y/h$"),
         ("$x$--$z$", (0, LX, 0, LZ), "$x/h$", "$z/h$"),
         ("$y$--$z$", (0, LY, 0, LZ), "$y/h$", "$z/h$")]
WIDTH_RATIOS = [LX / LY, LX / LZ, LY / LZ]


def draw(rows, iz, iy, ix, vmax, title, path, cmap="RdBu_r", cbar_label=None):
    nr = len(rows)
    fig = plt.figure(figsize=(11.2, 1.05 + 2.15 * nr))
    gs = GridSpec(nr, 4, figure=fig,
                  width_ratios=WIDTH_RATIOS + [0.10],
                  wspace=0.22, hspace=0.30,
                  left=0.075, right=0.93, top=1 - 0.62 / (1.05 + 2.15 * nr),
                  bottom=0.085)
    im = None
    for r, (label, f3, note) in enumerate(rows):
        sl = slices(f3, iz, iy, ix)
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(sl[c], origin="lower", cmap=cmap,
                           vmin=-vmax, vmax=vmax,
                           extent=PANEL[c][1], aspect="equal",
                           interpolation="nearest")
            if r == 0:
                ax.set_title(PANEL[c][0], fontsize=11, pad=6)
            if r == nr - 1:
                ax.set_xlabel(PANEL[c][2], fontsize=9)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(f"{label}\n{PANEL[c][3]}", fontsize=9.5)
            else:
                ax.set_ylabel(PANEL[c][3], fontsize=9)
            ax.tick_params(labelsize=7.5)
            if c == 2 and note:
                ax.text(1.035, 0.5, note, transform=ax.transAxes,
                        rotation=90, va="center", ha="left", fontsize=8.5)
    cax = fig.add_subplot(gs[:, 3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label or r"$u'\,/\,u_\tau$", fontsize=10)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=11.5, y=0.985)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-cache", required=True)
    ap.add_argument("--base", required=True, help="data/channel3d")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dofs", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--pair-divisor", type=int, default=1, choices=(1, 2),
                    help="2 reads --dofs as the RETAINED dimension the "
                         "summary figure plots: POD and OPINE then take "
                         "stored rank 2D, because their modes come in pairs, "
                         "while the CNN latent stays D.  1 (default) is the "
                         "old behaviour, every method at stored rank D")
    ap.add_argument("--gain-percentile", type=float, default=60.0)
    ap.add_argument("--component", type=int, default=0, help="0=u,1=v,2=w")
    a = ap.parse_args()

    dev = torch.device("cuda")
    os.makedirs(a.outdir, exist_ok=True)
    tr, va, te, mean, std = make_datasets(a.stats_cache)
    mean = np.asarray(mean); std = np.maximum(np.asarray(std), 1e-8)
    mk = lambda ds, bs: torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    train_ld = mk(tr, 32)
    Xte = np.stack([te[i].numpy() for i in range(len(te))]).astype(np.float32)
    print(f"test {Xte.shape}", flush=True)

    # wall-normal slice: the level with the strongest streamwise fluctuation,
    # which is where the near-wall structure is actually visible.  Kept off the
    # first few nodes so it is an interior plane, not the wall.
    cmp_ = a.component
    rms = std[cmp_]
    iz = int(np.argmax(rms[3:])) + 3
    iy, ix = NY // 2, NX // 2

    # exact full-training POD basis, computed once at the largest D
    pair = a.pair_divisor
    ident = OPINE3D(max(a.dofs) * pair).to(dev)
    ident.f = Identity3D().to(dev)
    Zi = cache_latents(ident, train_ld, dev)
    mu_pod = Zi.mean(0); Zi -= mu_pod.unsqueeze(0)
    U_pod, err_pod = exact_snapshot_pod(Zi, max(a.dofs) * pair)
    del Zi; torch.cuda.empty_cache()
    print(f"exact POD basis ready, |Q^TQ-I| {err_pod:.1e}", flush=True)

    report = {"slice": {"iz": iz, "iy": iy, "ix": ix,
                        "z_over_h": float((iz + 0.5) / NZ),
                        "z_plus": float((iz + 0.5) / NZ * RE_TAU),
                        "rule": "iz = argmax of the training rms profile of "
                                "the plotted component, restricted to iz>=3 "
                                "so it is an interior plane"},
              "component": "uvw"[cmp_], "dofs": {}}

    for D in a.dofs:
        R = D * pair          # stored rank for POD and OPINE
        print(f"\n===== D = {D} "
              f"(POD/OPINE stored rank {R}, CNN latent {D}) =====", flush=True)
        Q_pod = U_pod[:, :R].contiguous()
        R_pod = recon_projected(ident, Xte, dev, Q_pod, mu_pod)

        ck = os.path.join(a.base, f"cnn_D{D}", f"CNN3D{D}_best.pt")
        cnn = JointCNN3D(input_shape=(NZ, NY, NX), latent_dim=D).to(dev)
        cnn.load_state_dict(torch.load(ck, map_location=dev))
        R_cnn = recon_cnn(cnn, Xte, dev)
        del cnn; torch.cuda.empty_cache()

        ock = os.path.join(a.base, f"opine_full_D{R}",
                           f"opine3d_D{R}_best.pt")
        op = OPINE3D(R, n_blocks=8, hidden=64, init_scale=0.05,
                     s_max=2.0).to(dev)
        op.load_state_dict(torch.load(ock, map_location=dev)["best_state"])
        op.eval()
        # The projector is the one carried in the checkpoint -- the relaxed
        # subspace iteration of the training loop.  This is the protocol
        # behind every curve in the paper, so the figures and the summary
        # table now quote the same number at every D.
        #
        # An earlier version also refitted Q by an exact POD of the 16k
        # training latents and kept whichever of the two scored better on the
        # test split.  That selected on the test set, and it switched
        # protocol between D values (it won at D=256 and D=1024 and lost at
        # D=128 and D=512), so the figures quoted a different OPINE from the
        # summary curve at some D and the same at others.  Removed.
        R_op = recon_projected(op, Xte, dev, op.Q.clone(), op.z_mean.clone())
        n_op = nmse_global(R_op, Xte)
        op_label = "OPINE  (16k train + 16k subspace iteration)"
        op_variant = "C"
        print(f"  OPINE C {n_op:.6f}", flush=True)
        del op; torch.cuda.empty_cache()

        n_pod, n_cnn = nmse_global(R_pod, Xte), nmse_global(R_cnn, Xte)
        print(f"  POD {n_pod:.6f}  CNN {n_cnn:.6f}  OPINE {n_op:.6f}",
              flush=True)

        # representative snapshot: slightly above-median OPINE gain over POD
        g = ((nmse_per_sample(R_pod, Xte) - nmse_per_sample(R_op, Xte))
             / nmse_per_sample(R_pod, Xte))
        target = np.percentile(g, a.gain_percentile)
        si = int(np.argmin(np.abs(g - target)))
        print(f"  snapshot {si}: gain {g[si]*100:.2f}% "
              f"(p{a.gain_percentile:.0f} = {target*100:.2f}%, "
              f"median {np.median(g)*100:.2f}%, "
              f"range {g.min()*100:.1f}..{g.max()*100:.1f}%)", flush=True)

        # de-normalise to u' in u_tau units: multiply by the rms profile
        s = std[cmp_][:, None, None]
        f_true = Xte[si, cmp_] * s
        f_pod = R_pod[si, cmp_] * s
        f_cnn = R_cnn[si, cmp_] * s
        f_op = R_op[si, cmp_] * s
        vmax = float(np.abs(f_true).max())

        # Export the selected snapshot so the figures can be redesigned
        # without a GPU and without recomputing any reconstruction.  Fields
        # are the plotted component in u_tau units; per-snapshot NMSE is
        # computed on the full 3-component normalized snapshot, which is the
        # same definition as the split-level metric restricted to one sample.
        ax3 = (0, 1, 2)
        den_i = np.sum(Xte[si] ** 2)
        npz = os.path.join(a.outdir, f"recon3d_D{D}_snapshot{si}.npz")
        # all three components as well, de-normalised, so the Q-criterion
        # (which needs the full velocity gradient tensor) can be formed later
        # without recomputing any reconstruction
        s3 = std[:, :, None, None]
        np.savez_compressed(
            npz,
            truth=f_true.astype(np.float32), pod=f_pod.astype(np.float32),
            cnn=f_cnn.astype(np.float32), opine=f_op.astype(np.float32),
            truth_uvw=(Xte[si] * s3).astype(np.float32),
            pod_uvw=(R_pod[si] * s3).astype(np.float32),
            cnn_uvw=(R_cnn[si] * s3).astype(np.float32),
            opine_uvw=(R_op[si] * s3).astype(np.float32),
            rms_profile_uvw=std.astype(np.float32),
            mean_profile_uvw=mean.astype(np.float32),
            rms_profile=std[cmp_].astype(np.float32),
            mean_profile=mean[cmp_].astype(np.float32),
            snapshot_index=si, dof=D, component=cmp_,
            stored_rank_pod=R, stored_rank_opine=R, stored_rank_cnn=D,
            pair_divisor=pair,
            iz=iz, iy=iy, ix=ix,
            z_plus=float((iz + 0.5) / NZ * RE_TAU),
            Lx=LX, Ly=LY, Lz=LZ, re_tau=RE_TAU,
            nmse_split_pod=n_pod, nmse_split_cnn=n_cnn,
            nmse_split_opine=n_op,
            nmse_snapshot_pod=float(np.sum((R_pod[si] - Xte[si]) ** 2) / den_i),
            nmse_snapshot_cnn=float(np.sum((R_cnn[si] - Xte[si]) ** 2) / den_i),
            nmse_snapshot_opine=float(np.sum((R_op[si] - Xte[si]) ** 2) / den_i),
            opine_variant=op_variant)
        print(f"  exported {npz}", flush=True)

        comp = "uvw"[cmp_]
        base_title = (f"3-D minimal channel  Re$_\\tau$=180,  $D={D}$,  "
                      f"${comp}'$  |  test snapshot {si}  "
                      f"(slices: $z^+$={report['slice']['z_plus']:.0f}, "
                      f"$y$-index {iy}, $x$-index {ix})")
        draw([("Ground truth", f_true, ""),
              ("POD", f_pod, f"NMSE {n_pod:.4f}"),
              ("CNN", f_cnn, f"NMSE {n_cnn:.4f}"),
              (op_label.split("  ")[0], f_op, f"NMSE {n_op:.4f}")],
             iz, iy, ix, vmax, base_title,
             os.path.join(a.outdir,
                          f"fig_recon3d_D{D}_{comp}prime_main.png"))

        e_pod, e_cnn, e_op = f_pod - f_true, f_cnn - f_true, f_op - f_true
        evmax = float(max(np.abs(e_pod).max(), np.abs(e_cnn).max(),
                          np.abs(e_op).max()))
        draw([("POD", e_pod, f"NMSE {n_pod:.4f}"),
              ("CNN", e_cnn, f"NMSE {n_cnn:.4f}"),
              (op_label.split("  ")[0], e_op, f"NMSE {n_op:.4f}")],
             iz, iy, ix, evmax,
             base_title.replace("$  |", "$  error (recon $-$ truth)  |"),
             os.path.join(a.outdir,
                          f"fig_recon3d_D{D}_{comp}prime_error.png"),
             cmap="RdBu_r",
             cbar_label=rf"$\hat{{{comp}}}' - {comp}'\ /\ u_\tau$")

        report["dofs"][str(D)] = {
            "snapshot_index": si,
            "snapshot_gain_percent": float(g[si] * 100),
            "gain_percentile_used": a.gain_percentile,
            "gain_median_percent": float(np.median(g) * 100),
            "gain_range_percent": [float(g.min() * 100), float(g.max() * 100)],
            "nmse": {"POD": n_pod, "CNN": n_cnn, "OPINE": n_op},
            "stored_rank": {"POD": R, "OPINE": R, "CNN": D},
            "pair_divisor": pair,
            "opine_variant": op_variant,
            "_opine_variant_is": ("the projector carried in the checkpoint, "
                                  "the same one the summary curve uses at "
                                  "every D"),
            "field_vmax_utau": vmax, "error_vmax_utau": evmax}

    json.dump(report, open(os.path.join(a.outdir, "recon3d_provenance.json"),
                           "w"), indent=1, default=float)
    print("\n" + json.dumps(report, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
