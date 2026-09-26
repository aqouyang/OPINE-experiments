#!/usr/bin/env python3
"""Fill in the missing TRAIN NMSE for the 3-D minimal-channel baselines.

train_opine3d.py records `selected_train_nmse: None` for both the OPINE and
the POD (identity) path, and the CNN runs record it but under their own
evaluation loop.  The summary figure needs train and test on one axis for
every method, so this recomputes all of them from the stored checkpoints,
under one metric and one split.  No model is retrained.

  POD    exact snapshot POD of the full 16000-snapshot training split
         (16000x16000 float64 Gram, full eigh, QR-orthonormalised), sliced to
         each D from one decomposition
  OPINE  opine_full_D* -- f_theta trained with the 16k subspace, evaluated
         with its own stored mu_z and Q
  CNN    validation-selected best checkpoint

Metric is the canonical sum((xhat-x)^2)/sum(x^2), accumulated over the whole
split at once, in normalized units -- identical to every other 3-D number.
"""
import argparse, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from dataset3d import make_datasets, N_FEAT, NZ, NY, NX
from coupling_flow_3d import OPINE3D, Identity3D
from cnn3d import JointCNN3D


@torch.no_grad()
def cache_latents(model, loader, dev):
    model.eval()
    out, i = None, 0
    for xb in loader:
        z = model.encode(xb.to(dev, non_blocking=True)).float()
        if out is None:
            out = torch.empty(len(loader.dataset), z.shape[1],
                              dtype=torch.float32, device=dev)
        out[i:i + z.shape[0]] = z
        i += z.shape[0]
    return out


@torch.no_grad()
def exact_snapshot_pod(Zc, dof, chunk=16384):
    M, N = Zc.shape
    G = torch.zeros(M, M, dtype=torch.float64, device=Zc.device)
    for k in range(0, N, chunk):
        b = Zc[:, k:k + chunk].double()
        G += b @ b.T
    _, evecs = torch.linalg.eigh(G)
    V = evecs.flip(1)[:, :dof]
    U = torch.empty(N, dof, dtype=torch.float64, device=Zc.device)
    for k in range(0, N, chunk):
        U[k:k + chunk] = Zc[:, k:k + chunk].double().T @ V
    Q, _ = torch.linalg.qr(U)
    err = float((Q.T @ Q - torch.eye(dof, dtype=Q.dtype,
                                     device=Q.device)).abs().max())
    return Q.float(), err


@torch.no_grad()
def nmse_proj(model, loader, dev, Q, mu):
    model.eval()
    qQ, qmu = model.Q, model.z_mean
    model.Q, model.z_mean = Q, mu
    num = den = 0.0
    for xb in loader:
        x = xb.to(dev, non_blocking=True)
        r = model(x)
        num += float((r - x).double().pow(2).sum())
        den += float(x.double().pow(2).sum())
    model.Q, model.z_mean = qQ, qmu
    return num / den


@torch.no_grad()
def nmse_cnn(model, loader, dev):
    model.eval()
    num = den = 0.0
    for xb in loader:
        x = xb.to(dev, non_blocking=True)
        r = model(x)[0]
        num += float((r - x).double().pow(2).sum())
        den += float(x.double().pow(2).sum())
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-cache", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dofs", type=int, nargs="+",
                    default=[32, 64, 128, 256, 512, 1024])
    ap.add_argument("--opine-dir", default="opine_full_D{D}")
    a = ap.parse_args()
    dev = torch.device("cuda")
    tr, va, te, _, _ = make_datasets(a.stats_cache)
    mk = lambda ds, bs: torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    tr_ld, va_ld, te_ld = mk(tr, 32), mk(va, 32), mk(te, 32)
    print(f"train {len(tr)} val {len(va)} test {len(te)}  N={N_FEAT}",
          flush=True)
    rep = {"_metric": "sum((xhat-x)^2)/sum(x^2), whole split, normalized units",
           "_splits": {"train": len(tr), "val": len(va), "test": len(te)},
           "_pod": "exact snapshot POD of the full training split",
           "_opine": a.opine_dir, "rows": []}

    # ---- one exact POD decomposition, sliced to every D ------------------
    ident = OPINE3D(max(a.dofs)).to(dev)
    ident.f = Identity3D().to(dev)
    t0 = time.time()
    Z = cache_latents(ident, tr_ld, dev)
    mu = Z.mean(0); Z -= mu.unsqueeze(0)
    U, err = exact_snapshot_pod(Z, max(a.dofs))
    del Z; torch.cuda.empty_cache()
    print(f"exact POD basis ready in {time.time()-t0:.0f}s, "
          f"|Q^TQ-I| {err:.1e}", flush=True)

    for D in a.dofs:
        row = {"dof": D}
        Q = U[:, :D].contiguous()
        row["POD"] = {"train": nmse_proj(ident, tr_ld, dev, Q, mu),
                      "val": nmse_proj(ident, va_ld, dev, Q, mu),
                      "test": nmse_proj(ident, te_ld, dev, Q, mu)}

        ck = os.path.join(a.base, a.opine_dir.format(D=D),
                          f"opine3d_D{D}_best.pt")
        if os.path.isfile(ck):
            m = OPINE3D(D, n_blocks=8, hidden=64, init_scale=0.05,
                        s_max=2.0).to(dev)
            m.load_state_dict(torch.load(ck, map_location=dev)["best_state"])
            m.eval()
            row["OPINE"] = {
                "train": nmse_proj(m, tr_ld, dev, m.Q.clone(),
                                   m.z_mean.clone()),
                "val": nmse_proj(m, va_ld, dev, m.Q.clone(),
                                 m.z_mean.clone()),
                "test": nmse_proj(m, te_ld, dev, m.Q.clone(),
                                  m.z_mean.clone()),
                "n_params": m.n_trainable(), "checkpoint": ck}
            del m; torch.cuda.empty_cache()

        cck = os.path.join(a.base, f"cnn_D{D}", f"CNN3D{D}_best.pt")
        if os.path.isfile(cck):
            c = JointCNN3D(input_shape=(NZ, NY, NX), latent_dim=D).to(dev)
            c.load_state_dict(torch.load(cck, map_location=dev))
            c.eval()
            row["CNN"] = {"train": nmse_cnn(c, tr_ld, dev),
                          "val": nmse_cnn(c, va_ld, dev),
                          "test": nmse_cnn(c, te_ld, dev),
                          "n_params": c.param_groups()["total"],
                          "checkpoint": cck}
            del c; torch.cuda.empty_cache()

        rep["rows"].append(row)
        line = f"D={D:<5}"
        for k in ("POD", "OPINE", "CNN"):
            if k in row:
                line += (f"  {k} tr {row[k]['train']:.6f} "
                         f"te {row[k]['test']:.6f}")
        print(line, flush=True)
        os.makedirs(os.path.dirname(a.output), exist_ok=True)
        json.dump(rep, open(a.output, "w"), indent=1)

    print(f"\nwrote {a.output}", flush=True)


if __name__ == "__main__":
    main()
