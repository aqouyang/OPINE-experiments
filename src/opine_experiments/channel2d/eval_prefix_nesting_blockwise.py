#!/usr/bin/env python3
"""Symmetry-preserving nested-prefix test for the y-z channel.  Inference only.

The banded bottleneck stores one complex basis B_k per k_y band.  A GLOBAL
Rayleigh-Ritz rotation of the assembled Q mixes bands, so a truncated
projector stops being y-translation equivariant.  Here the Ritz rotation is
done INSIDE each band,

    M_k = B_k^H C_k B_k,   eigh,   B_k <- B_k V_k,

which leaves span(B_k) and therefore the full-rank projector untouched.  The
resulting modes are then ranked globally by transformed variance, and a
truncation keeps only COMPLETE band modes: each interior k_y mode carries 2
real DOF, DC and Nyquist carry 1.  Every retained subspace is then a direct
sum of whole band modes, so every truncated projector is exactly equivariant.

f_theta, the learned subspace and the checkpoint are never modified.
"""
import json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.abspath("."))
from opine_experiments.channel2d.dataset_joint import ChannelFlowYZJointDatasetLocal
from opine_experiments.channel2d.train_latent_rank import load_joint_preprocessing
from opine_experiments.channel2d.coupling.coupling_fine_yz import CouplingFINE_YZ
from opine_experiments.channel2d.iresnet_fine_yz import (
    latent_csym_blocks, blocks_to_dense_Q, is_real_ky,
    block_projector_equivariance_error)

NY, NZ, NC, K = 128, 64, 3, 512
DATA = "data/retau180_yz_extracted"
CK = f"{DATA}/coupling_fine/D512/couplingyz_D512_best.pt"
PREFIX = [32, 64, 128, 256, 512]

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
mp, ps, _, _, _ = load_joint_preprocessing(f"{DATA}/preprocessing_yz_joint.json")
ds = lambda s: ChannelFlowYZJointDatasetLocal(f"{DATA}/{s}.npz", mp, ps, NY)
tr, te = ds("train"), ds("test")
mk = lambda d: torch.utils.data.DataLoader(d, batch_size=64, shuffle=False)

m = CouplingFINE_YZ(NY, NZ, K, channels=NC).to(dev)
m.load_state_dict(torch.load(CK, map_location=dev, weights_only=False)["best_state"])
m.eval()
Q0, mu = m.Q.clone(), m.z_mean.clone()
bases0 = [b.clone() for b in m._bases_list()]
ranks0 = [int(r) for r in m.band_ranks]
n_ky = len(ranks0)
print(f"checkpoint {CK}\n  Q {tuple(Q0.shape)}  bands {n_ky}  "
      f"sum real DOF {sum((1 if is_real_ky(k, NY) else 2) * r for k, r in enumerate(ranks0))}")

# ---- training latents, centred, and the per-band symmetrized covariance
Zc, _ = m.latent_matrix(mk(tr), dev)
blocks = latent_csym_blocks(Zc, NY, NC)
del Zc; torch.cuda.empty_cache()

# ---- Ritz rotation INSIDE each band
rot, modes = [], []
for k, B in enumerate(bases0):
    if B.shape[1] == 0:
        rot.append(B); continue
    Bc = B.to(torch.complex128)
    Mk = Bc.conj().transpose(-1, -2) @ blocks[k] @ Bc
    Mk = 0.5 * (Mk + Mk.conj().transpose(-1, -2))          # exact Hermitian
    ev, V = torch.linalg.eigh(Mk)
    o = torch.argsort(ev.real, descending=True)
    rot.append((Bc @ V[:, o]).to(B.dtype))
    w = 1 if is_real_ky(k, NY) else 2
    for j, lam in enumerate(ev.real[o].tolist()):
        modes.append((lam, k, j, w))
modes.sort(key=lambda t: -t[0])
print(f"  band modes {len(modes)}  leading eigenvalue {modes[0][0]:.4g}  "
      f"smallest {modes[-1][0]:.4g}")

def build(target_real_dof):
    """Greedy global ranking, whole band modes only."""
    take = [0] * n_ky
    used = 0
    for lam, k, j, w in modes:
        if used + w > target_real_dof:
            continue
        if take[k] == j:                     # keep each band's modes contiguous
            take[k] += 1
            used += w
        if used == target_real_dof:
            break
    sub = [rot[k][:, :take[k]] for k in range(n_ky)]
    Q = blocks_to_dense_Q(sub, take, NY, NZ, NC).to(Q0.dtype)
    return Q, take, used

def nmse(Qr):
    num = den = 0.0
    for xb in mk(te):
        x = xb.to(dev)
        zc = m.encode(x) - mu.unsqueeze(0)
        zh = (zc @ Qr) @ Qr.T + mu.unsqueeze(0)
        r = m.decode(zh)
        num += float((r - x).double().pow(2).sum())
        den += float(x.double().pow(2).sum())
    return num / den

Qfull, take_full, used_full = build(K)
print(f"  full-rank rebuild: real DOF {used_full}, bands used "
      f"{sum(1 for t in take_full if t)}")
print("  max |Q0 Q0^T - Qf Qf^T| = %.3e"
      % float((Q0 @ Q0.T - Qfull @ Qfull.T).abs().max()))

prev = json.load(open(os.environ.get(
    "PREFIX_NESTING_YZ_JSON",
    "provenance/channel2d/prefix_nesting_yz.json")))
prevmap = {r["stored_rank"]: r for r in prev["rows"]}

rows = []
for r in PREFIX:
    Qr, take, used = build(r)
    e = nmse(Qr)
    eq = block_projector_equivariance_error(Qr.contiguous(), NY, NZ, NC)
    rows.append({"stored_rank": r, "reported_D": r // 2, "real_dof_used": used,
                 "n_bands": int(sum(1 for t in take if t)),
                 "blockwise_nmse": e, "projector_equivariance_error": eq,
                 "prev_global_rotation_nmse": prevmap[r]["prefix_varordered"],
                 "prev_raw_nmse": prevmap[r]["prefix_raw"],
                 "band_counts": take})
    print(f"  r={r:4d} (D={r//2:4d})  dof {used:4d}  bands {rows[-1]['n_bands']:3d}"
          f"  NMSE {e:.6f}  equiv {eq:.3e}", flush=True)

out = f"experiments/channel_retau180_yz/production_results/main_figures/prefix_nesting_yz_blockwise.json"
json.dump({"checkpoint": CK, "K": K,
           "ordering": "Ritz rotation inside each k_y band, then global ranking "
                       "by transformed variance over whole band modes",
           "full_rank_projector_max_abs_diff":
               float((Q0 @ Q0.T - Qfull @ Qfull.T).abs().max()),
           "rows": rows}, open(out, "w"), indent=1)
print("wrote", out)
