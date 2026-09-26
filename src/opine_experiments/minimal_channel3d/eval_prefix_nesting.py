"""Nested-prefix test, 3-D minimal channel.  Inference only: f and Q frozen."""
import json, os, sys
import numpy as np, torch
R = os.environ["CHANNEL3D_ROOT"]
sys.path.insert(0, R); sys.path.insert(0, f"{R}/experiments/channel3d_minimal")
from dataset3d import make_datasets, N_FEAT
from coupling_flow_3d import OPINE3D

B = f"{R}/data/channel3d"
CK = f"{B}/opine_full_D2048/opine3d_D2048_best.pt"
K = 2048
PREFIX = [32, 64, 128, 256, 512, 1024, 2048]

dev = torch.device("cuda")
torch.set_grad_enabled(False)
tr, va, te, _, _ = make_datasets(f"{B}/stats_train.npz")
mk = lambda d, bs: torch.utils.data.DataLoader(d, batch_size=bs, shuffle=False,
                                               num_workers=8, pin_memory=True)
tr_ld, te_ld = mk(tr, 32), mk(te, 32)

m = OPINE3D(K, n_blocks=8, hidden=64, init_scale=0.05, s_max=2.0).to(dev)
m.load_state_dict(torch.load(CK, map_location=dev)["best_state"]); m.eval()
Q, mu = m.Q.clone(), m.z_mean.clone()
print(f"checkpoint {CK}", flush=True)
print(f"  Q {tuple(Q.shape)}  |Q^TQ-I| "
      f"{float((Q.T@Q - torch.eye(K, device=dev)).abs().max()):.2e}", flush=True)

S = torch.zeros(K, K, dtype=torch.float64, device=dev); M = 0
for xb in tr_ld:
    a = (m.encode(xb.to(dev)) - mu.unsqueeze(0)) @ Q
    S += a.double().T @ a.double(); M += a.shape[0]
C = S / M
var_raw = torch.diagonal(C).cpu().numpy()
evals, V = torch.linalg.eigh(C)
o = torch.argsort(evals, descending=True); evals, V = evals[o], V[:, o].float()
inv = int(np.sum(np.diff(var_raw) > 0))
print(f"  raw column variance: first {var_raw[0]:.4g} last {var_raw[-1]:.4g} "
      f"inversions {inv}/{K-1} monotone={inv == 0}", flush=True)

def nmse(Qr):
    num = den = 0.0
    for xb in te_ld:
        x = xb.to(dev)
        zc = m.encode(x) - mu.unsqueeze(0)
        zh = (zc @ Qr) @ Qr.T + mu.unsqueeze(0)
        r = m.decode(zh)
        num += float((r - x).double().pow(2).sum())
        den += float(x.double().pow(2).sum())
    return num / den

rows = []
for r in PREFIX:
    raw = nmse(Q[:, :r].contiguous())
    ordd = nmse((Q @ V[:, :r]).contiguous())
    rows.append({"stored_rank": r, "reported_D": r // 2, "prefix_raw": raw,
                 "prefix_varordered": ordd,
                 "energy_frac": float(evals[:r].sum() / evals.sum())})
    print(f"  r={r:5d} (D={r//2:5d})  raw {raw:.6f}  var-ordered {ordd:.6f}"
          f"  energy {rows[-1]['energy_frac']:.4f}", flush=True)
json.dump({"checkpoint": CK, "K": K, "rows": rows,
           "raw_column_variance_inversions": inv},
          open(f"{B}/validation/prefix_channel3d.json", "w"), indent=1)
print("wrote prefix_channel3d.json", flush=True)
