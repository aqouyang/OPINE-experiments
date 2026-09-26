"""Nested-prefix test, Kolmogorov.  Inference only: f and Q frozen."""
import json, os, sys
import numpy as np, torch
R = os.environ["KOLMOGOROV_ROOT"]
sys.path.insert(0, R)
from opine_experiments.kolmogorov.canonical.dataset import make_datasets
from opine_experiments.kolmogorov.coupling.coupling_flow import CouplingFINE

BASE = f"{R}/data/kolmogorov/re100"
CK = f"{BASE}/results/coupling_fine_D512/coupling_D512_best.pt"
N, K = 128, 512
PREFIX = [32, 64, 128, 256, 512]

dev = torch.device("cuda")
torch.set_grad_enabled(False)
tr, _, te, _ = make_datasets(f"{BASE}/raw", f"{BASE}/splits.json")
rng = np.random.default_rng(42)                      # same recipe as latent dirs
idx = np.sort(rng.choice(len(tr), size=min(20000, len(tr)), replace=False))
sub = torch.utils.data.DataLoader(torch.utils.data.Subset(tr, idx.tolist()),
                                  batch_size=128, shuffle=False, num_workers=8)
te_ld = torch.utils.data.DataLoader(te, batch_size=128, shuffle=False,
                                    num_workers=8)

m = CouplingFINE(N, N, K, channels=1, n_blocks=8, hidden=64,
                 pad_mode="circular_both", init_scale=0.05, s_max=2.0,
                 bottleneck="unrestricted").to(dev)
st = torch.load(CK, map_location=dev, weights_only=False)
m.load_state_dict(st["best_state"]); m.eval()
Q, mu = m.Q.clone(), m.z_mean.clone()
print(f"checkpoint {CK}", flush=True)
print(f"  Q {tuple(Q.shape)}  |Q^TQ-I| "
      f"{float((Q.T@Q - torch.eye(K, device=dev)).abs().max()):.2e}", flush=True)

S = torch.zeros(K, K, dtype=torch.float64, device=dev); M = 0
for xb in sub:
    z = m.f(xb.to(dev)).reshape(xb.shape[0], -1)
    a = (z - mu.unsqueeze(0)) @ Q
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
        z = m.f(x).reshape(x.shape[0], -1)
        zc = z - mu.unsqueeze(0)
        zh = (zc @ Qr) @ Qr.T + mu.unsqueeze(0)
        r = m.f.inverse(zh.reshape(-1, 1, N, N))
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
    print(f"  r={r:4d} (D={r//2:4d})  raw {raw:.6f}  var-ordered {ordd:.6f}"
          f"  energy {rows[-1]['energy_frac']:.4f}", flush=True)
json.dump({"checkpoint": CK, "K": K, "rows": rows,
           "raw_column_variance_inversions": inv},
          open(f"{BASE}/results/diagnostics/prefix_kolmogorov.json", "w"), indent=1)
print("wrote prefix_kolmogorov.json", flush=True)
