"""Nested-prefix test, channel y-z.  Inference only: f and Q are frozen."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.abspath("."))
from opine_experiments.channel2d.dataset_joint import ChannelFlowYZJointDatasetLocal
from opine_experiments.channel2d.train_latent_rank import load_joint_preprocessing
from opine_experiments.channel2d.coupling.coupling_fine_yz import CouplingFINE_YZ

NY, NZ, NC = 128, 64, 3
DATA = "data/retau180_yz_extracted"
CK = f"{DATA}/coupling_fine/D512/couplingyz_D512_best.pt"
K = 512
PREFIX = [32, 64, 128, 256, 512]          # stored rank -> reported D = r/2

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
mp, ps, _, _, _ = load_joint_preprocessing(f"{DATA}/preprocessing_yz_joint.json")
ds = lambda s: ChannelFlowYZJointDatasetLocal(f"{DATA}/{s}.npz", mp, ps, NY)
tr, te = ds("train"), ds("test")
mk = lambda d: torch.utils.data.DataLoader(d, batch_size=64, shuffle=False)

m = CouplingFINE_YZ(NY, NZ, K, channels=NC).to(dev)
sd = torch.load(CK, map_location=dev, weights_only=False)["best_state"]
m.load_state_dict(sd, strict=True); m.eval()
Q, mu = m.Q.clone(), m.z_mean.clone()
print(f"checkpoint {CK}\n  Q {tuple(Q.shape)}  |Q^TQ-I| "
      f"{float((Q.T@Q - torch.eye(K, device=dev)).abs().max()):.2e}")

# ---- column variance on the TRAIN split, and the K x K latent covariance
S = torch.zeros(K, K, dtype=torch.float64, device=dev); M = 0
for xb in mk(tr):
    a = (m.encode(xb.to(dev)) - mu.unsqueeze(0)) @ Q
    S += a.double().T @ a.double(); M += a.shape[0]
C = S / M
var_raw = torch.diagonal(C).cpu().numpy()
evals, V = torch.linalg.eigh(C)
order = torch.argsort(evals, descending=True)
evals, V = evals[order], V[:, order].float()
inv = int(np.sum(np.diff(var_raw) > 0))
print(f"  raw column variance: first {var_raw[0]:.4g}  last {var_raw[-1]:.4g}  "
      f"inversions {inv}/{K-1}  monotone={inv == 0}")
print(f"  eigen-spectrum      : first {float(evals[0]):.4g}  "
      f"last {float(evals[-1]):.4g}")

def nmse(Qr):
    num = den = 0.0
    for xb in mk(te):
        x = xb.to(dev)
        z = m.encode(x)
        zc = z - mu.unsqueeze(0)
        zh = (zc @ Qr) @ Qr.T + mu.unsqueeze(0)
        r = m.decode(zh)
        num += float((r - x).double().pow(2).sum())
        den += float(x.double().pow(2).sum())
    return num / den

rows = []
for r in PREFIX:
    raw = nmse(Q[:, :r].contiguous())
    ordd = nmse((Q @ V[:, :r]).contiguous())
    rows.append({"stored_rank": r, "reported_D": r // 2,
                 "prefix_raw": raw, "prefix_varordered": ordd,
                 "energy_frac": float(evals[:r].sum() / evals.sum())})
    print(f"  r={r:4d} (D={r//2:4d})  raw {raw:.6f}   var-ordered {ordd:.6f}"
          f"   energy {rows[-1]['energy_frac']:.4f}", flush=True)
json.dump({"checkpoint": CK, "K": K, "rows": rows,
           "raw_column_variance_inversions": inv,
           "band_ranks": m.band_ranks.cpu().tolist()},
          open("/tmp/prefix_yz.json", "w"), indent=1)
