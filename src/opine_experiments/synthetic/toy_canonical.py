#!/usr/bin/env python3
"""The canonical 1-D toy:  f_omega(x) = tanh(sin(x + omega)),  omega in [0,2pi).

    atanh(f) = sin(x + omega) = sin(x)cos(omega) + cos(x)sin(omega)

so after the ideal invertible map the data lies EXACTLY in span{sin x, cos x}.

Three splits with identical architecture, optimizer, epochs and D = 2:

  dense    omega drawn at random for every split
  blocked  train/val in [0, pi), TEST in [pi, 2pi)
  sparse32 train on 32 uniformly spaced phases, test on unseen random phases

ORACLE USES THE ANALYTIC BASIS, not a fitted POD. Projecting atanh(f) onto
span{sin x, cos x} is split-independent by construction, which is exactly what
makes it a reference: any dependence on the training distribution would make
it a fourth model rather than a ceiling.

NEAREST-PHASE COPY is included as a no-training control. On a densely sampled
one-parameter manifold, copying the closest training waveform is already a
strong reconstructor, so it measures how much of any method's score is just
interpolation density.
"""
import argparse, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from toy_tanh import PolyphaseFlow, pod_fit, pod_project, nmse, te_error
from opine_experiments.synthetic.toy1d_coupling.models1d import FINE1D

N, EPS = 128, 1e-7
X = np.arange(N) * (2.0 * np.pi / N)
SIN, COS = np.sin(X), np.cos(X)
BASIS = np.stack([SIN, COS])
BASIS /= np.linalg.norm(BASIS, axis=1, keepdims=True)   # orthonormal
DELTAS = [1, 2, 3, 4, 5, 8, 13, 16, 32]


def fields(w):
    return np.tanh(np.sin(X[None, :] + np.asarray(w)[:, None]))[:, None, :]


def oracle(Xf):
    """atanh -> project onto span{sin,cos} -> tanh. Analytic, split-free."""
    z = np.arctanh(np.clip(Xf, -1 + EPS, 1 - EPS))
    return np.tanh((z @ BASIS.T) @ BASIS)


def nearest_copy(w_te, w_tr, Ftr):
    """No-training control: reconstruct with the closest training waveform."""
    d = np.abs(w_te[:, None] - w_tr[None, :])
    d = np.minimum(d, 2 * np.pi - d)
    j = d.argmin(1)
    return Ftr[j], d[np.arange(len(w_te)), j]


def train_coupling(tr, va, te, a, dev, dof=2):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = FINE1D(PolyphaseFlow(n_blocks=4, hidden=32, kernel=5,
                                 cond_layers=4), 1, N, dof).to(dev)
    T = lambda z: torch.tensor(z, dtype=torch.float32, device=dev)
    Xt, Xv, Xs = T(tr), T(va), T(te)
    m, mu, _ = pod_fit(tr.reshape(len(tr), -1))
    model.Q = T(m[:, :dof]); model.z_mean = T(mu)

    def ev(Z):
        model.eval(); num = den = 0.0
        with torch.no_grad():
            for i in range(0, len(Z), 512):
                zb = Z[i:i + 512]
                num += float((model(zb) - zb).double().pow(2).sum())
                den += float(zb.double().pow(2).sum())
        return num / den

    bs = min(a.batch, len(Xt))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.rounds)
    g = torch.Generator().manual_seed(a.seed)
    best, bstate = float("inf"), None
    for _ in range(a.rounds):
        model.train()
        perm = torch.randperm(len(Xt), generator=g).to(dev)
        for i in range(0, len(Xt) - bs + 1, bs):
            xb = Xt[perm[i:i + bs]]
            opt.zero_grad(set_to_none=True)
            loss = (model(xb) - xb).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.update_subspace(Xt, alpha=a.alpha)
        v = ev(Xv)
        if v < best:
            best, bstate = v, {k: x.detach().clone()
                               for k, x in model.state_dict().items()}
    model.load_state_dict(bstate)
    return model, ev(Xt), best, ev(Xs)


def latent_rank(model, te, dev):
    """Is the LEARNED latent actually close to rank 2?"""
    with torch.no_grad():
        Z = model.encode(torch.tensor(te, dtype=torch.float32,
                                      device=dev)).cpu().numpy()
        Q = model.Q.cpu().numpy(); mu = model.z_mean.cpu().numpy()
    Zc = Z - Z.mean(0)
    s = np.linalg.svd(Zc, compute_uv=False)
    resid = Z - mu - ((Z - mu) @ Q) @ Q.T
    return {"latent_top2_energy_fraction": float((s[:2] ** 2).sum()
                                                 / (s ** 2).sum()),
            "latent_sv_ratio_3_over_1": float(s[2] / s[0]),
            "latent_projection_residual": float((resid ** 2).sum()
                                                / (Z ** 2).sum())}


def run(tag, w_tr, w_va, w_te, a, dev):
    tr, va, te = fields(w_tr), fields(w_va), fields(w_te)
    Ftr, Fte = tr.reshape(len(tr), -1), te.reshape(len(te), -1)
    m_raw, mu_raw, s_raw = pod_fit(Ftr)
    model, c_tr, c_va, c_te = train_coupling(tr, va, te, a, dev)
    nn_rec, gap = nearest_copy(w_te, w_tr, Ftr)
    fgap = np.linalg.norm(nn_rec - Fte, axis=1) / np.linalg.norm(Fte, axis=1)

    raw_rec = pod_project(Fte, m_raw, mu_raw, 2)
    orc_rec = oracle(Fte)
    r = {"tag": tag, "n_train": len(w_tr), "n_test": len(w_te),
         "coupling_train": c_tr, "coupling_val": c_va, "coupling_test": c_te,
         "coupling_train_test_gap": c_te - c_tr,
         "raw_pod_test": nmse(raw_rec, Fte),
         "oracle_test": nmse(orc_rec, Fte),
         "nearest_copy_test": nmse(nn_rec, Fte),
         "nearest_phase_gap_rad": {"median": float(np.median(gap)),
                                   "max": float(gap.max())},
         "nearest_waveform_distance": {"median": float(np.median(fgap)),
                                       "max": float(fgap.max())},
         "raw_energy_fraction": (s_raw[:12] ** 2
                                 / (s_raw ** 2).sum()).tolist(),
         **latent_rank(model, te, dev)}
    r["raw_over_coupling"] = r["raw_pod_test"] / r["coupling_test"]

    def cpl(Z):
        with torch.no_grad():
            return model(torch.tensor(Z, dtype=torch.float32,
                                      device=dev)).cpu().numpy()
    probe = te[:256]
    r["te_error"] = {
        "raw_pod": te_error(lambda Z: pod_project(
            Z.reshape(len(Z), -1), m_raw, mu_raw, 2).reshape(Z.shape),
            probe, DELTAS),
        "oracle": te_error(lambda Z: oracle(
            Z.reshape(len(Z), -1)).reshape(Z.shape), probe, DELTAS),
        "coupling": te_error(cpl, probe, DELTAS)}

    k = [0, len(w_te) // 4, len(w_te) // 2, 3 * len(w_te) // 4]
    r["samples"] = {"omega": [float(w_te[i]) for i in k],
                    "truth": Fte[k].tolist(),
                    "raw_pod": raw_rec[k].tolist(),
                    "coupling": cpl(te[k])[:, 0].tolist(),
                    "oracle": orc_rec[k].tolist()}
    print(f"\n=== {tag} ===")
    print(f"  n_train {len(w_tr):<7} nearest train phase  median "
          f"{np.median(gap):.3e} rad  max {gap.max():.3e}")
    print(f"  {'':<16} nearest train waveform median {np.median(fgap):.3e}  "
          f"max {fgap.max():.3e}")
    print(f"  Coupling-FINE   train {c_tr:.3e}  val {c_va:.3e}  "
          f"TEST {c_te:.3e}   gap {c_te - c_tr:+.2e}")
    print(f"  Raw POD   D=2   TEST {r['raw_pod_test']:.3e}")
    print(f"  Oracle    D=2   TEST {r['oracle_test']:.3e}")
    print(f"  nearest-phase copy  TEST {r['nearest_copy_test']:.3e}   "
          f"(no training)")
    print(f"  raw / coupling  = {r['raw_over_coupling']:.1f}x")
    print(f"  learned latent: top-2 energy {r['latent_top2_energy_fraction']:.6f}"
          f"   sv3/sv1 {r['latent_sv_ratio_3_over_1']:.2e}"
          f"   ||z-QQ^Tz||^2/||z||^2 {r['latent_projection_residual']:.2e}")
    t = r["te_error"]["coupling"]
    print("  E_TE coupling: " + "  ".join(
        f"{d}{'*' if d % 2 else ''}={t[d]:.1e}" for d in DELTAS))
    print(f"         raw max {max(r['te_error']['raw_pod'].values()):.1e}"
          f"   oracle max {max(r['te_error']['oracle'].values()):.1e}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    out = {}
    out["dense"] = run("dense", rng.uniform(0, 2*np.pi, 20000),
                       rng.uniform(0, 2*np.pi, 2000),
                       rng.uniform(0, 2*np.pi, 2000), a, dev)
    w = rng.uniform(0.0, np.pi, 22000)
    out["blocked"] = run("blocked", w[:20000], w[20000:],
                         rng.uniform(np.pi, 2*np.pi, 2000), a, dev)
    out["sparse32"] = run("sparse32", np.arange(32) * (2*np.pi/32),
                          rng.uniform(0, 2*np.pi, 256),
                          rng.uniform(0, 2*np.pi, 2000), a, dev)
    json.dump(out, open(os.path.join(HERE, "canonical_results.json"), "w"),
              indent=2)
    print("\n[wrote] canonical_results.json")


if __name__ == "__main__":
    main()
