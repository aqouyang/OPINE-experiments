#!/usr/bin/env python3
"""Controlled 1-D toy: can a coupling flow learn a nonlinear linearizing map?

    f_omega(x) = tanh(beta sin(x + omega)),    omega ~ U[0, 2pi)

Ground truth: atanh(f) = beta sin(x + omega) = beta[sin x cos w + cos x sin w],
so after the ideal invertible map the whole dataset lies in span{sin x, cos x}
-- EXACT rank 2. Raw f is not rank 2, because tanh saturation injects odd
harmonics whose amplitudes grow with beta.

POLYPHASE SPLIT IS FORCED, NOT CHOSEN. The field is scalar, and an affine
coupling has nothing to split on with one channel -- AffineCoupling1D would
build its conditioner with c_a = 1//2 = 0 inputs. So the flow is wrapped in an
even/odd polyphase squeeze, exactly the situation flagged in advance.

CONSEQUENCE FOR EQUIVARIANCE, stated before any result is quoted: a shift by
ONE grid cell exchanges the even and odd sublattices, so it acts as a channel
swap composed with a one-cell shift of a single channel. The flow is built
from circular convolutions and channelwise ops, so it commutes with EVEN
shifts exactly and has no reason to commute with odd ones. Exact TE is
therefore claimed only for even delta, and E_TE(delta) is measured for both.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))


# ── data ────────────────────────────────────────────────────────────────
def make_data(n, N, beta, seed):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = np.arange(N) * (2.0 * np.pi / N)
    lat = beta * np.sin(x[None, :] + w[:, None])       # oracle latent
    return np.tanh(lat)[:, None, :], lat[:, None, :], w


def nmse(a, b):
    return float(np.sum((a - b) ** 2) / np.sum(b ** 2))


def pod_fit(X):
    mean = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return Vt.T, mean, S


def pod_project(X, modes, mean, D):
    Q = modes[:, :D]
    return ((X - mean) @ Q) @ Q.T + mean


# ── polyphase wrapper ───────────────────────────────────────────────────
def squeeze1d(x):
    """(B,1,N) -> (B,2,N/2), even samples then odd."""
    return torch.cat([x[:, :, 0::2], x[:, :, 1::2]], dim=1)


def unsqueeze1d(y):
    b, _, h = y.shape
    out = y.new_empty(b, 1, 2 * h)
    out[:, 0, 0::2] = y[:, 0]
    out[:, 0, 1::2] = y[:, 1]
    return out


class PolyphaseFlow(nn.Module):
    """squeeze -> CouplingFlow1D(2 channels) -> unsqueeze. Shape preserving."""
    exact_inverse = True

    def __init__(self, **kw):
        super().__init__()
        from opine_experiments.synthetic.toy1d_coupling.models1d import CouplingFlow1D
        self.flow = CouplingFlow1D(channels=2, **kw)
        self.receptive_field = self.flow.receptive_field

    def forward(self, x):
        return unsqueeze1d(self.flow(squeeze1d(x)))

    def inverse(self, y):
        return unsqueeze1d(self.flow.inverse(squeeze1d(y)))


# ── translation-equivariance probe ──────────────────────────────────────
def te_error(recon_fn, X, deltas):
    """E_TE(d) = || R(T_d x) - T_d R(x) ||_2 / || x ||_2, averaged over X."""
    out = {}
    base = recon_fn(X)
    for d in deltas:
        a = recon_fn(np.roll(X, d, axis=-1))
        b = np.roll(base, d, axis=-1)
        out[int(d)] = float(np.linalg.norm(a - b) / np.linalg.norm(X))
    return out


# ── coupling-FINE ───────────────────────────────────────────────────────
def train_coupling(Xtr, Xva, Xte, N, dof, a, dev):
    from opine_experiments.synthetic.toy1d_coupling.models1d import FINE1D
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tf = PolyphaseFlow(n_blocks=a.n_blocks, hidden=a.hidden, kernel=a.kernel,
                       cond_layers=a.cond_layers)
    model = FINE1D(tf, 1, N, dof).to(dev)
    T = lambda z: torch.tensor(z, dtype=torch.float32, device=dev)
    Xt, Xv, Xs = T(Xtr), T(Xva), T(Xte)
    modes, mean, _ = pod_fit(Xtr.reshape(len(Xtr), -1))
    model.Q = T(modes[:, :dof]); model.z_mean = T(mean)
    rt0 = model.roundtrip_error(Xs[:32])

    def ev(Z):
        model.eval(); num = den = 0.0
        with torch.no_grad():
            for i in range(0, len(Z), 512):
                zb = Z[i:i + 512]
                r = model(zb)
                num += float((r - zb).double().pow(2).sum())
                den += float(zb.double().pow(2).sum())
        return num / den

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.rounds)
    g = torch.Generator().manual_seed(a.seed)
    best, best_state, hist = float("inf"), None, []
    for rnd in range(1, a.rounds + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=g).to(dev)
        for i in range(0, len(Xt) - a.batch + 1, a.batch):
            xb = Xt[perm[i:i + a.batch]]
            opt.zero_grad(set_to_none=True)
            loss = (model(xb) - xb).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.update_subspace(Xt, alpha=a.alpha)
        v, t = ev(Xv), ev(Xs)
        hist.append({"round": rnd, "val": v, "test": t})
        if v < best:
            best = v
            best_state = {k: x.detach().clone()
                          for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, ev(Xs), ev(Xt), rt0, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--betas", type=float, nargs="+", default=[5.0])
    ap.add_argument("--dofs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--cond-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, EPS = a.N, 1e-7
    res = {"config": vars(a), "betas": {}}

    for beta in a.betas:
        tr, ltr, _ = make_data(a.n_train, N, beta, a.seed)
        va, lva, _ = make_data(a.n_val, N, beta, a.seed + 1)
        te, lte, wte = make_data(a.n_test, N, beta, a.seed + 2)
        Ftr, Fte = tr.reshape(a.n_train, -1), te.reshape(a.n_test, -1)
        Ltr, Lte = ltr.reshape(a.n_train, -1), lte.reshape(a.n_test, -1)
        b = {"beta": beta}

        # ---- analytic sanity: oracle latent must be exactly rank 2 ----
        s_lat = np.linalg.svd(Ltr - Ltr.mean(0), compute_uv=False)
        rank = int((s_lat > s_lat[0] * 1e-10).sum())
        b["oracle_latent_numerical_rank"] = rank
        b["oracle_latent_sv_ratio_3_over_1"] = float(s_lat[2] / s_lat[0])
        # ---- raw POD spectrum ----
        m_raw, mu_raw, s_raw = pod_fit(Ftr)
        b["raw_energy_fraction"] = (s_raw[:12] ** 2 / (s_raw ** 2).sum()).tolist()
        b["oracle_energy_fraction"] = (s_lat[:12] ** 2
                                       / (s_lat ** 2).sum()).tolist()
        # oracle transform: atanh with clipping, POD there, tanh back
        m_or, mu_or, _ = pod_fit(np.arctanh(np.clip(Ftr, -1 + EPS, 1 - EPS)))

        def oracle_recon(Xf, D):
            z = np.arctanh(np.clip(Xf, -1 + EPS, 1 - EPS))
            return np.tanh(pod_project(z, m_or, mu_or, D))

        b["by_dof"] = {}
        for D in a.dofs:
            row = {"raw_pod": nmse(pod_project(Fte, m_raw, mu_raw, D), Fte),
                   "oracle": nmse(oracle_recon(Fte, D), Fte)}
            model, t_test, t_train, rt0, hist = train_coupling(
                tr, va, te, N, D, a, dev)
            row.update({"coupling": t_test, "coupling_train": t_train,
                        "coupling_gap": t_test - t_train,
                        "roundtrip_f64": rt0})
            if D == 2:                       # keep the D=2 model for figures
                Xp = te[:6]
                with torch.no_grad():
                    xt = torch.tensor(Xp, dtype=torch.float32, device=dev)
                    b["sample_truth"] = Xp[:, 0].tolist()
                    b["sample_coupling"] = model(xt)[:, 0].cpu().numpy().tolist()
                    b["sample_latent"] = model.f(xt)[:, 0].cpu().numpy().tolist()
                b["sample_raw_pod"] = pod_project(
                    Fte[:6], m_raw, mu_raw, 2).reshape(6, N).tolist()
                b["sample_oracle"] = oracle_recon(Fte[:6], 2).reshape(6, N).tolist()
                b["sample_omega"] = wte[:6].tolist()
                # ---- translation equivariance of the three operators ----
                probe = te[:256]
                deltas = [1, 2, 3, 4, 8, 16, 17, 33]
                mdl = model

                def cpl(Z):
                    with torch.no_grad():
                        zt = torch.tensor(Z, dtype=torch.float32, device=dev)
                        return mdl(zt).cpu().numpy()
                b["te_error"] = {
                    "raw_pod": te_error(
                        lambda Z: pod_project(Z.reshape(len(Z), -1), m_raw,
                                              mu_raw, 2).reshape(Z.shape),
                        probe, deltas),
                    "oracle": te_error(
                        lambda Z: oracle_recon(Z.reshape(len(Z), -1),
                                               2).reshape(Z.shape),
                        probe, deltas),
                    "coupling": te_error(cpl, probe, deltas)}
                b["coupling_history"] = hist
            b["by_dof"][str(D)] = row
            print(f"  beta={beta}  D={D}: raw {row['raw_pod']:.3e}  "
                  f"oracle {row['oracle']:.3e}  coupling {row['coupling']:.3e}"
                  f"  (gap {row['coupling_gap']:+.1e}, rt {row['roundtrip_f64']:.1e})",
                  flush=True)
        res["betas"][str(beta)] = b
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\n[wrote] {a.out}")


if __name__ == "__main__":
    main()
