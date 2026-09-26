#!/usr/bin/env python3
"""CNN autoencoder baseline for the canonical 1-D toy.

The shipped models/one_d/CNN.py cannot be used as-is: its bottleneck is
(latent_dim, 4), i.e. 4*D scalars, so at latent_dim=2 it would get 8 degrees
of freedom against everyone else's 2. This follows the project convention set
by KolmogorovCNN instead -- strided conv encoder, a linear layer down to
EXACTLY D scalars, linear back up, transposed-conv decoder -- so D means the
same thing it means for POD and Coupling-FINE.

Circular padding throughout: the domain is periodic.

The CNN is an UNCONSTRAINED reference, not a like-for-like method. It has no
POD bottleneck, it is not invertible, and its decoder can simply memorise the
fixed waveform tanh(sin(.)) and use the 2 latent scalars to encode phase --
which is precisely the mechanism the toy exists to distinguish itself from.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from toy_canonical import fields
from toy_tanh import nmse

N = 128


class CircConv1d(nn.Module):
    def __init__(self, ci, co, k=3, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(ci, co, k, stride=stride, padding=0)
        self.k = k

    def forward(self, x):
        p = self.k // 2
        return self.conv(F.pad(x, (p, p), mode="circular"))


class Toy1DCNN(nn.Module):
    """(B,1,128) -> latent (B, D) -> (B,1,128)."""

    def __init__(self, latent_dim=2, base=16, n_down=4):
        super().__init__()
        enc, c_in, c = [], 1, base
        for _ in range(n_down):
            enc += [CircConv1d(c_in, c, 3, stride=2), nn.BatchNorm1d(c),
                    nn.ReLU(inplace=True)]
            c_in, c = c, min(c * 2, base * 8)
        self.encoder = nn.Sequential(*enc)
        self.bl, self.bc = N // 2 ** n_down, c_in
        flat = self.bc * self.bl
        self.enc_linear = nn.Linear(flat, latent_dim)
        self.dec_linear = nn.Linear(latent_dim, flat)
        dec, c = [], self.bc
        for _ in range(n_down - 1):
            co = max(c // 2, base)
            dec += [nn.Upsample(scale_factor=2, mode="linear",
                                align_corners=False),
                    CircConv1d(c, co, 3), nn.BatchNorm1d(co),
                    nn.ReLU(inplace=True)]
            c = co
        dec += [nn.Upsample(scale_factor=2, mode="linear",
                            align_corners=False), CircConv1d(c, 1, 3)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        z = self.enc_linear(h)
        g = self.dec_linear(z).view(-1, self.bc, self.bl)
        return self.decoder(g), z


def run_seed(seed, D, a, dev):
    rng = np.random.default_rng(1000 + seed)
    w_tr = rng.uniform(0, 2*np.pi, a.n_train)
    w_va = rng.uniform(0, 2*np.pi, a.n_val)
    w_te = rng.uniform(0, 2*np.pi, a.n_test)
    T = lambda w: torch.tensor(fields(w), dtype=torch.float32, device=dev)
    Xt, Xv, Xs = T(w_tr), T(w_va), T(w_te)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    m = Toy1DCNN(latent_dim=D).to(dev)
    npar = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    g = torch.Generator().manual_seed(a.seed)
    bs = min(a.batch, len(Xt))

    def ev(Z):
        m.eval()
        with torch.no_grad():
            r = m(Z)[0]
            return float((r - Z).double().pow(2).sum()
                         / Z.double().pow(2).sum())
    best, bstate = float("inf"), None
    for _ in range(a.epochs):
        m.train()
        perm = torch.randperm(len(Xt), generator=g).to(dev)
        for i in range(0, len(Xt) - bs + 1, bs):
            xb = Xt[perm[i:i+bs]]
            opt.zero_grad(set_to_none=True)
            loss = (m(xb)[0] - xb).pow(2).mean()
            loss.backward(); opt.step()
        sch.step()
        v = ev(Xv)
        if v < best:
            best, bstate = v, {k: x.detach().clone()
                               for k, x in m.state_dict().items()}
    m.load_state_dict(bstate)
    return {"seed": seed, "D": D, "train": ev(Xt), "val": best,
            "test": ev(Xs), "params": npar}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--dofs", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"config": vars(a), "by_dof": {}}
    for D in a.dofs:
        rs = [run_seed(s, D, a, dev) for s in range(a.seeds)]
        t = np.array([r["test"] for r in rs])
        tr = np.array([r["train"] for r in rs])
        out["by_dof"][str(D)] = {
            "runs": rs, "test_mean": float(t.mean()), "test_std": float(t.std()),
            "train_mean": float(tr.mean()), "params": rs[0]["params"],
            "gap_mean": float((t - tr).mean())}
        print(f"  D={D}  CNN test {t.mean():.4e} +/- {t.std():.2e}   "
              f"train {tr.mean():.4e}   gap {(t-tr).mean():+.2e}   "
              f"params {rs[0]['params']:,}", flush=True)
    json.dump(out, open(os.path.join(HERE, "cnn_results.json"), "w"), indent=2)
    print("\n[wrote] cnn_results.json")


if __name__ == "__main__":
    main()
