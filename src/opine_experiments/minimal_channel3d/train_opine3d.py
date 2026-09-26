#!/usr/bin/env python3
"""OPINE on the 3-D minimal channel. One DOF per invocation.

Two-phase rounds, as in every other OPINE run:
  Phase A   gradient on theta with Q fixed
  Phase B   one relaxed subspace-iteration step on Q with theta fixed
Q and mu_z are buffers and never receive gradient. alpha is pinned at the
canonical 0.03.

`--identity` replaces f_theta by the exact identity, which turns the same code
path into the plain POD baseline on the same split, same normalization and
same metric. That is how the POD column is produced -- not by a separate
implementation.
"""
import argparse, hashlib, json, math, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.append(ROOT)
sys.path.insert(0, HERE)
from dataset3d import make_datasets, N_FEAT
from coupling_flow_3d import (OPINE3D, Identity3D, streaming_subspace_step,
                              init_subspace_pod)


def sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except OSError:
        return "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dof", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--stats-cache", required=True)
    ap.add_argument("--identity", action="store_true",
                    help="f_theta = I exactly -> plain POD baseline")
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--s-max", type=float, default=2.0)
    ap.add_argument("--init-scale", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--inner-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sub-batch", type=int, default=32)
    ap.add_argument("--subspace-subsample", type=int, default=4000)
    ap.add_argument("--init-power", type=int, default=12)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)
    tag = f"{'pod' if a.identity else 'opine'}3d_D{a.dof}"

    tr, va, te, mean, std = make_datasets(a.stats_cache)
    mk = lambda ds, sh, bs: torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=sh, num_workers=a.workers,
        pin_memory=True, persistent_workers=a.workers > 0)
    train_ld = mk(tr, True, a.batch_size)
    val_ld, test_ld = mk(va, False, a.sub_batch), mk(te, False, a.sub_batch)
    rng = np.random.default_rng(a.seed)
    sub_idx = np.sort(rng.choice(len(tr),
                                 size=min(a.subspace_subsample, len(tr)),
                                 replace=False))
    sub_ld = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr, sub_idx.tolist()),
        batch_size=a.sub_batch, shuffle=False, num_workers=a.workers)

    model = OPINE3D(a.dof, n_blocks=a.n_blocks, hidden=a.hidden,
                    init_scale=a.init_scale, s_max=a.s_max).to(dev)
    if a.identity:
        model.f = Identity3D().to(dev)
    n_par = model.n_trainable()
    probe = torch.stack([te[i] for i in range(4)]).to(dev)
    rt0 = model.f.roundtrip_errors(probe)
    eq0 = model.f.equivariance_error(probe)
    with torch.no_grad():
        init_dev = float((model.f(probe) - probe).norm() / probe.norm())
    print(f"[{tag}] N={N_FEAT}  params {n_par:,}  rt64 "
          f"{rt0['verified_float64']:.2e}  equiv x {eq0['x']:.1e} y "
          f"{eq0['y']:.1e}  |f-x|/|x| {init_dev:.3e}", flush=True)

    t0 = time.time()
    init_subspace_pod(model, sub_ld, dev, n_power=a.init_power, seed=a.seed)
    Q_init = model.Q.clone()
    print(f"[{tag}] Q initialised in {time.time()-t0:.0f}s", flush=True)

    def ev(ld):
        model.eval(); num = den = 0.0
        with torch.no_grad():
            for xb in ld:
                xb = xb.to(dev, non_blocking=True)
                r = model(xb)
                num += float((r - xb).double().pow(2).sum())
                den += float(xb.double().pow(2).sum())
        return num / den

    round0 = ev(test_ld)
    print(f"[{tag}] round-0 test NMSE {round0:.8f}", flush=True)
    res = {"tag": tag, "dof": a.dof, "identity": a.identity,
           "n_trainable_params": n_par, "ambient_dim": N_FEAT,
           "round0_test_nmse": round0,
           "init_f_deviation_from_identity": init_dev,
           "init_roundtrip_float64": rt0["verified_float64"],
           "init_equivariance": eq0,
           "_metric": "sum|recon-truth|^2/sum|truth|^2 on normalized fields",
           "_split": "chronological 16000/2000/2000",
           "_normalization": "per (component, z) mean and rms from the "
                             "training split only; homogeneous x,y averaged",
           "code_revision": {"flow": sha(os.path.join(HERE,
                                                      "coupling_flow_3d.py")),
                             "dataset": sha(os.path.join(HERE,
                                                         "dataset3d.py"))},
           "config": vars(a), "history": []}
    if a.identity:                       # POD: no training to do
        res.update({"selected_test_nmse": round0, "best_round": 0,
                    "runtime_s": time.time() - t0})
        json.dump(res, open(os.path.join(a.output_dir, f"{tag}_results.json"),
                            "w"), indent=2, default=float)
        print(f"\n[{tag}] POD test NMSE {round0:.8f}", flush=True)
        return

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.rounds)
    best_val, best_round, best_state, start = float("inf"), 0, None, 1
    state_path = os.path.join(a.output_dir, f"{tag}_state.pt")
    if a.resume and os.path.isfile(state_path):
        st = torch.load(state_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); res["history"] = st["history"]
        best_val, best_round = st["best_val"], st["best_round"]
        best_state = st["best_state"]; Q_init = st["Q_init"].to(dev)
        start = st["round"] + 1
        torch.set_rng_state(st["rng"].cpu())
        print(f"[{tag}] resumed from round {st['round']}", flush=True)

    for rnd in range(start, a.rounds + 1):
        model.train(); run, nb = 0.0, 0
        for _ in range(a.inner_epochs):
            for xb in train_ld:
                xb = xb.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = (model(xb) - xb).pow(2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step()
                run += float(loss); nb += 1
        sched.step()
        sub = streaming_subspace_step(model, sub_ld, dev, alpha=a.alpha)
        rt = model.f.roundtrip_errors(probe)
        eq = model.f.equivariance_error(probe)
        with torch.no_grad():
            dev_id = float((model.f(probe) - probe).norm() / probe.norm())
        v, t = ev(val_ld), ev(test_ld)
        if v < best_val:
            best_val, best_round = v, rnd
            best_state = {k: x.detach().clone()
                          for k, x in model.state_dict().items()}
        from opine_experiments.channel2d.iresnet_fine_yz import (
            principal_angles)
        dq = float(principal_angles(Q_init, model.Q).max()) * 180 / math.pi
        res["history"].append(
            {"round": rnd, "train_mse": run / nb, "val_nmse": v,
             "test_nmse": t, "roundtrip_float64": rt["verified_float64"],
             "equiv_x": eq["x"], "equiv_y": eq["y"],
             "f_deviation_from_identity_rel": dev_id,
             "drift_max_angle_deg": dq,
             "subspace_step_angle_deg": sub["max_principal_angle_deg"]})
        print(f"[{tag}] R{rnd:3d}/{a.rounds} train {run/nb:.6f}  val {v:.6f}"
              f"  test {t:.6f}  ({100*(round0-t)/round0:+.2f}% vs round-0)"
              f"  dQ {dq:5.2f}deg  rt {rt['verified_float64']:.1e}"
              f"  eqx {eq['x']:.1e}  dev {dev_id:.3f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        if rt["verified_float64"] > 1e-6:
            raise RuntimeError(f"round {rnd}: roundtrip "
                               f"{rt['verified_float64']:.2e}")
        # home is NFS-backed; torch.save alone left the .tmp invisible to a
        # following os.replace once (D=128, round 11).  Flush and fsync the
        # handle, then confirm the file is there before renaming, and retry
        # rather than lose a round's work.
        payload = {"model": model.state_dict(), "opt": opt.state_dict(),
                   "sched": sched.state_dict(), "history": res["history"],
                   "best_val": best_val, "best_round": best_round,
                   "best_state": best_state, "Q_init": Q_init.cpu(),
                   "round": rnd, "rng": torch.get_rng_state()}
        for attempt in range(3):
            try:
                with open(state_path + ".tmp", "wb") as fh:
                    torch.save(payload, fh)
                    fh.flush()
                    os.fsync(fh.fileno())
                if not os.path.isfile(state_path + ".tmp"):
                    raise OSError("checkpoint .tmp missing after fsync")
                os.replace(state_path + ".tmp", state_path)
                break
            except OSError as exc:
                print(f"[{tag}] checkpoint attempt {attempt+1} failed: "
                      f"{exc}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(5)

    model.load_state_dict(best_state)
    res.update({"best_round": best_round, "best_val_nmse": best_val,
                "selected_test_nmse": ev(test_ld),
                "selected_train_nmse": None,
                "final_roundtrip_float64":
                    model.f.roundtrip_errors(probe)["verified_float64"],
                "final_equivariance": model.f.equivariance_error(probe),
                "runtime_s": time.time() - t0})
    json.dump(res, open(os.path.join(a.output_dir, f"{tag}_results.json"), "w"),
              indent=2, default=float)
    torch.save({"best_state": best_state, "Q_init": Q_init.cpu()},
               os.path.join(a.output_dir, f"{tag}_best.pt"))
    print(f"\n[{tag}] round-0 {round0:.6f} -> {res['selected_test_nmse']:.6f}"
          f"  round {best_round}  params {n_par:,}", flush=True)


if __name__ == "__main__":
    main()
