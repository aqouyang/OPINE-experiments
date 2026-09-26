#!/usr/bin/env python3
"""Coupling-FINE on the channel y-z planes. One DOF per invocation.

Everything except f_theta matches the canonical channel setting: same
extracted splits, same joint preprocessing, same sympod Q initialization,
same canonical subspace update at alpha=0.03, same validation-only selection,
same canonical NMSE. Outputs are prefixed `couplingyz_` and live in their own
tree, so they cannot be confused with the iResNet runs.
"""
import argparse, hashlib, json, math, os, subprocess, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
NY, NZ, NCOMP = 128, 64, 3
DEG = 180.0 / math.pi


def code_revision():
    rev, here = {}, os.path.dirname(os.path.abspath(__file__))
    for nm, rel in (("flow_yz", os.path.join(here, "coupling_fine_yz.py")),
                    ("flow_base", os.path.join(
                        ROOT, "experiments", "kolmogorov", "coupling",
                        "coupling_flow.py")),
                    ("model", os.path.join(
                        ROOT, "experiments", "channel_retau180_yz",
                        "iresnet_fine_yz.py"))):
        try:
            rev[f"sha256_{nm}"] = hashlib.sha256(
                open(rel, "rb").read()).hexdigest()
        except OSError as e:
            rev[f"sha256_{nm}"] = f"unavailable: {e}"
    try:
        rev["git_head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, timeout=20).stdout.strip()
    except Exception as e:
        rev["git_head"] = f"unavailable: {e}"
    return rev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-dir", required=True)
    ap.add_argument("--preprocessing-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dof", type=int, required=True)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--s-max", type=float, default=2.0)
    ap.add_argument("--init-scale", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--inner-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--init-power", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--stop-after-round", type=int, default=0)
    a = ap.parse_args()

    from opine_experiments.channel2d.dataset_joint import (
        ChannelFlowYZJointDatasetLocal)
    from opine_experiments.channel2d.train_latent_rank import (
        load_joint_preprocessing)
    from opine_experiments.channel2d.iresnet_fine_yz import (
        principal_angles, retained_energy)
    from opine_experiments.channel2d.coupling.coupling_fine_yz import (
        CouplingFINE_YZ)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)
    tag = f"couplingyz_D{a.dof}"
    revision = code_revision()

    mp, ps, _, _, _ = load_joint_preprocessing(a.preprocessing_file)
    ds = lambda s: ChannelFlowYZJointDatasetLocal(
        os.path.join(a.extracted_dir, f"{s}.npz"), mp, ps, NY)
    tr, va, te = ds("train"), ds("val"), ds("test")
    train_ld = torch.utils.data.DataLoader(tr, batch_size=a.batch_size,
                                           shuffle=True)
    train_eval_ld = torch.utils.data.DataLoader(tr, batch_size=64)
    val_ld = torch.utils.data.DataLoader(va, batch_size=64)
    test_ld = torch.utils.data.DataLoader(te, batch_size=64)

    model = CouplingFINE_YZ(NY, NZ, a.dof, channels=NCOMP,
                            n_blocks=a.n_blocks, hidden=a.hidden,
                            pad_mode="circular_y_zero_z",
                            init_scale=a.init_scale, s_max=a.s_max,
                            bottleneck="banded").to(dev)
    n_par = model.n_trainable()
    probe = torch.stack([te[i] for i in range(8)]).to(dev)

    # ── pre-flight: invertibility and the equivariance the band structure needs
    rt0 = model.f.roundtrip_errors(probe)
    eq0 = model.f.equivariance_error(probe)
    with torch.no_grad():
        init_dev = float((model.f(probe) - probe).norm() / probe.norm())
    if rt0["verified_float64"] > 1e-6:
        raise RuntimeError(f"init roundtrip {rt0['verified_float64']:.2e}")
    if eq0 > 1e-4:
        raise RuntimeError(
            f"f_theta is not y-translation equivariant ({eq0:.2e}); the "
            "banded projector assumes it is")
    print(f"[{tag}] params {n_par:,}  rt64 {rt0['verified_float64']:.2e}  "
          f"y-equivariance {eq0:.2e}  |f-x|/|x| {init_dev:.3e}", flush=True)

    model.init_subspace(train_eval_ld, dev, method="sympod",
                        n_power=a.init_power, seed=a.seed)
    Q_init = model.Q.clone()

    def ev(ld):
        model.eval(); num = den = 0.0
        with torch.no_grad():
            for xb in ld:
                xb = xb.to(dev)
                r = model(xb)
                num += float((r - xb).double().pow(2).sum())
                den += float(xb.double().pow(2).sum())
        return num / den

    round0_test = ev(test_ld)
    print(f"[{tag}] round-0 test NMSE {round0_test:.8f}  "
          f"(= the canonical POD baseline up to the near-identity init)",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.rounds)
    hist, best_val, best_round, best_state = [], float("inf"), 0, None
    t0, gstep, start = time.time(), 0, 1
    state_path = os.path.join(a.output_dir, f"{tag}_state.pt")
    if a.resume and os.path.isfile(state_path):
        st = torch.load(state_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); hist = st["history"]
        best_val, best_round = st["best_val"], st["best_round"]
        best_state = st["best_state"]; Q_init = st["Q_init"].to(dev)
        gstep, start = st["gstep"], st["round"] + 1
        torch.set_rng_state(st["rng_torch"].cpu())
        if st["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in st["rng_cuda"]])
        np.random.set_state(st["rng_numpy"])
        print(f"[{tag}] resumed from round {st['round']}", flush=True)

    def checkpoint(rnd):
        torch.save({"tag": tag, "round": rnd, "gstep": gstep,
                    "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "history": hist,
                    "best_val": best_val, "best_round": best_round,
                    "best_state": best_state, "Q_init": Q_init.cpu(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
                    "rng_numpy": np.random.get_state(),
                    "code_revision": revision}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

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
                run += float(loss); nb += 1; gstep += 1
        sched.step()
        sub = model.update_subspace(train_eval_ld, dev, alpha=a.alpha)
        rt = model.f.roundtrip_errors(probe)
        eq = model.f.equivariance_error(probe)
        with torch.no_grad():
            dev_id = float((model.f(probe) - probe).norm() / probe.norm())
        v, t = ev(val_ld), ev(test_ld)
        if v < best_val:
            best_val, best_round = v, rnd
            best_state = {k: x.detach().clone()
                          for k, x in model.state_dict().items()}
        dq = float(principal_angles(Q_init, model.Q).max()) * DEG
        hist.append({"round": rnd, "opt_step": gstep, "train_mse": run / nb,
                     "val_nmse": v, "test_nmse": t,
                     "roundtrip_float64": rt["verified_float64"],
                     "y_equivariance_error": eq,
                     "f_deviation_from_identity_rel": dev_id,
                     "drift_max_angle_deg": dq,
                     "retained_energy_after": sub["retained_energy_after"],
                     "projector_equivariance_error":
                         sub.get("projector_equivariance_error")})
        print(f"[{tag}] R{rnd:3d}/{a.rounds} train {run/nb:.6f}  "
              f"val {v:.6f}  test {t:.6f}  "
              f"({100*(round0_test-t)/round0_test:+.2f}% vs round-0)  "
              f"dQ {dq:5.2f}deg  rt {rt['verified_float64']:.1e}  "
              f"eq {eq:.1e}  dev {dev_id:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)
        if rt["verified_float64"] > 1e-6:
            raise RuntimeError(f"round {rnd}: roundtrip "
                               f"{rt['verified_float64']:.2e}")
        checkpoint(rnd)
        if a.stop_after_round and rnd >= a.stop_after_round:
            print(f"[{tag}] stopping after round {rnd} (test hook)", flush=True)
            return

    model.load_state_dict(best_state)
    sel_test, sel_train = ev(test_ld), ev(train_eval_ld)
    with torch.no_grad():
        Z, _ = model.latent_matrix(train_eval_ld, dev)
        ret = retained_energy(Z, model.Q); del Z
        x0 = torch.stack([te[i] for i in range(4)]).to(dev)
        r0 = model(x0)
    np.savez_compressed(os.path.join(a.output_dir, f"{tag}_samples.npz"),
                        truth=x0.cpu().numpy(), recon=r0.cpu().numpy())
    res = {"method": "Coupling-FINE (channel y-z, banded bottleneck)",
           "tag": tag, "dof": a.dof, "canonical": False,
           "n_trainable_params": n_par, "alpha": a.alpha,
           "_metric": "sum|recon-truth|^2/sum|truth|^2, test split",
           "_selection": "validation NMSE only",
           "round0_test_nmse": round0_test,
           "init_f_deviation_from_identity": init_dev,
           "init_y_equivariance_error": eq0,
           "best_round": best_round, "best_val_nmse": best_val,
           "selected_test_nmse": sel_test, "selected_train_nmse": sel_train,
           "generalization_gap": sel_test - sel_train,
           "gain_vs_round0_pct": 100.0 * (round0_test - sel_test) / round0_test,
           "final_roundtrip_float64": model.f.roundtrip_errors(
               probe)["verified_float64"],
           "final_y_equivariance_error": model.f.equivariance_error(probe),
           "retained_energy_final": ret,
           "final_drift_max_angle_deg": hist[-1]["drift_max_angle_deg"],
           "runtime_s": time.time() - t0, "code_revision": revision,
           "config": vars(a), "history": hist}
    json.dump(res, open(os.path.join(a.output_dir, f"{tag}_results.json"), "w"),
              indent=2, default=float)
    torch.save({"best_state": best_state, "Q_init": Q_init.cpu(),
                "code_revision": revision},
               os.path.join(a.output_dir, f"{tag}_best.pt"))
    print(f"\n[{tag}] round-0 {round0_test:.6f} -> {sel_test:.6f}  "
          f"({res['gain_vs_round0_pct']:+.2f}%)  round {best_round}  "
          f"params {n_par:,}", flush=True)


if __name__ == "__main__":
    main()
