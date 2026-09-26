#!/usr/bin/env python3
"""Coupling-FINE trainer.  SEPARATE from the Q-timescale ablation.

Everything except f_theta matches the canonical setting: same dataset, same
normalization, same POD initialization of Q, same canonical subspace update at
alpha = 0.03 once per round, same near-identity initialization convention,
same reconstruction loss, same validation-only checkpoint selection, same test
metric.  Q-timescale is NOT a variable here -- alpha is pinned.

Outputs live under a coupling_* directory and every file is prefixed
`coupling_`, so nothing can be confused with the q_timescale_* results.
"""
import argparse, hashlib, json, math, os, subprocess, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

GRID, DEG = 128, 180.0 / math.pi


def code_revision():
    rev, here = {}, os.path.dirname(os.path.abspath(__file__))
    for name, rel in (
            ("flow", os.path.join(here, "coupling_flow.py")),
            ("trainer", os.path.join(here, "train_coupling_fine.py")),
            ("model", os.path.join(ROOT, "experiments", "channel_retau180_yz",
                                   "iresnet_fine_yz.py"))):
        try:
            rev[f"sha256_{name}"] = hashlib.sha256(
                open(rel, "rb").read()).hexdigest()
        except OSError as e:
            rev[f"sha256_{name}"] = f"unavailable: {e}"
    for key, cmd in (("git_head", ["git", "rev-parse", "HEAD"]),
                     ("git_status", ["git", "status", "--porcelain"])):
        try:
            rev[key] = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                                      text=True, timeout=20).stdout.strip()
        except Exception as e:
            rev[key] = f"unavailable: {e}"
    return rev


def drift(Q, Q0):
    from opine_experiments.channel2d.iresnet_fine_yz import principal_angles
    with torch.no_grad():
        th = principal_angles(Q0, Q)
        return {"drift_max_angle_deg": float(th.max()) * DEG,
                "drift_mean_angle_deg": float(th.mean()) * DEG,
                "drift_rms_angle_deg": float(th.pow(2).mean().sqrt()) * DEG,
                "drift_projector_dist": float(torch.sin(th.max()))}


def nmse_of(model, loader, device):
    model.eval(); num = den = 0.0
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            r = model(xb)
            num += float((r - xb).double().pow(2).sum())
            den += float(xb.double().pow(2).sum())
    return num / den


def main():
    ap = argparse.ArgumentParser(description="Coupling-FINE (Glow/RealNVP "
                                             "f_theta, POD bottleneck).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dof", type=int, required=True)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--s-max", type=float, default=2.0)
    ap.add_argument("--init-scale", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.03,
                    help="canonical subspace update rate; pinned, not swept")
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--stop-after-round", type=int, default=0)
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--pod-basis", default=None,
                    help="override the POD basis .npz; the canonical one is "
                         "truncated at rank 256, so a run above that rank "
                         "must point at an extended basis")
    a = ap.parse_args()

    from opine_experiments.kolmogorov.canonical.config import FROZEN, dataset_paths
    from opine_experiments.kolmogorov.canonical.dataset import make_datasets
    from opine_experiments.kolmogorov.coupling.coupling_flow import CouplingFINE
    from opine_experiments.channel2d.iresnet_fine_yz import retained_energy
    for k, v in FROZEN.items():
        if k not in ("alpha", "hidden"):        # both are owned by the CLI here
            setattr(a, k, v)
    paths = dataset_paths(a.dataset)
    revision = code_revision()
    device = torch.device(a.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)
    tag = f"coupling_D{a.dof}"

    tr, va, te, sigma = make_datasets(paths["raw"], paths["splits"])
    mk = lambda ds, sh, bs: torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=sh, num_workers=a.num_workers,
        pin_memory=True)
    train_ld = mk(tr, True, a.batch_size)
    val_ld, test_ld = mk(va, False, 128), mk(te, False, 128)
    rng = np.random.default_rng(a.seed)
    sub_idx = np.sort(rng.choice(len(tr), size=min(a.latent_subsample, len(tr)),
                                 replace=False))
    sub_ld = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr, sub_idx.tolist()), batch_size=256,
        shuffle=False, num_workers=a.num_workers)

    basis_path = a.pod_basis or paths["pod_basis"]
    pb = np.load(basis_path)
    # The stored basis is truncated at whatever rank it was written with, and
    # numpy slicing past the end returns FEWER columns without complaining --
    # which silently turns a --dof 512 run into a 256-dimensional one.  Fail
    # loudly instead.
    if pb["modes"].shape[1] < a.dof:
        raise SystemExit(
            f"POD basis {basis_path} holds only {pb['modes'].shape[1]} modes "
            f"but --dof {a.dof} was requested; rerun canonical/run_pod.py "
            f"with --dofs including {a.dof} and pass --pod-basis")
    Q0 = torch.from_numpy(pb["modes"].astype(np.float64)[:, :a.dof]) \
              .float().to(device).contiguous()
    assert Q0.shape[1] == a.dof, (Q0.shape, a.dof)
    pod_mean = torch.from_numpy(pb["mean"].astype(np.float64)).float().to(device)

    model = CouplingFINE(GRID, GRID, a.dof, channels=1, n_blocks=a.n_blocks,
                         hidden=a.hidden, pad_mode="circular_both",
                         init_scale=a.init_scale, s_max=a.s_max,
                         bottleneck="unrestricted").to(device)
    model.Q, model.z_mean = Q0.clone(), pod_mean.clone()
    model.q_initialized.fill_(1.0)
    n_params = model.n_trainable()
    assert model.f._certificate_applicable is False

    probe = torch.stack([te[i] for i in range(8)]).to(device)
    rt0 = model.f.roundtrip_errors(probe)
    with torch.no_grad():
        init_dev = float((model.f(probe) - probe).norm() / probe.norm())

    # POD baseline, computed directly rather than read off round 0.
    pod_flat = torch.stack([te[i] for i in range(len(te))]).reshape(len(te), -1)
    with torch.no_grad():
        num = den = 0.0
        for i in range(0, pod_flat.shape[0], 256):
            xb = pod_flat[i:i + 256].to(device)
            rec = ((xb - pod_mean) @ Q0) @ Q0.T + pod_mean
            num += float((rec - xb).double().pow(2).sum())
            den += float(xb.double().pow(2).sum())
        pod_test = num / den
    del pod_flat
    round0_test = nmse_of(model, test_ld, device)
    print(f"[{tag}] params {n_params:,}  POD test {pod_test:.8f}  "
          f"round-0 {round0_test:.8f}  |f-x|/|x| {init_dev:.3e}  "
          f"rt64 {rt0['verified_float64']:.2e}", flush=True)

    Q_init = model.Q.clone()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.rounds)
    history, best_val, best_round, best_state = [], float("inf"), 0, None
    t0, gstep, start_round = time.time(), 0, 1
    state_path = os.path.join(a.output_dir, f"{tag}_state.pt")
    if a.resume and os.path.isfile(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); history = st["history"]
        best_val, best_round = st["best_val"], st["best_round"]
        best_state = st["best_state"]; Q_init = st["Q_init"].to(device)
        gstep, start_round = st["gstep"], st["round"] + 1
        torch.set_rng_state(st["rng_torch"].cpu())
        if st["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in st["rng_cuda"]])
        np.random.set_state(st["rng_numpy"])
        print(f"[{tag}] resumed from round {st['round']}", flush=True)

    def checkpoint(rnd):
        torch.save({"tag": tag, "round": rnd, "gstep": gstep,
                    "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "history": history,
                    "best_val": best_val, "best_round": best_round,
                    "best_state": best_state, "Q_init": Q_init.cpu(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
                    "rng_numpy": np.random.get_state(),
                    "code_revision": revision}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    for rnd in range(start_round, a.rounds + 1):
        model.train(); run_loss, nb = 0.0, 0
        for xb in train_ld:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = (model(xb) - xb).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            run_loss += float(loss); nb += 1; gstep += 1
            if a.max_train_batches and nb >= a.max_train_batches:
                break
        sched.step()
        sub = model.update_subspace(sub_ld, device, alpha=a.alpha,
                                    n_power=a.subspace_power)
        with torch.no_grad():
            rt = model.f.roundtrip_errors(probe)
            dev_id = float((model.f(probe) - probe).norm() / probe.norm())
        v, t = nmse_of(model, val_ld, device), nmse_of(model, test_ld, device)
        if v < best_val:
            best_val, best_round = v, rnd
            best_state = {k: x.detach().clone()
                          for k, x in model.state_dict().items()}
        row = {"round": rnd, "opt_step": gstep, "train_mse": run_loss / nb,
               "val_nmse": v, "test_nmse": t,
               "test_gain_vs_pod_pct": 100.0 * (pod_test - t) / pod_test,
               "retained_energy_after": sub["retained_energy_after"],
               "retained_energy_before": sub["retained_energy_before"],
               "subspace_step_angle_deg": sub["max_principal_angle"] * DEG,
               "roundtrip_verified_float64": rt["verified_float64"],
               "roundtrip_train_path_float32": rt["train_path_float32"],
               "f_deviation_from_identity_rel": dev_id,
               **drift(model.Q, Q_init)}
        history.append(row)
        print(f"[{tag}] R{rnd:3d}/{a.rounds} step {gstep:6d}  "
              f"train {row['train_mse']:.6f}  val {v:.6f}  test {t:.6f}  "
              f"({row['test_gain_vs_pod_pct']:+.2f}% vs POD)  "
              f"dQ {row['drift_max_angle_deg']:6.2f}deg  "
              f"rt64 {rt['verified_float64']:.1e}  dev {dev_id:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        # The flow is invertible by construction, so there is no contraction
        # certificate to guard.  The round trip is the real invariant.
        if rt["verified_float64"] > 1e-6:
            raise RuntimeError(f"round {rnd}: float64 roundtrip "
                               f"{rt['verified_float64']:.2e}")
        checkpoint(rnd)
        if a.stop_after_round and rnd >= a.stop_after_round:
            print(f"[{tag}] stopping after round {rnd} (test hook)", flush=True)
            return

    model.load_state_dict(best_state)
    sel_test = nmse_of(model, test_ld, device)
    sel_train = nmse_of(model, train_ld, device)
    Q_sel = model.Q.clone()
    with torch.no_grad():
        Z_lat, _ = model.latent_matrix(sub_ld, device)
        ret_latent = retained_energy(Z_lat, Q_sel); del Z_lat
        raw = torch.stack([tr[int(i)] for i in sub_idx]).to(device)
        raw = raw.reshape(raw.shape[0], -1) - pod_mean.unsqueeze(0)
        ret_raw_Qsel = retained_energy(raw, Q_sel)
        ret_raw_Q0 = retained_energy(raw, Q_init); del raw
        rt_final = model.f.roundtrip_errors(probe)
        x0 = torch.stack([te[i] for i in range(4)]).to(device)
        r0 = model(x0)
    np.savez_compressed(os.path.join(a.output_dir, f"{tag}_samples.npz"),
                        truth=x0[:, 0].cpu().numpy(),
                        recon=r0[:, 0].cpu().numpy(),
                        error=(r0 - x0)[:, 0].cpu().numpy())

    ofine = os.path.join(ROOT, "experiments", "kolmogorov", "results",
                         "canonical", f"OriginalFINE_D{a.dof}_results.json")
    orig = None
    if os.path.isfile(ofine):
        try:
            o = json.load(open(ofine))
            orig = o.get("selected_test_nmse", o.get("test_nmse"))
        except Exception:
            orig = None

    res = {"method": "Coupling-FINE (Glow/RealNVP f_theta, POD bottleneck)",
           "tag": tag, "dof": a.dof, "canonical": False,
           "n_trainable_params": n_params,
           "_metric": "sum|recon-truth|^2/sum|truth|^2, test split",
           "_selection": "validation NMSE only",
           "alpha": a.alpha, "alpha_is_swept": False,
           "pod_test_nmse": pod_test, "round0_test_nmse": round0_test,
           "round0_minus_pod": round0_test - pod_test,
           "init_f_deviation_from_identity": init_dev,
           "best_round": best_round, "best_val_nmse": best_val,
           "selected_test_nmse": sel_test,
           "selected_train_nmse": sel_train,
           "generalization_gap": sel_test - sel_train,
           "final_test_nmse": history[-1]["test_nmse"],
           "gain_vs_pod_pct": 100.0 * (pod_test - sel_test) / pod_test,
           "original_fine_test_nmse": orig,
           "gain_vs_original_fine_pct":
               (None if orig is None
                else 100.0 * (orig - sel_test) / orig),
           "inverse_roundtrip_float64": rt_final["verified_float64"],
           "inverse_roundtrip_float32": rt_final["train_path_float32"],
           "retained_energy_raw_Q0": ret_raw_Q0,
           "retained_energy_raw_Qsel": ret_raw_Qsel,
           "retained_energy_latent_Qsel": ret_latent,
           "final_drift": drift(Q_sel, Q_init),
           "runtime_s": time.time() - t0, "code_revision": revision,
           "config": {k: getattr(a, k) for k in
                      ("dof", "n_blocks", "hidden", "s_max", "init_scale",
                       "alpha", "rounds", "lr", "batch_size", "seed",
                       "dataset")},
           "history": history}
    json.dump(res, open(os.path.join(a.output_dir, f"{tag}_results.json"), "w"),
              indent=2, default=float)
    torch.save({"best_state": best_state, "Q_selected": Q_sel.cpu(),
                "Q_init": Q_init.cpu(), "code_revision": revision},
               os.path.join(a.output_dir, f"{tag}_best.pt"))
    print(f"\n[{tag}] POD           {pod_test:.8f}")
    print(f"[{tag}] Coupling-FINE {sel_test:.8f}  "
          f"({res['gain_vs_pod_pct']:+.2f}% vs POD)  round {best_round}")
    if orig is not None:
        print(f"[{tag}] Original FINE {orig:.8f}  "
              f"({res['gain_vs_original_fine_pct']:+.2f}% gain)")
    print(f"[{tag}] params {n_params:,}  rt64 "
          f"{rt_final['verified_float64']:.2e}  saved -> {a.output_dir}",
          flush=True)


if __name__ == "__main__":
    main()
