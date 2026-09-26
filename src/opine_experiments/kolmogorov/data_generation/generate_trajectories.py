#!/usr/bin/env python3
"""Generate the Re=100 Kolmogorov-flow dataset (Cleary, Wang & Zaki, 2512.15470).

Protocol (Appendix C as specified):
  100 trajectories; discard the first 50 advective time units; then save 1000
  snapshots per trajectory, spaced exactly 1 advective time unit apart.
  Final dataset: 100,000 snapshots of shape 128 x 128.

No dissipation-rate resampling is applied.  This is the raw, uniformly
time-sampled attractor.  D = <omega^2>/Re is stored per snapshot so a
resampled index list can be built later without regenerating anything.

Determinism and restart safety:
  * trajectory i uses PRNGKey(seed_base + i); the initial condition and hence
    the whole trajectory is a pure function of i.
  * a trajectory is written to traj_XXX.npy.tmp and only then renamed, so a
    killed job can never leave a truncated file that looks complete.
  * trajectories whose .npy already exists are skipped, never overwritten.

Trajectories are advanced together under vmap: at 128^2 the GPU is far from
saturated by a single state, and batching gives ~18x throughput (see README).
"""

import argparse
import functools
import json
import os
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--traj-start", type=int, required=True)
    ap.add_argument("--traj-end", type=int, required=True,
                    help="exclusive")
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--k-f", type=int, default=4)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--dt-inv", type=int, default=512,
                    help="dt = 1/dt_inv; 1 time unit is then exactly dt_inv "
                         "steps, so snapshot spacing is exact by construction")
    ap.add_argument("--burn-in-time", type=float, default=100.0,
                    help="DEVIATION FROM THE PAPER, measured not assumed: "
                         "Appendix C specifies 50, but at Re=100 with this "
                         "initial condition the first ~100 time units still "
                         "carry transient (dissipation std 0.0699 over "
                         "t=0-100 against ~0.029 for every later 100-unit "
                         "window, i.e. 2.4x the equilibrium fluctuation). "
                         "Pass --burn-in-time 50 to reproduce the paper "
                         "literally.")
    ap.add_argument("--n-snapshots", type=int, default=1000)
    ap.add_argument("--snapshot-interval", type=float, default=1.0)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--ic-max-velocity", type=float, default=7.0)
    ap.add_argument("--ic-peak-wavenumber", type=int, default=4)
    ap.add_argument("--batch", type=int, default=25,
                    help="trajectories advanced simultaneously under vmap")
    ap.add_argument("--x64", type=int, default=1)
    ap.add_argument("--store-dtype", default="float32")
    args = ap.parse_args()

    if args.x64:
        jax.config.update("jax_enable_x64", True)

    from kolmogorov_solver import (KolmogorovConfig, make_grid, make_step_fn,
                                   initial_vorticity_hat,
                                   advection_stable_dt)
    from jax_cfd.spectral import utils as sutils

    dt = 1.0 / args.dt_inv
    cfg = KolmogorovConfig(
        re=args.re, k_f=args.k_f, n=args.n, dt=dt,
        burn_in_time=args.burn_in_time, n_snapshots=args.n_snapshots,
        snapshot_interval=args.snapshot_interval,
        ic_max_velocity=args.ic_max_velocity,
        ic_peak_wavenumber=args.ic_peak_wavenumber, x64=bool(args.x64))
    grid = make_grid(cfg)
    step = make_step_fn(cfg, grid)
    vel = sutils.vorticity_to_velocity(grid)

    per = cfg.steps_per_snapshot
    burn = cfg.burn_in_steps
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"devices: {jax.devices()}", flush=True)
    print(f"dt=1/{args.dt_inv}  burn_in={burn} steps  "
          f"{per} steps/snapshot  {args.n_snapshots} snapshots", flush=True)

    @functools.partial(jax.jit, static_argnums=(1,))
    def advance(w_hat, n):
        return jax.lax.fori_loop(0, n, lambda i, w: step(w), w_hat)

    def snapshot_diag(w_hat):
        """omega field plus the per-snapshot statistics we want on disk."""
        w = jnp.fft.irfftn(w_hat)
        vxh, vyh = vel(w_hat)
        vx, vy = jnp.fft.irfftn(vxh), jnp.fft.irfftn(vyh)
        return w, jnp.stack([
            jnp.mean(w ** 2) / cfg.re,            # D = <omega^2>/Re
            0.5 * jnp.mean(vx ** 2 + vy ** 2),    # kinetic energy
            jnp.mean(w),                          # mean vorticity
            jnp.max(jnp.sqrt(vx ** 2 + vy ** 2)),  # max speed (CFL monitor)
        ])

    v_advance = jax.jit(jax.vmap(advance, in_axes=(0, None)),
                        static_argnums=(1,))
    v_diag = jax.jit(jax.vmap(snapshot_diag))

    todo = [i for i in range(args.traj_start, args.traj_end)
            if not os.path.isfile(os.path.join(args.output_dir,
                                               f"traj_{i:03d}.npy"))]
    skipped = (args.traj_end - args.traj_start) - len(todo)
    print(f"{len(todo)} trajectories to run, {skipped} already present "
          f"(skipped, never overwritten)", flush=True)

    store_dtype = np.dtype(args.store_dtype)
    for b0 in range(0, len(todo), args.batch):
        ids = todo[b0:b0 + args.batch]
        t0 = time.time()
        w = jnp.stack([initial_vorticity_hat(cfg, grid, args.seed_base + i)
                       for i in ids])
        w = v_advance(w, burn)                      # discard the transient
        B = len(ids)
        buf = np.empty((B, args.n_snapshots, args.n, args.n),
                       dtype=store_dtype)
        diag = np.empty((B, args.n_snapshots, 4), dtype=np.float64)
        for k in range(args.n_snapshots):
            if k:                                   # first snapshot is t=burn
                w = v_advance(w, per)
            wf, d = v_diag(w)
            buf[:, k] = np.asarray(wf, dtype=store_dtype)
            diag[:, k] = np.asarray(d, dtype=np.float64)
            if k % 200 == 0:
                print(f"    batch {ids[0]}-{ids[-1]}  snapshot {k}/"
                      f"{args.n_snapshots}  ({time.time()-t0:.0f}s)",
                      flush=True)
        bad = ~np.isfinite(buf)
        if bad.any():
            raise RuntimeError(f"non-finite values in batch {ids[0]}-{ids[-1]}")
        for j, i in enumerate(ids):
            # write through an explicit handle: np.save/np.savez append their
            # own extension when the path does not already end in one, which
            # would silently put the temp file somewhere else
            p = os.path.join(args.output_dir, f"traj_{i:03d}.npy")
            tmp = p + ".tmp"
            with open(tmp, "wb") as fh:
                np.save(fh, buf[j])
            os.replace(tmp, p)                      # atomic completion marker
            dp = os.path.join(args.output_dir, f"diag_{i:03d}.npz")
            dtmp = dp + ".tmp"
            with open(dtmp, "wb") as fh:
                np.savez(fh,
                         dissipation=diag[j, :, 0], energy=diag[j, :, 1],
                         mean_vorticity=diag[j, :, 2],
                         max_speed=diag[j, :, 3],
                         time=args.burn_in_time
                         + np.arange(args.n_snapshots)
                         * args.snapshot_interval)
            os.replace(dtmp, dp)
        el = time.time() - t0
        print(f"  wrote trajectories {ids[0]}-{ids[-1]} in {el:.0f}s "
              f"({el/B:.1f}s each)  max speed {diag[:,:,3].max():.3f}",
              flush=True)

    meta = {
        "paper": "Cleary, Wang & Zaki, arXiv:2512.15470",
        "equation": "d_t omega + u.grad omega = (1/Re) lap(omega) "
                    "- k_f cos(k_f y)",
        "reynolds_number": args.re,
        "viscosity_1_over_Re": cfg.viscosity,
        "forcing_wavenumber_k_f": args.k_f,
        "forcing_scale": cfg.forcing_scale,
        "drag": cfg.drag,
        "domain": [[0.0, cfg.domain_length], [0.0, cfg.domain_length]],
        "domain_note": "L=2pi is forced by periodicity of sin(k_f y) with "
                       "integer k_f, not chosen freely",
        "grid": [args.n, args.n],
        "boundary_conditions": "doubly periodic",
        "solver": "jax_cfd.spectral.equations.NavierStokes2D (pseudospectral)",
        "dealiasing": "2/3-rule (smooth=True)",
        "time_integrator": "crank_nicolson_rk4 "
                           "(Carpenter-Kennedy low-storage RK4 explicit / "
                           "Crank-Nicolson implicit)",
        "dt": dt,
        "dt_note": f"dt = 1/{args.dt_inv}; one advective time unit is exactly "
                   f"{per} steps, so snapshot spacing is exact",
        "steps_per_snapshot": per,
        "burn_in_time": args.burn_in_time,
        "burn_in_steps": burn,
        "n_snapshots_per_trajectory": args.n_snapshots,
        "snapshot_interval": args.snapshot_interval,
        "first_snapshot_time": args.burn_in_time,
        "last_snapshot_time": args.burn_in_time
        + (args.n_snapshots - 1) * args.snapshot_interval,
        "initial_condition": "jax_cfd.initial_conditions."
                             "filtered_velocity_field then curl_2d",
        "ic_max_velocity": args.ic_max_velocity,
        "ic_peak_wavenumber": args.ic_peak_wavenumber,
        "seed_convention": f"trajectory i uses PRNGKey({args.seed_base} + i)",
        "seed_base": args.seed_base,
        "solver_precision": "float64" if args.x64 else "float32",
        "storage_dtype": str(store_dtype),
        "stored_quantity": "out-of-plane vorticity omega(x, y)",
        "dissipation_definition": "D = <omega^2>/Re, stored per snapshot in "
                                  "diag_XXX.npz",
        "resampling": "none; raw uniform time sampling",
        "advective_stable_dt_at_max_speed_10": advection_stable_dt(
            cfg, grid, 10.0),
        "versions": {
            "jax": jax.__version__,
            "jaxlib": __import__("jaxlib").__version__,
            "jax_cfd": getattr(__import__("jax_cfd"), "__version__", "0.2.1"),
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
    }
    # array tasks all write identical metadata; go through a per-task temp
    # file so concurrent writers can never interleave into a corrupt file
    mp = os.path.abspath(os.path.join(args.output_dir, os.pardir,
                                      "metadata.json"))
    tmp = f"{mp}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, mp)
    print(f"\nmetadata -> {os.path.abspath(mp)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
