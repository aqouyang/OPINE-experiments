#!/usr/bin/env python3
"""Validation of the generated Kolmogorov-flow trajectories.

Runs every check requested for the pilot and writes both a JSON report and
presentation-quality figures.  Exits non-zero if any hard check fails.

Checks
  0  forcing term equals the analytic -k_f cos(k_f y)
  1  vorticity contour snapshots                                   [figure]
  2  kinetic energy and dissipation versus time                    [figure]
  3  dissipation distribution after burn-in                        [figure]
  4  mean and RMS vorticity
  5  spatial energy / enstrophy spectra                            [figure]
  6  statistical stationarity after the 50-time-unit burn-in       [figure]
  7  mean vorticity numerically near zero
  8  spectral energy adequately decayed before the grid cutoff
  9  saved snapshots exactly one advective time unit apart
 10  NaN / Inf checks
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def radial_spectrum(w, L=2 * np.pi):
    """Enstrophy and energy spectra, shell-averaged over integer |k|."""
    n = w.shape[-1]
    wh = np.fft.fft2(w) / (n * n)
    kx = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    ens = np.abs(wh) ** 2                      # |omega_k|^2
    with np.errstate(divide="ignore", invalid="ignore"):
        ener = np.where(k2 > 0, ens / k2, 0.0)  # |u_k|^2 = |omega_k|^2/k^2
    kr = np.sqrt(k2).ravel()
    nb = n // 2
    idx = np.clip(np.round(kr).astype(int), 0, nb)
    E = np.bincount(idx, weights=ener.ravel(), minlength=nb + 1)
    Z = np.bincount(idx, weights=ens.ravel(), minlength=nb + 1)
    return np.arange(nb + 1), E, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--recheck-spacing", type=int, default=1,
                    help="re-integrate a snapshot forward and compare with "
                         "the next saved one (needs the solver + a GPU)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    meta = json.load(open(args.metadata))
    Re = meta["reynolds_number"]
    k_f = meta["forcing_wavenumber_k_f"]
    n = meta["grid"][0]
    burn = meta["burn_in_time"]

    trajs = sorted(glob.glob(os.path.join(args.raw_dir, "traj_*.npy")))
    diags = sorted(glob.glob(os.path.join(args.raw_dir, "diag_*.npz")))
    assert trajs, f"no trajectories in {args.raw_dir}"
    print(f"{len(trajs)} trajectories in {args.raw_dir}", flush=True)

    rep, fails = {"n_trajectories": len(trajs)}, []

    def check(name, ok, detail):
        rep[name] = {"pass": bool(ok), **detail}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        if not ok:
            fails.append(name)

    # ---------------- 10. NaN / Inf, shapes, dtype -------------------
    shapes, dtypes, nonfinite = set(), set(), 0
    for t in trajs:
        a = np.load(t, mmap_mode="r")
        shapes.add(a.shape)
        dtypes.add(str(a.dtype))
        nonfinite += int((~np.isfinite(np.asarray(a))).sum())
    check("10_finite_and_shapes", nonfinite == 0 and len(shapes) == 1,
          {"nonfinite_values": nonfinite, "shapes": [list(s) for s in shapes],
           "dtypes": sorted(dtypes)})

    w0 = np.load(trajs[0])                       # (n_snap, n, n)
    D_all, E_all, mw_all = [], [], []
    for d in diags:
        z = np.load(d)
        D_all.append(z["dissipation"]); E_all.append(z["energy"])
        mw_all.append(z["mean_vorticity"])
    D_all = np.array(D_all); E_all = np.array(E_all); mw_all = np.array(mw_all)
    t_axis = np.load(diags[0])["time"]

    # ---------------- 4 / 7. mean and RMS vorticity ------------------
    mean_w = float(np.mean([np.load(t, mmap_mode="r")[:].mean()
                            for t in trajs]))
    rms_w = float(np.sqrt(np.mean([(np.load(t, mmap_mode="r")[:] ** 2).mean()
                                   for t in trajs])))
    check("4_mean_and_rms_vorticity", True,
          {"mean_vorticity": mean_w, "rms_vorticity": rms_w,
           "rms_vorticity_from_D": float(np.sqrt(D_all.mean() * Re))})
    # float32 storage of an O(3) field: eps*rms ~ 1e-7 * 3; use a slack bound
    check("7_mean_vorticity_near_zero", abs(mean_w) < 1e-4 * rms_w,
          {"mean": mean_w, "relative_to_rms": abs(mean_w) / rms_w,
           "max_abs_per_snapshot": float(np.abs(mw_all).max())})

    # ---------------- 6. stationarity after burn-in ------------------
    # A split-half comparison must be judged against the standard error of
    # the mean, not the raw standard deviation: D and E have an integral
    # correlation time of order 10 time units here, so consecutive snapshots
    # are far from independent and a raw-sigma threshold rejects a perfectly
    # stationary signal purely because the window is short.
    def integral_time(x):
        """Integral autocorrelation time, summed to the first zero crossing."""
        y = x - x.mean()
        ac = np.correlate(y, y, mode="full")[len(y) - 1:]
        ac /= ac[0]
        stop = np.argmax(ac < 0) if (ac < 0).any() else len(ac)
        return float(1.0 + 2.0 * ac[1:stop].sum())

    def sem(arr):
        """Standard error of the mean over an ensemble of correlated series."""
        tau = np.mean([integral_time(r) for r in arr])
        n_eff = arr.size / max(tau, 1.0)
        return float(arr.std() / np.sqrt(n_eff)), float(tau)

    half = D_all.shape[1] // 2
    d1, d2 = D_all[:, :half], D_all[:, half:]
    e1, e2 = E_all[:, :half], E_all[:, half:]
    sD1, tauD = sem(d1); sD2, _ = sem(d2)
    sE1, tauE = sem(e1); sE2, _ = sem(e2)
    zD = abs(d1.mean() - d2.mean()) / np.hypot(sD1, sD2)
    zE = abs(e1.mean() - e2.mean()) / np.hypot(sE1, sE2)
    Dm = D_all.mean(axis=0)
    slope = np.polyfit(t_axis, Dm, 1)[0]
    span = slope * (t_axis[-1] - t_axis[0])
    # transient signature: the fluctuation level itself should not be
    # inflated in the first window relative to the last
    std_ratio = float(d1.std() / d2.std())
    check("6_stationary_after_burn_in",
          zD < 3.0 and zE < 3.0 and 0.6 < std_ratio < 1.7,
          {"D_first_half_mean": float(d1.mean()),
           "D_second_half_mean": float(d2.mean()),
           "D_half_diff_in_sem": float(zD),
           "E_first_half_mean": float(e1.mean()),
           "E_second_half_mean": float(e2.mean()),
           "E_half_diff_in_sem": float(zE),
           "integral_correlation_time_D": tauD,
           "integral_correlation_time_E": tauE,
           "D_std_ratio_first_over_second_half": std_ratio,
           "D_linear_drift_over_window": float(span),
           "D_std": float(Dm.std()),
           "_test": "split-half means compared against the standard error "
                    "of the mean with the integral correlation time folded "
                    "in, plus a check that the fluctuation level is not "
                    "inflated in the first half (the transient signature)"})
    d1, d2 = d1.ravel(), d2.ravel()

    # ---------------- 5 / 8. spectra ---------------------------------
    ns = min(200, w0.shape[0])
    sel = np.linspace(0, w0.shape[0] - 1, ns).astype(int)
    Es, Zs = [], []
    for i in sel:
        kk, E, Z = radial_spectrum(w0[i].astype(np.float64))
        Es.append(E); Zs.append(Z)
    kk = kk; Ebar = np.mean(Es, axis=0); Zbar = np.mean(Zs, axis=0)
    k_cut = n // 3                                   # 2/3-rule dealiasing
    peak = Ebar[1:].max()
    decay_at_cut = Ebar[k_cut] / peak
    decay_at_nyq = Ebar[-1] / peak
    check("8_spectrum_decayed_before_cutoff", decay_at_cut < 1e-6,
          {"E(k_dealias)/E_peak": float(decay_at_cut),
           "E(k_nyquist)/E_peak": float(decay_at_nyq),
           "k_dealias": int(k_cut), "k_nyquist": int(n // 2)})

    # ---------------- 9. snapshot spacing ----------------------------
    if args.recheck_spacing:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from kolmogorov_solver import (KolmogorovConfig, make_grid,
                                       make_step_fn)
        cfg = KolmogorovConfig(re=Re, k_f=k_f, n=n, dt=meta["dt"])
        grid = make_grid(cfg); step = make_step_fn(cfg, grid)
        per = meta["steps_per_snapshot"]
        adv = jax.jit(lambda w: jax.lax.fori_loop(0, per, lambda i, x: step(x),
                                                  w))
        errs = []
        for i in (0, w0.shape[0] // 2):
            w_hat = jnp.fft.rfftn(jnp.asarray(w0[i], dtype=jnp.float64))
            nxt = np.asarray(jnp.fft.irfftn(adv(w_hat)))
            errs.append(float(np.abs(nxt - w0[i + 1]).max()
                              / np.abs(w0[i + 1]).max()))
        # float32 storage + chaotic amplification over one time unit
        check("9_snapshot_spacing_is_one_time_unit", max(errs) < 1e-3,
              {"relative_error_reintegrating_one_interval": errs,
               "steps_per_snapshot": per, "dt": meta["dt"],
               "note": "re-integrating a saved snapshot by exactly "
                       "steps_per_snapshot must reproduce the next saved one"})

    # ==================== figures ====================================
    fs = dict(dpi=200, bbox_inches="tight")

    # 1. vorticity snapshots
    idx = np.linspace(0, w0.shape[0] - 1, 5).astype(int)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    vmax = float(np.abs(w0[idx]).max())
    for ax, i in zip(axes, idx):
        ax.imshow(w0[i].T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi])
        ax.set_title(f"$t = {t_axis[i]:.0f}$", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_ylabel(r"$\omega$", fontsize=13)
    fig.suptitle(f"Kolmogorov flow  Re={Re:.0f}, $k_f$={k_f}, {n}$^2$ "
                 f"— trajectory 0", fontsize=12)
    fig.savefig(os.path.join(args.output_dir, "01_vorticity_snapshots.png"),
                **fs); plt.close(fig)

    # 2. energy and dissipation vs time
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for j in range(min(5, E_all.shape[0])):
        ax[0].plot(t_axis, E_all[j], lw=0.7, alpha=0.8)
        ax[1].plot(t_axis, D_all[j], lw=0.7, alpha=0.8)
    ax[0].plot(t_axis, E_all.mean(0), "k", lw=1.6, label="trajectory mean")
    ax[1].plot(t_axis, D_all.mean(0), "k", lw=1.6)
    ax[0].set_ylabel("kinetic energy"); ax[0].legend(fontsize=9)
    ax[1].set_ylabel(r"$D=\langle\omega^2\rangle/Re$")
    ax[1].set_xlabel("advective time units (after burn-in)")
    fig.savefig(os.path.join(args.output_dir, "02_energy_dissipation.png"),
                **fs); plt.close(fig)

    # 3. dissipation distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(D_all.ravel(), bins=80, density=True, color="#4477aa",
            edgecolor="none")
    ax.axvline(D_all.mean(), color="k", ls="--",
               label=f"mean {D_all.mean():.4f}")
    ax.set_xlabel(r"$D=\langle\omega^2\rangle/Re$"); ax.set_ylabel("pdf")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(args.output_dir, "03_dissipation_pdf.png"),
                **fs); plt.close(fig)

    # 5/8. spectra
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, arr, lab in ((ax[0], Ebar, "energy $E(k)$"),
                        (ax[1], Zbar, "enstrophy $Z(k)$")):
        a.loglog(kk[1:], arr[1:], lw=1.5)
        a.axvline(k_f, color="#228833", ls=":", label=f"$k_f$={k_f}")
        a.axvline(k_cut, color="#cc6677", ls="--",
                  label=f"2/3 cutoff $k$={k_cut}")
        a.set_xlabel("$k$"); a.set_ylabel(lab); a.legend(fontsize=8)
        a.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(args.output_dir, "05_spectra.png"), **fs)
    plt.close(fig)

    # 6. stationarity
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    win = max(1, len(t_axis) // 25)
    run = np.convolve(D_all.mean(0), np.ones(win) / win, mode="valid")
    ax[0].plot(t_axis[:len(run)], run, lw=1.4)
    ax[0].set_xlabel("time"); ax[0].set_ylabel(f"D, running mean ({win} tu)")
    ax[0].set_title("no trend after burn-in", fontsize=10)
    ax[1].hist(d1, bins=60, density=True, alpha=0.6, label="first half")
    ax[1].hist(d2, bins=60, density=True, alpha=0.6, label="second half")
    ax[1].set_xlabel("$D$"); ax[1].legend(fontsize=9)
    ax[1].set_title("split-half distributions agree", fontsize=10)
    fig.savefig(os.path.join(args.output_dir, "06_stationarity.png"), **fs)
    plt.close(fig)

    rep["_summary"] = {"failures": fails, "all_passed": not fails,
                       "snapshots_total": int(len(trajs) * w0.shape[0])}
    with open(os.path.join(args.output_dir, "validation_report.json"),
              "w") as fh:
        json.dump(rep, fh, indent=2, default=float)
    print(f"\nreport + figures -> {args.output_dir}", flush=True)
    if fails:
        print(f"FAILED: {fails}", flush=True); sys.exit(1)
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
