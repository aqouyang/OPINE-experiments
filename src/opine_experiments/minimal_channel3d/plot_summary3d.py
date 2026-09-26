#!/usr/bin/env python3
"""Single summary figure for the 3-D minimal channel.

Left   retained DOF D vs NMSE, train and test, for every baseline.
Right  representative reconstruction at one D, wall-parallel plane.

Everything is read from stored results; nothing is retrained and no NMSE is
recomputed here.

  left panel   validation/train_test_nmse3d.json, written by
               eval_train_nmse3d.py, which evaluates the stored checkpoints of
               all three methods on train/val/test under the one canonical
               metric.  POD is the exact snapshot POD of the full 16000-
               snapshot training split; OPINE is the opine_full_* column
               (f_theta trained with the 16k subspace); CNN is the
               validation-selected best checkpoint.
  right panel  figures/recon3d_D<D>_snapshot<i>.npz, exported by
               plot_recon3d.py -- the same representative snapshot, chosen at
               the 60th percentile of OPINE-over-POD per-snapshot gain.
"""
import argparse, math, glob, json, os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
from opine_experiments.common.paper_style import (  # noqa: E402
    apply as _apply_style, FS, LW, DASH_TRAIN, NMSE, DIM)
_apply_style()

# Plotting convention: POD and OPINE retain modes in pairs, because the
# covariance of a field that is statistically homogeneous in x and y commutes
# with the translation group and the non-trivial real irreps are
# two-dimensional.  Their retained dimension is therefore reported as the
# number of mode PAIRS, i.e. stored rank / 2.  The CNN latent is a plain
# vector with no such structure, so its dimension is reported as stored.
# NOTE: in this 3-D case the bottleneck is unrestricted, so the pairing is a
# statistical property of the empirical covariance, not enforced by
# construction as it is for the 2-D banded SymPOD.
# (name, colour, pair divisor, draw train curve)
# POD shows test only: it is a linear projection fitted on the training split,
# its train and test curves coincide to within 0.003 at every D, and the two
# lines sat exactly on top of each other.
# Reporting convention: the retained REAL dimension is reported directly
# for every method.  POD and OPINE are indexed by the real rank they
# retain, the CNN by its latent width.  Nothing is divided by two -- the
# earlier D = r/2 mode-pair convention is withdrawn, because the pairing
# is enforced only in the banded y-z bottleneck and is a finite-sample
# property elsewhere.
METHODS = [("POD", "#4a7c59", 1, False), ("CNN", "#c98b2e", 1, True),
           ("OPINE", "#b2404a", 1, True)]
PAIRED = {"POD": 1, "CNN": 1, "OPINE": 1}
ROWS = [("truth", "Reference"), ("pod", "POD"), ("cnn", "CNN"),
        ("opine", "OPINE")]


# candidate tick ladders within a decade, densest first
_LADDERS = ((1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8), (1, 1.5, 2, 3, 5, 7),
            (1, 2, 3, 5), (1, 2, 5), (1,))
_MAX_TICKS = 7


def decimal_log_ticks(lo, hi):
    """Ticks for a log NMSE axis, as plain decimals rather than powers of ten.

    The default log locator labels whole decades only, so a curve spanning a
    little more than one decade ends up with a single labelled tick.  These
    are placed by hand: the densest ladder that still fits under _MAX_TICKS
    across the plotted range wins, so a narrow range gets fine ticks and a
    wide one stays at 1-2-5, and every tick drawn is labelled.
    """
    def gen(ladder):
        e, out = math.floor(math.log10(lo)), []
        while 10.0 ** e <= hi:
            out += [m * 10.0 ** e for m in ladder if lo <= m * 10.0 ** e <= hi]
            e += 1
        return out

    for ladder in _LADDERS:
        t = gen(ladder)
        if len(t) <= _MAX_TICKS:
            # a range too narrow to hold three ticks of any ladder keeps the
            # densest one rather than falling through to a bare decade
            return t if len(t) >= 3 else gen(_LADDERS[0])
    return gen(_LADDERS[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmse-json", required=True)
    ap.add_argument("--recon-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vel-percentile", type=float, default=99.0)
    ap.add_argument("--min-plotted-d", type=int, default=32,
                    help="drop points below this retained dimension; the\n"
                         "methods share a D grid that starts at 16, and a\n"
                         "stored rank evaluated only for the CNN would otherwise\n"
                         "put POD alone at half of it")
    a = ap.parse_args()

    j = json.load(open(a.nmse_json))
    rows = sorted(j["rows"], key=lambda r: r["dof"])
    m = {k: v for k, v in np.load(a.recon_npz, allow_pickle=False).items()}
    Dstar = int(m["dof"])
    iz = int(m["iz"])
    Lx, Ly = float(m["Lx"]), float(m["Ly"])
    nyy, nxx = m["truth"].shape[1], m["truth"].shape[2]
    xc = (np.arange(nxx) + 0.5) * Lx / nxx
    yc = (np.arange(nyy) + 0.5) * Ly / nyy

    # No in-figure title and no block of caption text: everything that would
    # go there lives in the provenance JSON and belongs in the paper caption.
    fig = plt.figure(figsize=(10.4, 3.75))
    outer = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.80, 0.028],
                     wspace=0.24, left=0.088, right=0.912,
                     top=0.925, bottom=0.175)

    # ----------------------------------------------------- left: D vs NMSE
    ax = fig.add_subplot(outer[0, 0])
    xs_all = set()
    for name, col, div, show_train in METHODS:
        have = [r for r in rows if name in r
                and r["dof"] // div >= a.min_plotted_d]
        D = [r["dof"] // div for r in have]
        xs_all.update(D)
        te = [r[name]["test"] for r in have]
        ax.plot(D, te, "o-", color=col, ms=3.6, lw=1.5, label=f"{name}, test")
        if show_train:
            tr = [r[name]["train"] for r in have]
            ax.plot(D, tr, "s", color=col, ls=DASH_TRAIN, ms=3.2, lw=1.0, mfc="white",
                    label=f"{name}, train")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ticks = sorted(xs_all)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.minorticks_off()
    # the decade locator would label only 10^-1 across this range
    lo, hi = ax.get_ylim()
    yt = decimal_log_ticks(lo, hi)
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.minorticks_off()
    ax.set_xlabel(DIM)
    ax.set_ylabel(NMSE)
    ax.legend(loc="lower left", fontsize=FS["legend"])

    # -------------------------------- right: representative fields, 2 x 2
    inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 1],
                                    hspace=0.14, wspace=0.09)
    gt = m["truth"][iz]
    U = float(np.percentile(np.abs(gt), a.vel_percentile))
    nm = {r["dof"]: r for r in rows}[Dstar]
    im = None
    for i, (key, lab) in enumerate(ROWS):
        r_, c_ = divmod(i, 2)
        axi = fig.add_subplot(inner[r_, c_])
        im = axi.pcolormesh(xc, yc, m[key][iz], cmap="RdBu_r",
                            vmin=-U, vmax=U, shading="nearest",
                            rasterized=True)
        axi.set_aspect("equal")
        axi.set_xlim(0, Lx); axi.set_ylim(0, Ly)
        axi.set_yticks([0, 1.5]); axi.set_xticks([0, 1, 2, 3])
        axi.set_title(lab, pad=5, fontsize=FS["panel_title"])
        if r_ == 1:
            axi.set_xlabel("$x/h$", labelpad=3)
        else:
            axi.set_xticklabels([])
        if c_ == 0:
            axi.set_ylabel("$y/h$", labelpad=3)
        else:
            axi.set_yticklabels([])

    cb = fig.colorbar(im, cax=fig.add_subplot(outer[0, 2]))
    cb.set_label(r"$u'/u_\tau$", labelpad=6,
                 fontsize=FS["axis_label"])
    cb.ax.tick_params(labelsize=FS["tick"], width=LW["tick"])
    cb.outline.set_linewidth(LW["outline"])

    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        p = f"{a.out}.{ext}"
        fig.savefig(p, **kw)
        print("wrote", p)
    plt.close(fig)

    prov = {"nmse_source": a.nmse_json, "recon_source": a.recon_npz,
            "representative_dof": Dstar, "snapshot": int(m["snapshot_index"]),
            "plane_k": iz, "z_plus": float(m["z_plus"]),
            "velocity_limit_U": U, "velocity_percentile": a.vel_percentile,
            "dimension_convention": {
                "POD": "plotted D = stored rank / 2 (mode pairs)",
                "OPINE": "plotted D = stored rank / 2 (mode pairs)",
                "CNN": "plotted D = stored latent dimension",
                "_why": "the covariance of a field statistically homogeneous "
                        "in x and y commutes with the translation group, so "
                        "non-trivial modes come in 2-D irreps; in this 3-D "
                        "case the bottleneck is unrestricted, so the pairing "
                        "is statistical, not enforced by construction"},
            "table": {str(r["dof"]): {k: {"stored_rank": r["dof"],
                                          "plotted_D": r["dof"] // div,
                                          "train": r[k]["train"],
                                          "test": r[k]["test"]}
                                      for k, _, div, _s in METHODS if k in r}
                      for r in rows}}
    q = f"{a.out}_provenance.json"
    json.dump(prov, open(q, "w"), indent=1)
    print("wrote", q)
    print(json.dumps(prov["table"], indent=1))


if __name__ == "__main__":
    main()
