#!/usr/bin/env python3
"""Summary figure for the 2-D y-z channel, matching the 3-D and Kolmogorov ones.

Left   retained dimension D vs NMSE, train AND test, for POD, CNN and OPINE.
Right  representative reconstruction at one D as a grid: one ROW per velocity
       component (u, v, w) and one COLUMN per method, so each row is the same
       y-z plane of one component seen through every baseline.

The panels put the SPANWISE direction y on the horizontal axis and the
WALL-NORMAL direction z on the vertical axis, so the wall is the bottom edge
of every panel and the periodic direction is the long one.  Each panel is
therefore 128 wide by 64 tall.

Same conventions as plot_summary3d.py / plot_summary_kolmogorov.py:
  - POD and OPINE report D as the number of mode PAIRS (stored rank / 2); the
    CNN reports its stored latent dimension.  In the y-z case the bottleneck
    is the banded SymPOD, whose retained modes are grouped into complete k_y
    bands by construction, so the pairing is ENFORCED rather than statistical.
  - no in-figure titles and no block of caption text
  - Original FINE and iResNet FINE are excluded

Each row carries its OWN symmetric colour limit and its OWN colorbar, set from
the 99th percentile of |reference| for that component alone.  v' and w' are
several times weaker than u', so a shared scale would render those two rows
nearly blank; a per-row scale keeps every row comparable ACROSS methods, which
is the comparison the figure is making.  The limits are recorded per component
in the provenance JSON.

The axes are placed in inches rather than by GridSpec: each cell must have
exactly the 2:1 aspect of the plane or aspect-equal shrinks the panel inside
its cell and opens ragged gaps across a 3 x 4 grid.

Nothing is retrained and no NMSE is recomputed: the left panel reads
nmse2d_train_test.json (gathered by collect_nmse_2d.py) and the right panel
reads the stored representative snapshot.
"""
import argparse, math, json, os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
from opine_experiments.common.paper_style import (  # noqa: E402
    apply as _apply_style, FS, LW, DASH_TRAIN, NMSE, DIM)
_apply_style()

# (name, colour, pair divisor, draw train curve).  POD shows test only.
# Reporting convention: the retained REAL dimension is reported directly
# for every method.  POD and OPINE are indexed by the real rank they
# retain, the CNN by its latent width.  Nothing is divided by two -- the
# earlier D = r/2 mode-pair convention is withdrawn, because the pairing
# is enforced only in the banded y-z bottleneck and is a finite-sample
# property elsewhere.
METHODS = [("POD", "#4a7c59", 1, False), ("CNN", "#c98b2e", 1, True),
           ("OPINE", "#b2404a", 1, True)]
# Reporting convention: the retained REAL dimension is reported directly
# for every method.  POD and OPINE are indexed by the real rank they
# retain, the CNN by its latent width.  Nothing is divided by two: the
# earlier D = r/2 mode-pair convention is withdrawn, because the pairing
# is enforced only in the banded y-z bottleneck and is a finite-sample
# property elsewhere.
PAIRED = {"POD": 1, "CNN": 1, "OPINE": 1}
# npz key -> display label, one per column of the reconstruction block
PANELS = [("truth", "Reference"), ("pod", "POD"), ("cnn", "CNN"),
          ("coupling", "OPINE")]

# ------------------------------------------------- layout, all in inches
PW = 1.45          # width of one reconstruction panel; the height follows
WS, HS = 0.06, 0.13   # column / row gap, as a fraction of panel size
LEFT_W, LEFT_H = 3.00, 2.75   # the NMSE axes
M_L, M_R, M_T, M_B = 0.62, 0.74, 0.30, 0.46   # figure margins
GAP_LEFT = 0.62    # NMSE axes -> reconstruction block
GAP_CB = 0.22      # reconstruction block -> colorbars
CB_W = 0.10


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
    ap.add_argument("--components", type=int, nargs="+", default=[0, 1, 2],
                    help="0=u, 1=v, 2=w; one row each")
    ap.add_argument("--vel-percentile", type=float, default=99.0)
    ap.add_argument("--min-plotted-d", type=int, default=32,
                    help="drop points below this retained dimension; the\n"
                         "methods share a D grid that starts at 16, and a\n"
                         "stored rank evaluated only for the CNN would otherwise\n"
                         "put POD alone at half of it")
    a = ap.parse_args()

    j = json.load(open(a.nmse_json))
    rows = sorted(j["rows"], key=lambda r: r["dof"])
    z = np.load(a.recon_npz, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    Dstar = int(meta["dof"])
    comps = [meta["components"][c] for c in a.components]

    missing = [k for k, _ in PANELS if k not in z.files]
    if missing:
        raise SystemExit(f"{a.recon_npz} has no {missing}; it is not the "
                         f"combined file (it holds {list(z.files)})")

    n_row, n_col = len(a.components), len(PANELS)
    # the stored arrays are (component, y, z); the panels show y across and z
    # up, so every field is transposed on its way into pcolormesh
    ny, nz = z["truth"].shape[1], z["truth"].shape[2]
    ph = PW * nz / ny
    dx, dy = PW * (1 + WS), ph * (1 + HS)
    recon_w = n_col * PW + (n_col - 1) * WS * PW
    recon_h = n_row * ph + (n_row - 1) * HS * ph

    content_h = max(recon_h, LEFT_H)
    fig_w = M_L + LEFT_W + GAP_LEFT + recon_w + GAP_CB + CB_W + M_R
    fig_h = M_B + content_h + M_T
    fig = plt.figure(figsize=(fig_w, fig_h))

    def box(x, y, w, h):
        """inches, origin bottom-left -> figure fraction"""
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    # ----------------------------------------------------- left: D vs NMSE
    ax = fig.add_axes(box(M_L, M_B + 0.5 * (content_h - LEFT_H),
                          LEFT_W, LEFT_H))
    xs = set()
    for name, col, div, show_train in METHODS:
        have = [r for r in rows if name in r
                and r["dof"] // div >= a.min_plotted_d]
        D = [r["dof"] // div for r in have]
        xs.update(D)
        ax.plot(D, [r[name]["test"] for r in have], "o-", color=col,
                ms=LW["marker"], lw=LW["curve"], label=f"{name}, test")
        if show_train:
            ax.plot(D, [r[name]["train"] for r in have], "s", color=col, ls=DASH_TRAIN,
                    ms=LW["marker"] * 0.85, lw=LW["curve_thin"],
                    mfc="white", mew=LW["marker_edge"],
                    label=f"{name}, train")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ticks = sorted(xs)
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    # the decade locator would label only 10^-1 across this range
    lo, hi = ax.get_ylim()
    yt = decimal_log_ticks(lo, hi)
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.minorticks_off()
    ax.set_xlabel(DIM)
    ax.set_ylabel(NMSE)
    ax.legend(loc="lower left", fontsize=FS["legend"])

    # ------------- right: one row per component, one column per method
    x0 = M_L + LEFT_W + GAP_LEFT
    y_top = M_B + 0.5 * (content_h + recon_h)
    limits = {}
    for r, c in enumerate(a.components):
        gt = z["truth"][c]
        U = float(np.percentile(np.abs(gt), a.vel_percentile))
        limits[meta["components"][c]] = U
        y = y_top - r * dy - ph
        im = None
        for i, (key, lab) in enumerate(PANELS):
            axi = fig.add_axes(box(x0 + i * dx, y, PW, ph))
            im = axi.pcolormesh(np.arange(ny), np.arange(nz), z[key][c].T,
                                cmap="RdBu_r", vmin=-U, vmax=U,
                                shading="nearest", rasterized=True)
            axi.set_aspect("equal")
            axi.set_xlim(-0.5, ny - 0.5); axi.set_ylim(-0.5, nz - 0.5)
            axi.set_xticks([0, 64, 127]); axi.set_yticks([0, 32, 63])
            if r == 0:
                axi.set_title(lab, pad=5, fontsize=FS["panel_title"])
            # every panel is the same plane on the same grid, so the ticks
            # are labelled once per figure: repeating them makes the "127" of
            # one panel collide with the "0" of the next
            if r == n_row - 1 and i == 0:
                axi.set_xlabel("$y$ index", labelpad=3)
            else:
                axi.set_xticklabels([])
            if i == 0:
                axi.set_ylabel(rf"${meta['components'][c]}'$", labelpad=4,
                               fontsize=FS["axis_label"])
            else:
                axi.set_yticklabels([])

        cax = fig.add_axes(box(x0 + recon_w + GAP_CB, y, CB_W, ph))
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(rf"${meta['components'][c]}'$", labelpad=5,
                     fontsize=FS["axis_label"])
        cb.outline.set_linewidth(LW["outline"])
        cb.ax.tick_params(labelsize=FS["tick"], width=LW["tick"])
        cb.locator = matplotlib.ticker.MaxNLocator(nbins=4, symmetric=True)
        cb.update_ticks()

    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        p = f"{a.out}.{ext}"
        fig.savefig(p, **kw)
        print("wrote", p)
    plt.close(fig)

    # The recon panel's D is the RETAINED dimension, the one the left panel
    # plots: POD and OPINE hold twice that many stored modes.  Older exports
    # carry only "dof", the stored rank, and are read as rank / 2.
    retained = meta.get("retained_D", {})
    prov = {"nmse_source": a.nmse_json, "recon_source": a.recon_npz,
            "representative_stored_rank": Dstar,
            "representative_stored_rank_per_method":
                meta.get("stored_rank", {"POD": Dstar, "OPINE": Dstar,
                                         "CNN": Dstar}),
            "snapshot": meta["sample_index"],
            "snapshot_selection_rule": meta["selection_rule"],
            "recon_layout": "rows = velocity components, columns = methods",
            "components": comps,
            "velocity_limits_per_component": limits,
            "velocity_percentile": a.vel_percentile,
            "_colour_scale_note":
                "each row has its own symmetric limit, taken from the "
                "reference of THAT component only, so the comparison the "
                "figure makes -- across methods within a row -- is on one "
                "scale; limits are NOT comparable between rows.  The fields "
                "are normalized by the training-split RMS of each component.",
            "axes": {"panel_x": "y index (spanwise, periodic, 128 points)",
                     "panel_y": "z index (wall-normal, 64 points, wall at 0, "
                                "so the wall is the bottom edge)"},
            "excluded_methods": j.get("_excluded", []),
            "dimension_convention": j.get("_dimension_convention", {}),
            "table": {str(r["dof"]): {k: {"stored_rank": r["dof"],
                                          "plotted_D": r["dof"] // div,
                                          "train": r[k]["train"],
                                          "test": r[k]["test"]}
                                      for k, _, div, _s in METHODS if k in r}
                      for r in rows}}
    q = f"{a.out}_provenance.json"
    json.dump(prov, open(q, "w"), indent=1)
    print("wrote", q)


if __name__ == "__main__":
    main()
