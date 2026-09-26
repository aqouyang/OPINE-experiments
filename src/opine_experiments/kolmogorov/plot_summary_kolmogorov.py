#!/usr/bin/env python3
"""Summary figure for the Kolmogorov flow, matching the 2-D and 3-D ones.

Left   retained dimension D vs NMSE.  POD on test only; CNN and OPINE on
       train AND test.
Right  representative reconstruction at one D, as one horizontal row labelled
       by subplot title.

Same conventions as plot_summary2d.py / plot_summary3d.py:
  - POD and OPINE report D as the number of mode PAIRS (stored rank / 2); the
    CNN reports its stored latent dimension.  x is the homogeneous periodic
    direction of Kolmogorov flow, so non-trivial k_x modes span 2-D real
    irreps of the x-translation group.  Both bottlenecks are unrestricted, so
    the pairing is statistical rather than enforced -- as in the 3-D case.
  - POD means the unrestricted global POD, the live Kolmogorov POD and the
    OPINE round-0 bottleneck.  The PODxTE / PODxC4 symmetry line is
    discontinued and is not used.
  - no in-figure title and no block of caption text
  - Original FINE and iResNet FINE are excluded

Nothing is retrained and no NMSE is recomputed: the left panel reads
nmse_kolmogorov_train_test.json (gathered by collect_nmse_kolmogorov.py) and
the right panel reads the stored representative snapshot.
"""
import argparse, math, json, os

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

# (name, colour, pair divisor, draw train curve).  POD shows test only.
# Reporting convention: the retained REAL dimension is reported directly
# for every method.  POD and OPINE are indexed by the real rank they
# retain, the CNN by its latent width.  Nothing is divided by two -- the
# earlier D = r/2 mode-pair convention is withdrawn, because the pairing
# is enforced only in the banded y-z bottleneck and is a finite-sample
# property elsewhere.
METHODS = [("POD", "#4a7c59", 1, False), ("CNN", "#c98b2e", 1, True),
           ("OPINE", "#b2404a", 1, True)]
# npz key -> display label
PANELS = [("truth", "Reference"), ("pod", "POD"), ("cnn", "CNN"),
          ("coupling", "OPINE")]


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

    # Wider canvas than the y-z figure: the Kolmogorov panels are square
    # (128 x 128) rather than tall, so four of them side by side need more
    # width before aspect-equal stops shrinking them vertically.  The width
    # ratio is set so the panels are height-limited rather than width-limited,
    # which is what makes them as large as the row allows.
    fig = plt.figure(figsize=(11.6, 3.05))
    outer = GridSpec(1, 3, figure=fig, width_ratios=[1.00, 2.60, 0.024],
                     wspace=0.14, left=0.082, right=0.935,
                     top=0.895, bottom=0.200)

    # ----------------------------------------------------- left: D vs NMSE
    ax = fig.add_subplot(outer[0, 0])
    xs = set()
    for name, col, div, show_train in METHODS:
        have = [r for r in rows if name in r
                and r["dof"] // div >= a.min_plotted_d]
        D = [r["dof"] // div for r in have]
        xs.update(D)
        ax.plot(D, [r[name]["test"] for r in have], "o-", color=col,
                ms=LW["marker"], lw=LW["curve"], label=f"{name}, test")
        if show_train:
            ax.plot(D, [r[name]["train"] for r in have], "s", color=col,
                    ls=DASH_TRAIN, ms=LW["marker"] * 0.85,
                    lw=LW["curve_thin"], mfc="white",
                    mew=LW["marker_edge"], label=f"{name}, train")
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

    # ------------------- right: representative fields, one horizontal row
    # The stored symmetric limits come from the reference field only, so every
    # panel is on one scale and no method is autoscaled.
    U = float(meta["vmax"])
    inner = GridSpecFromSubplotSpec(1, len(PANELS), subplot_spec=outer[0, 1],
                                    wspace=0.05)
    ny, nx = z["truth"].shape
    im = None
    for i, (key, lab) in enumerate(PANELS):
        axi = fig.add_subplot(inner[0, i])
        im = axi.pcolormesh(np.arange(nx), np.arange(ny), z[key],
                            cmap="RdBu_r", vmin=-U, vmax=U,
                            shading="nearest", rasterized=True)
        axi.set_aspect("equal")
        axi.set_xlim(-0.5, nx - 0.5); axi.set_ylim(-0.5, ny - 0.5)
        # the panels are a qualitative comparison on one identical grid, so
        # they carry no axes at all -- only the method name above each one
        axi.set_xticks([]); axi.set_yticks([])
        axi.set_title(lab, pad=5, fontsize=FS["panel_title"])

    cb = fig.colorbar(im, cax=fig.add_subplot(outer[0, 2]))
    cb.set_label(r"$\omega$ (normalized)", labelpad=6,
                 fontsize=FS["axis_label"])
    cb.ax.tick_params(labelsize=FS["tick"], width=LW["tick"])
    cb.outline.set_linewidth(LW["outline"])

    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        p = f"{a.out}.{ext}"
        fig.savefig(p, **kw)
        print("wrote", p)
    plt.close(fig)

    # The panel's D is the RETAINED dimension, the one the left panel plots:
    # POD and OPINE hold twice that many stored modes.  Older exports carry
    # only "dof", the stored rank, and are read as rank / 2.
    retained = meta.get("retained_D", {})
    prov = {
        "nmse_source": a.nmse_json, "recon_source": a.recon_npz,
        "representative_stored_rank": Dstar,
        "representative_stored_rank_per_method":
            meta.get("stored_rank", {"POD": Dstar, "OPINE": Dstar,
                                     "CNN": Dstar}),
        "snapshot": meta["sample_index"],
        "snapshot_selection_rule": meta["selection_rule"],
        "field": "out-of-plane vorticity omega, normalized by sigma_train",
        "velocity_limit_U": U,
        "_recon_panel_caveats": [
            "The meta of the representative-sample file reports "
            "*_nmse_test_set as the MEAN of per-snapshot ratios "
            "(select_representative.py: p.sum()/len(p)), not the canonical "
            "ratio of sums.  Those meta numbers are therefore NOT the "
            "canonical NMSE and are not used anywhere in this figure; the "
            "left panel reads the canonical table.  The reconstructed fields "
            "themselves come from the canonical D=64 models.",
            "The POD panel is the unrestricted POD basis, which is the POD "
            "baseline, so it matches the left panel."],
        "excluded_methods": j.get("_excluded", []),
        "pod_definition": j.get("_pod_definition"),
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
