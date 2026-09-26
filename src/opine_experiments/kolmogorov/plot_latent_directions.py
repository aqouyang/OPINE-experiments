#!/usr/bin/env python3
"""Decoded latent directions for the Kolmogorov flow: POD, CNN and OPINE.

A standalone paper figure -- deliberately NOT part of the summary figure,
which carries the NMSE curve and the reconstruction row.

Each panel is the physical field obtained by activating exactly ONE reduced
coordinate of one method at an amplitude of one training standard deviation
and mapping it back to the vorticity plane.  The fields are computed by
compute_latent_directions.py; nothing is recomputed here.

LAYOUT.  Two blocks side by side, one per pair; inside a block the two ROWS
are the pair's two coordinates and the COLUMNS are the methods:

        pair 1                        pair 2
    POD  OPINE | CNN              POD  OPINE | CNN
    j=1   j=1  | #32              j=3   j=3  | #37
    j=2   j=2  | #15              j=4   j=4  | #42

Two panel rows rather than three is what makes the figure wide and low: a
three-row arrangement caps the aspect near 1.9 whatever the panel size, this
one reaches 3.  It also puts POD and OPINE in adjacent columns, which is the
comparison the figure exists for -- OPINE's directions are the POD structure
sheared -- and it turns a pair into two vertically adjacent panels, where the
quarter-wavelength shift is easiest to see.

PAIRING BELONGS TO POD AND OPINE ONLY.  x is a homogeneous periodic direction
of Kolmogorov flow (the cos(k_f y) forcing breaks homogeneity in y only), so
the covariance commutes with the x-translation group and its non-trivial real
irreps are two-dimensional: directions come in pairs, the same structure a
quarter wavelength apart.  A plain autoencoder latent has no such structure,
so the CNN column is set apart from the two paired ones inside each block,
labelled unpaired, and its panels carry stored latent indices rather than an
ordinal.

These are NOT all "modes".  Only POD is a mode in the linear sense; the other
two are responses of a nonlinear decoder to a one-hot excursion.
"""
import argparse, json, os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
from opine_experiments.common.paper_style import (  # noqa: E402
    apply as _apply_style, FS, LW)
_apply_style()

# column order inside a block: the two paired methods adjacent, the unpaired
# one set apart on the right
COLS = [("pod", "POD", False), ("opine", "OPINE", False),
        ("cnn", "CNN", True)]
N_PAIRS = 2

# --------------------------------------------- layout, all in inches
PW = 1.02           # square panel
WS = 0.03           # gap between the two paired columns, fraction of a panel
HS = 0.03           # gap between the two rows of a pair, fraction of a panel
CNN_GAP = 0.24      # gap setting the unpaired column apart
BLOCK_GAP = 0.60    # channel between the two pair blocks
M_L, M_R, M_T, M_B = 0.10, 0.58, 0.60, 0.08
GAP_CB, CB_W = 0.20, 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", choices=("shared", "method"), default="shared")
    ap.add_argument("--percentile", type=float, default=99.5,
                    help="percentile of |field| setting the symmetric limit")
    a = ap.parse_args()

    z = np.load(a.fields_npz, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    ny, nx = z["pod_0"].shape

    def fields(key):
        return [z[f"{key}_{j}"] for j in range(2 * N_PAIRS)]

    if a.scale == "shared":
        pool = np.concatenate([np.abs(f).ravel()
                               for k, _, _ in COLS for f in fields(k)])
        U = {k: float(np.percentile(pool, a.percentile)) for k, _, _ in COLS}
    else:
        U = {k: float(np.percentile(np.abs(np.stack(fields(k))),
                                    a.percentile)) for k, _, _ in COLS}

    dyr = PW * (1 + HS)
    block_w = 2 * PW + WS * PW + CNN_GAP + PW
    grid_w = N_PAIRS * block_w + BLOCK_GAP
    grid_h = 2 * PW + HS * PW
    fig_w = M_L + grid_w + GAP_CB + CB_W + M_R
    fig_h = M_B + grid_h + M_T
    fig = plt.figure(figsize=(fig_w, fig_h))

    def box(x, y, w, h):
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    def col_x(p, c):
        """x of method column c inside pair block p."""
        x = M_L + p * (block_w + BLOCK_GAP)
        return x + c * PW * (1 + WS) + (CNN_GAP - WS * PW if c == 2 else 0.0)

    y_top = fig_h - M_T
    ims = {}
    for p in range(N_PAIRS):
        for c, (key, lab, apart) in enumerate(COLS):
            idx = meta[lab]["directions"]
            f = fields(key)
            for k in range(2):                      # the pair's two members
                y = y_top - k * dyr - PW
                ax = fig.add_axes(box(col_x(p, c), y, PW, PW))
                ims[key] = ax.pcolormesh(
                    np.arange(nx), np.arange(ny), f[2 * p + k], cmap="RdBu_r",
                    vmin=-U[key], vmax=U[key], shading="nearest",
                    rasterized=True)
                ax.set_aspect("equal")
                ax.set_xlim(-0.5, nx - 0.5); ax.set_ylim(-0.5, ny - 0.5)
                ax.set_xticks([]); ax.set_yticks([])
                # each panel names its own coordinate: the CNN's are stored
                # latent indices, not an ordinal, so no column header could
                # carry them
                n = idx[2 * p + k]
                tag = f"#{n}" if key == "cnn" else rf"$j\,{{=}}\,{n}$"
                ax.text(0.05, 0.95, tag, transform=ax.transAxes,
                        ha="left", va="top", fontsize=FS["small"], color="0.12",
                        bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
            xc = (col_x(p, c) + 0.5 * PW) / fig_w
            fig.text(xc, (y_top + 0.06) / fig_h, lab, ha="center",
                     va="bottom", fontsize=FS["panel_title"])
            if apart:
                # clear of the method name below it: at 13 pt that name is
                # about 0.18 in tall above its own baseline at +0.06
                fig.text(xc, (y_top + 0.28) / fig_h, "unpaired", ha="center",
                         va="bottom", fontsize=FS["small"], color="0.45")
        # the block title is the only pair marking; the layout does the rest
        xc = (col_x(p, 0) + 0.5 * (2 * PW + WS * PW)) / fig_w
        fig.text(xc, (y_top + 0.34) / fig_h, f"pair {p + 1}", ha="center",
                 va="bottom", fontsize=FS["panel_tag"], color="0.15")

    cax = fig.add_axes(box(M_L + grid_w + GAP_CB,
                           y_top - grid_h, CB_W, grid_h))
    cb = fig.colorbar(ims["pod"], cax=cax)
    cb.set_label(r"$\Delta\omega$", labelpad=6,
                 fontsize=FS["axis_label"])
    cb.ax.tick_params(labelsize=FS["tick"], width=LW["tick"])
    cb.outline.set_linewidth(LW["outline"])
    cb.ax.tick_params(labelsize=6.8)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        q = f"{a.out}.{ext}"
        fig.savefig(q, **kw)
        print("wrote", q)
    plt.close(fig)

    prov = {"fields_source": a.fields_npz,
            "dof": meta["dof"],
            "_dof_is": meta.get("_dof_is"),
            "stored_rank": meta.get("stored_rank"),
            "pair_divisor": meta.get("pair_divisor"),
            "what": "decoded latent directions: one reduced coordinate "
                    "activated at one training standard deviation, mapped "
                    "back to the vorticity plane",
            "layout": "two blocks side by side, one per pair; inside a block "
                      "the rows are the pair's two coordinates and the "
                      "columns are the methods, POD and OPINE adjacent and "
                      "the unpaired CNN set apart",
            "figure_size_in": [fig_w, fig_h],
            "colour_scale": a.scale,
            "colour_percentile": a.percentile,
            "colour_limits": U,
            "_naming": "only POD is a mode in the linear sense; the CNN and "
                       "OPINE panels are decoder responses to a one-hot "
                       "excursion, so the figure is called decoded latent "
                       "directions rather than modes",
            "directions": {lab: meta[lab]["directions"] for _, lab, _ in COLS},
            "alpha": {lab: meta[lab]["alpha"] for _, lab, _ in COLS},
            "cnn_selection": {
                "criterion": meta["CNN"].get("selection_criterion"),
                "chosen": meta["CNN"]["directions"],
                "top4_by_decoded_response_rms":
                    meta["CNN"].get("top4_by_decoded_response_rms"),
                "criteria_agree_on": meta["CNN"].get("criteria_agree_on")},
            "pairing": {
                "POD": meta["POD"]["adjacent_pair_relative_split"],
                "OPINE": meta["OPINE"]["adjacent_pair_relative_split"],
                "_what": "relative split |l_a - l_b| / mean(l_a, l_b) of the "
                         "coordinate variances inside each adjacent pair; "
                         "small means the two directions carry the same "
                         "energy, as a translation pair must",
                "CNN": "not paired; a plain autoencoder latent has no "
                       "translation structure"},
            "compute_meta": meta}
    q = f"{a.out}_provenance.json"
    json.dump(prov, open(q, "w"), indent=1, default=float)
    print("wrote", q)


if __name__ == "__main__":
    main()
