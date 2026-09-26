#!/usr/bin/env python3
"""Canonical iid-Uniform toy: per-seed spread and error vs distance to training."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "iid_results.json")))
INK, GREY, FAINT = "#1a1a1a", "#8a8a8a", "#c9c9c9"
C = {"raw": "#3a3a3a", "cpl": "#b2432a", "orc": "#2f6f5e", "nn": "#7a6ea8"}
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from opine_experiments.common.paper_style import (  # noqa: E402
    apply as _apply_style, FS, LW, NMSE)
_apply_style()
plt.rcParams.update({"figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def style(ax):
    ax.grid(alpha=0.22, lw=0.8, color=FAINT); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
        ax.spines[s].set_linewidth(LW["axes"])
    ax.tick_params(labelsize=FS["tick"], length=4.0,
                   width=LW["tick"], pad=3, colors=INK)


fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

# left: per-seed test NMSE
ax = axes[0]
runs = R["runs"]
xs = np.arange(len(runs))
for k, lab, col, mk in (("raw_pod_test", "POD", C["raw"], "o"),
                        ("nearest_copy_test", "nearest-phase copy",
                         C["nn"], "v"),
                        ("coupling_test", "OPINE", C["cpl"], "s"),
                        ("oracle_test", "Oracle", C["orc"], "^")):
    v = [max(r[k], 1e-33) for r in runs]
    ax.semilogy(xs, v, mk, color=col, ms=LW["marker"], mfc="white", mew=LW["marker_edge"], label=lab)
    m = R["summary"][k]["mean"]
    ax.axhline(max(m, 1e-33), color=col, lw=LW["curve_thin"], ls="--", alpha=0.55)
ax.set_xlabel("seed", labelpad=3); ax.set_xticks(xs)
ax.set_ylabel(r"test $\mathrm{NMSE}$ at $D=2$", labelpad=4)
ax.set_ylim(1e-33, 3e2)
ax.legend(fontsize=FS["legend"], ncol=1, loc="center left"); style(ax)

# right: NMSE by distance-to-nearest-training-phase bin
ax = axes[1]
b = R["binned"]
labs = ["closest\nthird", "middle\nthird", "farthest\nthird"]
w, xp = 0.26, np.arange(3)
for i, (k, lab, col) in enumerate((("raw_pod", "POD", C["raw"]),
                                   ("nearest_copy", "nearest-phase copy",
                                    C["nn"]),
                                   ("coupling", "OPINE", C["cpl"]))):
    m = [b[k][j]["mean"] for j in range(3)]
    e = [b[k][j]["std"] for j in range(3)]
    ax.bar(xp + (i - 1) * w, m, w * 0.9, yerr=e, color=col, label=lab,
           linewidth=0, error_kw={"lw": LW["curve_thin"], "capsize": 4,
                                  "ecolor": "#555555"})
ax.set_yscale("log"); ax.set_ylim(1e-5, 8.0)
for j in range(3):
    r = b["nearest_copy"][j]["mean"] / b["coupling"][j]["mean"]
    # below the bars, not above: above collides with the legend
    ax.text(xp[j], 1.5e-5, f"copy/OPINE  {r:.0f}$\\times$", ha="center", fontsize=FS["small"],
            color=INK)
ax.set_xticks(xp); ax.set_xticklabels(labs)
ax.set_xlabel("distance to nearest training phase", labelpad=3)
ax.set_ylabel(NMSE, labelpad=4)
ax.legend(fontsize=FS["legend"], loc="upper left"); style(ax)
fig.subplots_adjust(wspace=0.26)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"iid_canonical.{e}"), dpi=400,
                bbox_inches="tight", pad_inches=0.015)
print("  [wrote] iid_canonical")
