#!/usr/bin/env python3
"""One publication style for every result figure in the paper.

The figures are drawn at 7-7.5 in and placed at roughly the ICLR text width,
so everything on the page is about 0.75x what it is here.  The sizes below
are chosen so the SMALLEST text in a figure still prints at about 8 pt and
axis labels at about 10 pt; that is why they look large in the raw png.

Mathematics is typeset with matplotlib's own mathtext in the STIX fonts, so
Greek letters, primes, subscripts and superscripts come out in a Times-like
face that matches the body text of the paper.  No external LaTeX is needed.

Import and call `apply()` before creating any figure.  Scripts that need a
size for an individual artist should take it from `FS` rather than hard-code
a number, so the hierarchy stays consistent across figures.
"""
import matplotlib

# one hierarchy, used everywhere
FS = {
    "axis_label": 13.0,     # x/y labels, colorbar labels
    "tick": 11.5,           # tick numerals
    "legend": 11.5,
    "panel_title": 13.0,    # "POD", "CNN", "OPINE", "Reference"
    "panel_tag": 13.0,      # (a), (b), ...
    "annotation": 11.5,     # in-panel text, colour-key captions
    "small": 10.5,          # the least important text in any figure
}

LW = {
    "curve": 2.4,           # quantitative curves
    "curve_thin": 1.8,      # train curves drawn under the test curves
    "marker": 7.0,
    "marker_edge": 1.6,
    "axes": 1.1,
    "tick": 1.1,
    "outline": 0.9,         # colorbar and panel outlines
}


def apply(dpi=400):
    matplotlib.rcParams.update({
        # keep text as text in the pdf
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        # Times-like faces for both prose and mathematics
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": FS["annotation"],
        "axes.labelsize": FS["axis_label"],
        "axes.titlesize": FS["panel_title"],
        "xtick.labelsize": FS["tick"],
        "ytick.labelsize": FS["tick"],
        "legend.fontsize": FS["legend"],
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.35,
        "axes.linewidth": LW["axes"],
        "xtick.major.width": LW["tick"], "ytick.major.width": LW["tick"],
        "xtick.minor.width": LW["tick"] * 0.7,
        "ytick.minor.width": LW["tick"] * 0.7,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "xtick.minor.size": 2.6, "ytick.minor.size": 2.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": LW["curve"],
        "lines.markersize": LW["marker"],
        "lines.markeredgewidth": LW["marker_edge"],
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "figure.dpi": dpi, "savefig.dpi": dpi,
        "savefig.bbox": None,
    })


# dash patterns that stay distinguishable at print size
DASH_TRAIN = (0, (5.5, 2.4))
DASH_ALT = (0, (1.6, 1.8))

NMSE = r"$\mathrm{NMSE}$"
DIM = r"Retained dimension  $D$"
