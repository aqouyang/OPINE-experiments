#!/usr/bin/env python3
"""u' isosurfaces of the y-z channel reconstruction, stacked over x.

The y-z methods reconstruct one plane at a time.  compute_volume_recon.py
runs every x-plane of one DNS snapshot through them independently and stacks
the results; this draws the near-wall u' isosurfaces of that volume, one
panel per method.

The figure is a test the methods were never trained for.  Nothing in POD,
the CNN or OPINE couples neighbouring x-planes, so any streamwise coherence
in these surfaces is a by-product of each plane being reconstructed well.
Where a method breaks up into plane-thin fragments, that is exactly the
missing coupling showing itself.

Conventions are those of minimal_channel3d/plot_recon3d_paper.py,
whose rendering helpers are imported rather than copied: one threshold taken
from the ground truth and reused, one camera, one parallel scale, true
physical aspect, and surfaces ramped by wall distance over z/h in [0, 1] --
warm for u' > 0 and cool for u' < 0, so the sign still reads.
"""
import argparse, json, os, sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opine_experiments.minimal_channel3d.plot_recon3d_paper import (   # noqa: E402
    _grid, _add_iso, _z_colorbar, _colour_top, _slab_mark,
    ISO_CMAP_POS, ISO_CMAP_NEG, ISO_POS, ISO_NEG, FS, LW)

METHODS = [("truth", "Reference", None), ("pod", "POD", "pod"),
           ("cnn", "CNN", "cnn"), ("opine", "OPINE", "opine")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--component", type=int, default=0, help="0=u,1=v,2=w")
    ap.add_argument("--iso-percentile", type=float, default=90.0)
    ap.add_argument("--zmax", type=float, default=0.40,
                    help="restrict the isosurface box to z/h <= this")
    ap.add_argument("--elev", type=float, default=24.0)
    ap.add_argument("--azim", type=float, default=-62.0)
    ap.add_argument("--opacity", type=float, default=1.0)
    ap.add_argument("--zoom", type=float, default=1.15)
    ap.add_argument("--px", type=int, default=1500)
    ap.add_argument("--z-clim", default="0.5",
                    help="top of the z/h colour scale: an explicit z/h "
                         "(default 0.5), 'unit' for the channel half-height "
                         "or 'slab' for the plotted slab")
    a = ap.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True

    z = np.load(a.volume_npz, allow_pickle=True)
    m = json.loads(str(z["meta"]))
    comp = m["components"][a.component]
    Lx, Ly, Lz = m["Lx"], m["Ly"], m["Lz"]
    nx, ny, nz = m["nx"], m["ny"], m["nz"]
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    kmax = int(min(nz, max(4, int(np.ceil(a.zmax / dz)))))
    zt = kmax * dz
    zc = _colour_top(a.z_clim, Lz, zt)    # top of the COLOUR scale
    cnn_dof_meta = m.get("stored_rank", {}).get("CNN", m["dof"])

    # the stored layout is (component, x, y, z); the renderer wants (z, y, x)
    def vol(key):
        return np.ascontiguousarray(
            z[key][a.component].transpose(2, 1, 0)[:kmax])

    truth = vol("truth")
    c = float(np.percentile(np.abs(truth), a.iso_percentile))
    print(f"threshold |{comp}'|/u_tau = {c:.4f}  (percentile "
          f"{a.iso_percentile:g} of the GROUND TRUTH over z/h <= {zt:.3f})",
          flush=True)

    ctr = np.array([Lx / 2, Ly / 2, zt / 2])
    e, az = np.deg2rad(a.elev), np.deg2rad(a.azim)
    R = 2.35 * Lx
    pos = ctr + R * np.array([np.cos(e) * np.cos(az),
                              np.cos(e) * np.sin(az), np.sin(e)])
    cpos = [tuple(pos), tuple(ctr), (0.0, 0.0, 1.0)]
    bounds = (0.0, Lx, 0.0, Ly, 0.0, zt)

    pscale, shots, counts = [None], {}, {}
    for key, name, _ in METHODS:
        g = _grid(vol(key), dx, dy, dz)
        pl = pv.Plotter(off_screen=True,
                        window_size=(a.px, int(a.px * 0.42)))
        pl.set_background("white")
        n_cells = 0
        for level, flat, cmap in ((+c, ISO_POS, ISO_CMAP_POS),
                                  (-c, ISO_NEG, ISO_CMAP_NEG)):
            surf = g.contour([level], scalars="u")
            if surf.n_points == 0:
                continue
            n_cells += surf.n_cells
            _add_iso(pl, surf, "z", flat, cmap, zc, a.opacity)
        counts[key] = n_cells
        pl.add_mesh(pv.Box(bounds).outline(), color="#999999", line_width=1.2)
        pl.camera_position = cpos
        pl.enable_parallel_projection()
        pl.enable_depth_peeling(number_of_peels=8)
        if pscale[0] is None:
            pl.reset_camera(bounds=bounds)
            pscale[0] = float(pl.camera.parallel_scale) / a.zoom
        pl.camera_position = cpos
        pl.camera.parallel_scale = pscale[0]
        shots[key] = pl.screenshot(return_img=True)
        pl.close()

    # one crop box for every panel: framing, not per-method cropping
    def _bbox(img):
        nzp = np.argwhere((img < 250).any(axis=2))
        return (nzp[:, 0].min(), nzp[:, 0].max(),
                nzp[:, 1].min(), nzp[:, 1].max()) if len(nzp) else None
    bxs = [b for b in (_bbox(v) for v in shots.values()) if b is not None]
    if bxs:
        h, w = next(iter(shots.values())).shape[:2]
        r0, r1 = max(0, min(b[0] for b in bxs) - 6), min(h, max(b[1] for b in bxs) + 7)
        c0, c1 = max(0, min(b[2] for b in bxs) - 6), min(w, max(b[3] for b in bxs) + 7)
        shots = {k: v[r0:r1, c0:c1] for k, v in shots.items()}

    # Inch-based layout, matching the 3-D isosurface figure exactly so the
    # two read as one pair: panel band on top, two sign ramps sharing one
    # z/h axis below, with pitch enough for publication font sizes.
    PANEL_H, RAMP_H, RAMP_PITCH = 1.00, 0.155, 0.235
    KEY_GAP, AXIS_H, TOP_PAD = 0.26, 0.62, 0.30
    fig_h = TOP_PAD + PANEL_H + KEY_GAP + RAMP_PITCH + RAMP_H + AXIS_H
    fig = plt.figure(figsize=(7.4, fig_h))

    def _f(inches_from_bottom):
        return inches_from_bottom / fig_h

    panel_bot = AXIS_H + RAMP_H + RAMP_PITCH + KEY_GAP
    gs = GridSpec(1, 4, figure=fig, wspace=0.012,
                  left=0.004, right=0.996,
                  top=_f(panel_bot + PANEL_H), bottom=_f(panel_bot))
    for i, (key, name, nk) in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(shots[key], interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(name, pad=6, fontsize=FS["panel_title"])

    y_pos, y_neg = AXIS_H + RAMP_PITCH, AXIS_H
    _z_colorbar(fig, [0.400, _f(y_pos), 0.210, _f(RAMP_H)], ISO_CMAP_POS, zc,
                mark=_slab_mark(zc, zt), ticks=False)
    _z_colorbar(fig, [0.400, _f(y_neg), 0.210, _f(RAMP_H)], ISO_CMAP_NEG, zc,
                mark=_slab_mark(zc, zt), ticks=True, label="$z/h$")
    fig.text(0.388, _f(y_pos + 0.5 * RAMP_H), rf"${comp}'/u_\tau=+{c:.2f}$",
             ha="right", va="center", fontsize=FS["annotation"])
    fig.text(0.388, _f(y_neg + 0.5 * RAMP_H), rf"${comp}'/u_\tau=-{c:.2f}$",
             ha="right", va="center", fontsize=FS["annotation"])
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        p = f"{a.out}.{ext}"
        fig.savefig(p, **kw)
        print("wrote", p, flush=True)
    plt.close(fig)

    prov = {"volume_source": a.volume_npz, "dof": m["dof"], "step": m["step"],
            "component": comp,
            "iso_threshold_utau": c, "iso_percentile": a.iso_percentile,
            "threshold_source": "percentile of |u'| in the GROUND TRUTH only, "
                                "over the plotted slab; reused unchanged for "
                                "every method",
            "retained_D": m.get("retained_D", {"POD": m["dof"] // 2,
                                               "OPINE": m["dof"] // 2,
                                               "CNN": cnn_dof_meta}),
            "stored_rank": m.get("stored_rank", {"POD": m["dof"],
                                                 "OPINE": m["dof"],
                                                 "CNN": cnn_dof_meta}),
            "z_range_h": [0.0, zt], "k_max": kmax,
            "z_plus_max": zt * m["re_tau"],
            "colour_by": "z", "clim_z_over_h": [0.0, zc],
            "colours": {"positive_cmap": "jet 1.00-0.68",
                        "negative_cmap": "jet 0.00-0.35"},
            "camera_position": [list(map(float, t)) for t in cpos],
            "elev_deg": a.elev, "azim_deg": a.azim, "opacity": a.opacity,
            "isosurface_cells_per_method": counts,
            "screenshot_px": a.px, "parallel_scale": pscale[0],
            "volume_nmse": m["volume_nmse"],
            "_reconstruction": m["_reconstruction"],
            "_why_x_is_fair": m["_why_x_is_fair"],
            "renderer": f"pyvista {__import__('pyvista').__version__} "
                        f"(off-screen VTK), marching cubes via contour()"}
    q = f"{a.out}_provenance.json"
    json.dump(prov, open(q, "w"), indent=1)
    print("wrote", q, flush=True)


if __name__ == "__main__":
    main()
