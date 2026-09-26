#!/usr/bin/env python3
"""Publication figures for the 3-D minimal-channel reconstruction comparison.

Reads the snapshot exported by plot_recon3d.py (--outdir/recon3d_D*_snapshot*.npz)
so nothing is retrained, no reconstruction is recomputed and no NMSE is
recalculated here: this script only draws.

FIGURE 1  wall-parallel x-y plane at z+ ~ 15
          rows   ground truth / POD / CNN / OPINE
          left   u'/u_tau            one shared diverging norm, from GT only
          right  signed error        one shared diverging norm, POD/CNN/OPINE
          The ground-truth error cell is left empty and labelled "reference".

FIGURE 2  isosurfaces of u'/u_tau = +c and -c, one panel per method.
          c is fixed ONCE from a percentile of |u'| in the GROUND TRUTH and
          reused unchanged for every method.  Its numerical value is printed
          and written into the provenance JSON.

No smoothing anywhere: pcolormesh with shading="nearest" for the planes, raw
marching cubes on the native grid for the isosurfaces.
"""
import argparse, glob, json, os, sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# keep PDF text as text
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
from opine_experiments.common.paper_style import (  # noqa: E402
    apply as _apply_style, FS, LW)
_apply_style()

METHODS = [("truth", "Ground truth", None),
           ("pod", "POD", "nmse_snapshot_pod"),
           ("cnn", "CNN", "nmse_snapshot_cnn"),
           ("opine", "OPINE", "nmse_snapshot_opine")]


def load(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    m = {k: d[k] for k in d.files}
    return m


# ------------------------------------------------------------------ figure 1
def figure1(m, out_base, vel_pct, err_pct):
    iz = int(m["iz"])
    Lx, Ly = float(m["Lx"]), float(m["Ly"])
    nyy, nxx = m["truth"].shape[1], m["truth"].shape[2]
    # cell-centred physical coordinates; shading="nearest" then draws one cell
    # per grid point with no interpolation at all
    xc = (np.arange(nxx) + 0.5) * Lx / nxx
    yc = (np.arange(nyy) + 0.5) * Ly / nyy

    planes = {k: m[k][iz] for k, _, _ in METHODS}
    gt = planes["truth"]
    U = float(np.percentile(np.abs(gt), vel_pct))
    errs = {k: planes[k] - gt for k in ("pod", "cnn", "opine")}
    E = float(np.percentile(np.abs(np.concatenate(
        [e.ravel() for e in errs.values()])), err_pct))

    # columns: field | its colourbar | error | its colourbar.  Each colourbar
    # spans all four rows so it is unambiguous which column it belongs to.
    # The panels are 2:1 (Lx:Ly); constrained_layout removes the dead space
    # that equal-aspect axes otherwise leave inside their gridspec cells.
    fig = plt.figure(figsize=(8.0, 5.0), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.012, h_pad=0.006,
                                wspace=0.012, hspace=0.012)
    gs = GridSpec(4, 4, figure=fig, width_ratios=[1, 0.045, 1, 0.045])
    imf = ime = None
    for r, (key, name, nk) in enumerate(METHODS):
        ax = fig.add_subplot(gs[r, 0])
        imf = ax.pcolormesh(xc, yc, planes[key], cmap="RdBu_r",
                            vmin=-U, vmax=U, shading="nearest",
                            rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
        # The row identity is a separate left-margin label, not part of the
        # axis label: at publication font sizes a stacked
        # "name / NMSE / y/h" ylabel is taller than the panel and the rows
        # run into each other.
        ax.set_ylabel("$y/h$")
        lab = name if nk is None else f"{name}\n$\\mathrm{{NMSE}}$ {float(m[nk]):.3f}"
        ax.annotate(lab, xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-52, 0), textcoords="offset points",
                    rotation=90, ha="center", va="center",
                    fontsize=FS["annotation"], linespacing=1.6)
        if r == 0:
            ax.set_title(r"$u'/u_\tau$", pad=3)
        if r == len(METHODS) - 1:
            ax.set_xlabel("$x/h$")
        else:
            ax.set_xticklabels([])

        ax2 = fig.add_subplot(gs[r, 2])
        if key == "truth":
            ax2.set_xlim(0, Lx); ax2.set_ylim(0, Ly)
            ax2.set_aspect("equal")
            ax2.set_xticks([]); ax2.set_yticks([])
            for sp in ax2.spines.values():
                sp.set_linestyle((0, (3, 3)))
                sp.set_color("0.6")
            ax2.text(0.5, 0.5, "reference", transform=ax2.transAxes,
                     ha="center", va="center", color="0.45",
                     fontsize=FS["panel_title"])
            ax2.set_title(r"$\hat{u}' - u'$  $(/u_\tau)$", pad=3)
        else:
            ime = ax2.pcolormesh(xc, yc, errs[key], cmap="RdBu_r",
                                 vmin=-E, vmax=E, shading="nearest",
                                 rasterized=True)
            ax2.set_aspect("equal")
            ax2.set_xlim(0, Lx); ax2.set_ylim(0, Ly)
            ax2.set_yticklabels([])
            if r == len(METHODS) - 1:
                ax2.set_xlabel("$x/h$")
            else:
                ax2.set_xticklabels([])

    cb1 = fig.colorbar(imf, cax=fig.add_subplot(gs[:, 1]))
    cb1.set_label(r"$u'/u_\tau$", labelpad=4)
    cb1.outline.set_linewidth(LW["outline"])
    cb1.ax.tick_params(labelsize=FS["tick"], width=LW["tick"])
    cb2 = fig.colorbar(ime, cax=fig.add_subplot(gs[:, 3]))
    cb2.set_label(r"$\hat{u}'-u'\ (/u_\tau)$", labelpad=4)
    cb2.outline.set_linewidth(LW["outline"])
    cb2.ax.tick_params(labelsize=FS["tick"], width=LW["tick"])

    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        p = f"{out_base}.{ext}"
        fig.savefig(p, **kw)
        print("wrote", p, flush=True)
    plt.close(fig)
    return {"velocity_limit_U": U, "error_limit_E": E,
            "velocity_percentile": vel_pct, "error_percentile": err_pct,
            "plane_index_k": iz, "z_plus": float(m["z_plus"])}


# ------------------------------------------------------------------ figure 2
# Restrained contrasting pair for +/- u'.  Identical in every panel.  Used
# when --iso-colour flat asks for the old two-tone rendering.
ISO_POS, ISO_NEG = "#b2404a", "#3f6f9f"
Q_COLOUR = "#4a7c59"


def _trunc(name, lo, hi, n=256):
    """A slice of a colormap, so the pale end never fades into the page."""
    c = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_{lo}_{hi}", c(np.linspace(lo, hi, n)))


# Colouring the isosurfaces by wall distance: the near-wall slab is crowded
# and a flat colour gives the eye nothing to sort front from back by.  The
# +/- u' surfaces keep their warm / cool hue families, so the sign still
# reads, and the ramp inside each family carries z.  Both families are cut
# from jet, the same map the Q figure uses: its warm half runs the positive
# surfaces and its cool half the negative ones, each from dark at the wall to
# bright at the top of the scale.  They stop short of jet's green middle so
# the two signs never meet.  Q has no sign to preserve and takes the whole
# map.
ISO_CMAP_POS = _trunc("jet", 1.00, 0.68)
ISO_CMAP_NEG = _trunc("jet", 0.00, 0.35)
Q_CMAP = _trunc("jet", 0.0, 1.0)


def _add_iso(pl, surf, colour_by, flat_colour, cmap, zc, opacity):
    """One isosurface, either flat or ramped by wall-normal position.

    zc is the TOP of the colour scale, not the top of the plotted slab: the
    mapping z/h -> colour is fixed in absolute wall units, so the same colour
    means the same height in every figure whatever the crop.
    """
    kw = dict(opacity=opacity, smooth_shading=False, specular=0.25,
              specular_power=12, ambient=0.28, diffuse=0.82)
    if colour_by == "z":
        surf.point_data["z"] = np.ascontiguousarray(surf.points[:, 2])
        pl.add_mesh(surf, scalars="z", cmap=cmap, clim=(0.0, zc),
                    show_scalar_bar=False, **kw)
    else:
        pl.add_mesh(surf, color=flat_colour, **kw)


def _slab_mark(zc, zt):
    """The slab top, ticked on the z/h ramp only when it is far enough below
    the top of the colour scale to read as a separate label."""
    return zt if zc > 1.5 * zt else None


def _colour_top(z_clim, Lz, zt):
    """Top of the z/h colour scale.

    An absolute z/h, so one colour means one height in every figure: either
    the channel half-height ("unit"), the plotted slab ("slab", which makes
    the scale crop-dependent) or an explicit z/h.
    """
    if z_clim == "unit":
        return Lz
    if z_clim == "slab":
        return zt
    return float(z_clim)


def _z_colorbar(fig, rect, cmap, zc, mark=None, ticks=True, label=None):
    """A thin horizontal z/h ramp, drawn as a plain image rather than a
    colorbar so two of them can be stacked against one shared axis.

    `mark` is the top of the plotted slab: it is ticked so the reader can see
    which part of the channel the panels actually show.
    """
    ax = fig.add_axes(rect)
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap,
              extent=(0.0, zc, 0.0, 1.0))
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(LW["outline"]); sp.set_color("0.35")
    if ticks:
        tk = [0.0, zc] if mark is None else [0.0, mark, zc]
        ax.set_xticks(tk)
        ax.set_xticklabels([f"{t:g}" if t in (0.0, zc) else f"{t:.2f}"
                            for t in tk], fontsize=FS["tick"])
        ax.tick_params(length=3.5, width=LW["tick"], pad=3)
        if label:
            ax.set_xlabel(label, fontsize=FS["axis_label"],
                          labelpad=3)
    else:
        ax.set_xticks([])
    if mark is not None:
        ax.axvline(mark, color="0.25", lw=LW["outline"] * 1.3)
    return ax


def _grid(vol, dx, dy, dz):
    """Wrap a (z, y, x) array as a PyVista uniform grid in physical units.

    The samples are the DNS nodes: x_i = i*dx, y_j = j*dy, z_k = (k+1/2)*dz
    (u and v live on the staggered uv-grid), so the origin carries the half
    cell in z and nothing else.  No resampling, no interpolation.
    """
    import pyvista as pv
    nz, ny, nx = vol.shape
    g = pv.ImageData(dimensions=(nx, ny, nz), spacing=(dx, dy, dz),
                     origin=(0.0, 0.0, 0.5 * dz))
    g.point_data["u"] = np.ascontiguousarray(
        vol.transpose(2, 1, 0)).ravel(order="F")
    return g


def figure2(m, out_base, iso_pct, zmax, elev, azim, opacity, px,
            iso_zoom=1.15, colour_by="z", z_clim="unit"):
    """Isosurfaces of u'/u_tau = +/- c, one panel per method.

    Every panel shares: camera position, domain limits, aspect ratio (true
    physical spacing), isosurface threshold, opacity and the +/- colour
    convention.  Nothing is autoscaled per method.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True
    Lx, Ly, Lz = float(m["Lx"]), float(m["Ly"]), float(m["Lz"])
    nz, ny, nx = m["truth"].shape
    dz, dy, dx = Lz / nz, Ly / ny, Lx / nx
    kmax = nz if zmax is None else int(np.ceil(zmax / dz))
    kmax = int(min(nz, max(4, kmax)))
    zt = kmax * dz
    zc = _colour_top(z_clim, Lz, zt)      # top of the COLOUR scale

    # threshold fixed ONCE from the ground truth, reused unchanged everywhere
    c = float(np.percentile(np.abs(m["truth"][:kmax]), iso_pct))
    print(f"isosurface threshold c = {c:.4f} u_tau  "
          f"(|u'| percentile {iso_pct:g} of the GROUND TRUTH over "
          f"z/h <= {zt:.3f}); identical for all methods", flush=True)

    # one camera, computed once from the shared bounds
    ctr = np.array([Lx / 2, Ly / 2, zt / 2])
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    R = 2.35 * Lx
    pos = ctr + R * np.array([np.cos(e) * np.cos(a),
                              np.cos(e) * np.sin(a), np.sin(e)])
    cpos = [tuple(pos), tuple(ctr), (0.0, 0.0, 1.0)]
    bounds = (0.0, Lx, 0.0, Ly, 0.0, zt)

    # One parallel scale, derived once from the shared bounds and then
    # forced on every panel, so "same camera" is enforced rather than hoped
    # for.  Parallel projection means no perspective foreshortening differs
    # between panels either.
    pscale = [None]
    shots, counts = {}, {}
    for key, name, nk in METHODS:
        g = _grid(m[key][:kmax], dx, dy, dz)
        pl = pv.Plotter(off_screen=True, window_size=(px, int(px * 0.46)))
        pl.set_background("white")
        n_cells = 0
        for level, colour, cmap in ((+c, ISO_POS, ISO_CMAP_POS),
                                    (-c, ISO_NEG, ISO_CMAP_NEG)):
            surf = g.contour([level], scalars="u")
            if surf.n_points == 0:
                continue
            n_cells += surf.n_cells
            _add_iso(pl, surf, colour_by, colour, cmap, zc, opacity)
        counts[key] = n_cells
        pl.add_mesh(pv.Box(bounds).outline(), color="#999999", line_width=1.2)
        pl.camera_position = cpos
        pl.enable_parallel_projection()
        pl.enable_depth_peeling(number_of_peels=8)
        if pscale[0] is None:
            pl.reset_camera(bounds=bounds)
            pscale[0] = float(pl.camera.parallel_scale) / iso_zoom
        pl.camera_position = cpos
        pl.camera.parallel_scale = pscale[0]
        shots[key] = pl.screenshot(return_img=True)
        pl.close()

    # Trim the dead canvas margin.  The crop box is the UNION of the four
    # panels' content bounds, so every panel is cropped identically -- this is
    # framing, not per-method cropping.
    def _bbox(img):
        nz_ = np.argwhere((img < 250).any(axis=2))
        return (nz_[:, 0].min(), nz_[:, 0].max(),
                nz_[:, 1].min(), nz_[:, 1].max()) if len(nz_) else None
    bxs = [b for b in (_bbox(v) for v in shots.values()) if b is not None]
    if bxs:
        r0 = max(0, min(b[0] for b in bxs) - 6)
        r1 = min(next(iter(shots.values())).shape[0],
                 max(b[1] for b in bxs) + 7)
        c0 = max(0, min(b[2] for b in bxs) - 6)
        c1 = min(next(iter(shots.values())).shape[1],
                 max(b[3] for b in bxs) + 7)
        shots = {k: v[r0:r1, c0:c1] for k, v in shots.items()}

    # Inch-based layout.  The panels take the top band; the colour key sits
    # below with enough pitch that the two sign captions never touch and the
    # z/h numerals and label clear the lower ramp at publication font sizes.
    PANEL_H = 1.00          # in, the row of four renders
    RAMP_H = 0.155          # in, one colour ramp
    RAMP_PITCH = 0.235      # in, ramp centre to ramp centre
    KEY_GAP = 0.26          # in, panels to the upper ramp
    AXIS_H = 0.62           # in, numerals + "z/h" under the lower ramp
    TOP_PAD = 0.30          # in, room for the panel titles
    h = TOP_PAD + PANEL_H + KEY_GAP + RAMP_PITCH + RAMP_H + AXIS_H
    fig = plt.figure(figsize=(7.4, h))

    def _f(inches_from_bottom):
        return inches_from_bottom / h

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

    if colour_by == "z":
        # two stacked ramps, one per sign, sharing a single z/h axis: the hue
        # family still says which sign a surface is, the lightness says how
        # far it sits from the wall
        mk = _slab_mark(zc, zt)
        y_pos = AXIS_H + RAMP_PITCH
        y_neg = AXIS_H
        _z_colorbar(fig, [0.400, _f(y_pos), 0.210, _f(RAMP_H)], ISO_CMAP_POS,
                    zc, mark=mk, ticks=False)
        _z_colorbar(fig, [0.400, _f(y_neg), 0.210, _f(RAMP_H)], ISO_CMAP_NEG,
                    zc, mark=mk, ticks=True, label="$z/h$")
        fig.text(0.388, _f(y_pos + 0.5 * RAMP_H), f"$u'/u_\\tau=+{c:.2f}$",
                 ha="right", va="center", fontsize=FS["annotation"])
        fig.text(0.388, _f(y_neg + 0.5 * RAMP_H), f"$u'/u_\\tau=-{c:.2f}$",
                 ha="right", va="center", fontsize=FS["annotation"])
    else:
        fig.legend(handles=[
            Line2D([], [], marker="s", ls="none", ms=9, color=ISO_POS,
                   label=f"$u'/u_\\tau = +{c:.2f}$"),
            Line2D([], [], marker="s", ls="none", ms=9, color=ISO_NEG,
                   label=f"$u'/u_\\tau = -{c:.2f}$")],
            loc="lower center", ncol=2, frameon=False, handletextpad=0.5,
            columnspacing=2.2, fontsize=FS["annotation"],
            bbox_to_anchor=(0.5, _f(0.10)))

    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 400})):
        q = f"{out_base}.{ext}"
        fig.savefig(q, **kw)
        print("wrote", q, flush=True)
    plt.close(fig)
    return {"iso_threshold_c_utau": c, "iso_percentile": iso_pct,
            "threshold_source": "90th-percentile-style quantile of |u'| in "
                                "the GROUND TRUTH only, over the plotted "
                                "slab; reused unchanged for every method",
            "z_range_h": [0.0, zt], "k_max": kmax,
            "z_plus_max": zt * float(m["re_tau"]),
            "camera_position": [list(map(float, t)) for t in cpos],
            "elev_deg": elev, "azim_deg": azim, "opacity": opacity,
            "colour_by": colour_by,
            "colours": ({"positive_cmap": "jet 1.00-0.68",
                         "negative_cmap": "jet 0.00-0.35",
                         "clim_z_over_h": [0.0, zc],
                         "clim_is": ("the plotted slab, so the scale "
                                     "depends on the crop"
                                     if z_clim == "slab" else
                                     "an absolute z/h, not the plotted "
                                     "slab, so one colour means one "
                                     "height in every figure"),
                         "_why": "wall distance ramps the lightness inside "
                                 "each sign's hue family, so the sign still "
                                 "reads and the crowded near-wall slab gains "
                                 "a depth cue"}
                        if colour_by == "z"
                        else {"positive": ISO_POS, "negative": ISO_NEG}),
            "renderer": f"pyvista {__import__('pyvista').__version__} "
                        f"(off-screen VTK), marching cubes via contour()",
            "isosurface_cells_per_method": counts,
            "screenshot_px": px, "parallel_scale": pscale[0],
            "iso_zoom": iso_zoom}


# ------------------------------------------------------------------ Q figure
def velocity_gradient(uvw, dx, dy, dz):
    """Jacobian J[i][j] = d u_i / d x_j of the FULL velocity field.

    uvw is (3, z, y, x) in u_tau; lengths in h.  x and y are periodic, so
    their derivatives use a periodic second-order central difference.  z is
    wall-bounded, so np.gradient's one-sided ends are used there.

    CAVEAT, stated rather than hidden: LESGO stores u,v on the uv-grid and w
    on the w-grid, offset by dz/2.  This treats them as co-located, which is
    the usual shortcut for visualisation but is a half-cell error in the
    w-derivatives.  It is applied identically to all four fields, so it
    cannot favour any method.
    """
    def d_per(f, axis, h):                      # periodic central difference
        return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * h)

    J = [[None] * 3 for _ in range(3)]
    for i in range(3):
        f = uvw[i]
        J[i][0] = d_per(f, 2, dx)               # d/dx
        J[i][1] = d_per(f, 1, dy)               # d/dy
        J[i][2] = np.gradient(f, dz, axis=0)    # d/dz, one-sided at the walls
    return J


def q_criterion(uvw, dx, dy, dz):
    """Q = -1/2 * J_ij J_ji  ==  1/2 (|Omega|^2 - |S|^2).  Units (u_tau/h)^2."""
    J = velocity_gradient(uvw, dx, dy, dz)
    Q = np.zeros_like(uvw[0])
    for i in range(3):
        for j in range(3):
            Q -= 0.5 * J[i][j] * J[j][i]
    return Q


def figure_q(m, out_base, q_pct, zmax, elev, azim, opacity, px,
             iso_zoom=1.15, colour_by="z", z_clim="unit"):
    """Q-criterion isosurfaces under exactly the constraints of figure 2.

    One threshold, fixed once from a percentile of the POSITIVE part of Q in
    the ground truth, reused unchanged.  Same camera, limits, aspect, opacity
    and colour for every method.  The per-method Q distribution is returned so
    the threshold can be judged rather than trusted.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True
    if "truth_uvw" not in m:
        raise SystemExit("this export has no (u,v,w); re-run plot_recon3d.py")
    Lx, Ly, Lz = float(m["Lx"]), float(m["Ly"]), float(m["Lz"])
    nc, nz, ny, nx = m["truth_uvw"].shape
    dz, dy, dx = Lz / nz, Ly / ny, Lx / nx
    kmax = nz if zmax is None else int(np.ceil(zmax / dz))
    kmax = int(min(nz, max(4, kmax)))
    zt = kmax * dz
    zc = _colour_top(z_clim, Lz, zt)      # top of the COLOUR scale

    # Q from the FULL velocity: fluctuation plus the training mean profile.
    # The mean profile is identical for every method (it is never
    # reconstructed), so it is a common additive term, but it must be there:
    # the cross terms between mean shear and fluctuation are a real part of Q.
    mprof = m["mean_profile_uvw"][:, :, None, None]
    Qs, stats = {}, {}
    for key, name, nk in METHODS:
        full = m[f"{key}_uvw"] + mprof
        q = q_criterion(full, dx, dy, dz)[:kmax]
        Qs[key] = q
        pos = q[q > 0]
        stats[name] = {
            "max": float(q.max()), "min": float(q.min()),
            "positive_fraction": float((q > 0).mean()),
            "p90_of_positive": float(np.percentile(pos, 90)) if pos.size else None,
            "p99_of_positive": float(np.percentile(pos, 99)) if pos.size else None,
            "rms": float(np.sqrt((q ** 2).mean()))}

    qpos = Qs["truth"][Qs["truth"] > 0]
    c = float(np.percentile(qpos, q_pct))
    print(f"Q threshold = {c:.4f} (u_tau/h)^2  "
          f"(percentile {q_pct:g} of POSITIVE Q in the GROUND TRUTH over "
          f"z/h <= {zt:.3f}); identical for all methods", flush=True)
    for k, v in stats.items():
        print(f"    {k:12s} Qmax {v['max']:9.1f}  rms {v['rms']:8.1f}  "
              f"Q>0 {100*v['positive_fraction']:5.1f}%  "
              f"p99+ {v['p99_of_positive']:9.1f}", flush=True)

    ctr = np.array([Lx / 2, Ly / 2, zt / 2])
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    R = 2.35 * Lx
    pos = ctr + R * np.array([np.cos(e) * np.cos(a),
                              np.cos(e) * np.sin(a), np.sin(e)])
    cpos = [tuple(pos), tuple(ctr), (0.0, 0.0, 1.0)]
    bounds = (0.0, Lx, 0.0, Ly, 0.0, zt)

    pscale, shots, counts = [None], {}, {}
    for key, name, nk in METHODS:
        g = _grid(Qs[key], dx, dy, dz)
        pl = pv.Plotter(off_screen=True, window_size=(px, int(px * 0.46)))
        pl.set_background("white")
        surf = g.contour([c], scalars="u")
        counts[key] = int(surf.n_cells)
        if surf.n_points:
            # colouring by u' would confound two quantities; wall distance is
            # a coordinate, not a second field, so it adds a depth cue to a
            # crowded slab without making the figure about anything else
            _add_iso(pl, surf, colour_by, Q_COLOUR, Q_CMAP, zc, opacity)
        pl.add_mesh(pv.Box(bounds).outline(), color="#999999", line_width=1.2)
        pl.camera_position = cpos
        pl.enable_parallel_projection()
        pl.enable_depth_peeling(number_of_peels=8)
        if pscale[0] is None:
            pl.reset_camera(bounds=bounds)
            pscale[0] = float(pl.camera.parallel_scale) / iso_zoom
        pl.camera_position = cpos
        pl.camera.parallel_scale = pscale[0]
        shots[key] = pl.screenshot(return_img=True)
        pl.close()

    def _bbox(img):
        nz_ = np.argwhere((img < 250).any(axis=2))
        return (nz_[:, 0].min(), nz_[:, 0].max(),
                nz_[:, 1].min(), nz_[:, 1].max()) if len(nz_) else None
    bxs = [b for b in (_bbox(v) for v in shots.values()) if b is not None]
    if bxs:
        h, w = next(iter(shots.values())).shape[:2]
        r0, r1 = max(0, min(b[0] for b in bxs) - 6), min(h, max(b[1] for b in bxs) + 7)
        c0, c1 = max(0, min(b[2] for b in bxs) - 6), min(w, max(b[3] for b in bxs) + 7)
        shots = {k: v[r0:r1, c0:c1] for k, v in shots.items()}

    # Deliberately bare: the only text is which model each panel is.  The
    # threshold, domain, snapshot and NMSE all live in the provenance JSON
    # and belong in the caption of the paper, not burned into the image.
    PANEL_H, RAMP_H, KEY_GAP, AXIS_H, TOP_PAD = 1.00, 0.155, 0.26, 0.62, 0.30
    if colour_by != "z":
        RAMP_H = KEY_GAP = AXIS_H = 0.0
    fig_h = TOP_PAD + PANEL_H + KEY_GAP + RAMP_H + AXIS_H
    fig = plt.figure(figsize=(7.4, fig_h))

    def _f(inches_from_bottom):
        return inches_from_bottom / fig_h

    bot = AXIS_H + RAMP_H + KEY_GAP
    gs = GridSpec(1, 4, figure=fig, wspace=0.012,
                  left=0.004, right=0.996,
                  top=_f(bot + PANEL_H), bottom=_f(bot))
    for i, (key, name, nk) in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(shots[key], interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(name, pad=5, fontsize=FS["panel_title"])

    if colour_by == "z":
        _z_colorbar(fig, [0.400, _f(AXIS_H), 0.210, _f(RAMP_H)], Q_CMAP, zc,
                    mark=_slab_mark(zc, zt), ticks=True,
                    label="$z/h$")

    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 400})):
        q = f"{out_base}.{ext}"
        fig.savefig(q, **kw)
        print("wrote", q, flush=True)
    plt.close(fig)
    return {"q_threshold": c, "q_percentile_of_positive": q_pct,
            "units": "(u_tau/h)^2", "z_range_h": [0.0, zt],
            "distribution_per_method": stats,
            "surface_cells_per_method": counts,
            "gradient_scheme": "periodic 2nd-order central in x and y, "
                               "np.gradient in z (one-sided at the walls)",
            "staggering_caveat": "u,v on the uv-grid and w on the w-grid are "
                                 "treated as co-located; a half-cell error in "
                                 "the w-derivatives, applied identically to "
                                 "all four fields",
            "colour_by": colour_by,
            "colour": ({"cmap": "jet",
                        "clim_z_over_h": [0.0, zc],
                        "clim_is": ("the plotted slab, so the scale "
                                    "depends on the crop"
                                    if z_clim == "slab" else
                                    "an absolute z/h, not the plotted "
                                    "slab, so one colour means one "
                                    "height in every figure"),
                        "_why": "wall distance, a coordinate rather than a "
                                "second field, so the surfaces sort front "
                                "from back without the figure becoming about "
                                "anything but vortices"}
                       if colour_by == "z" else Q_COLOUR),
            "opacity": opacity}


# ------------------------------------------------------------------ figure 3
def figure3(m, out_base, abs_pct):
    """Optional diagnostic: |reconstruction - truth| on the same plane.

    One shared absolute-error scale for POD, CNN and OPINE, sequential
    colormap, no per-method autoscaling.  Kept apart from the paper figures.
    """
    iz = int(m["iz"])
    Lx, Ly = float(m["Lx"]), float(m["Ly"])
    nyy, nxx = m["truth"].shape[1], m["truth"].shape[2]
    xc = (np.arange(nxx) + 0.5) * Lx / nxx
    yc = (np.arange(nyy) + 0.5) * Ly / nyy
    gt = m["truth"][iz]
    ae = {k: np.abs(m[k][iz] - gt) for k in ("pod", "cnn", "opine")}
    A = float(np.percentile(np.concatenate([v.ravel() for v in ae.values()]),
                            abs_pct))

    fig = plt.figure(figsize=(6.9, 1.95), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.012, h_pad=0.006, wspace=0.02)
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.06])
    im = None
    for i, (k, name) in enumerate((("pod", "POD"), ("cnn", "CNN"),
                                   ("opine", "OPINE"))):
        ax = fig.add_subplot(gs[0, i])
        im = ax.pcolormesh(xc, yc, ae[k], cmap="magma", vmin=0.0, vmax=A,
                           shading="nearest", rasterized=True)
        ax.set_aspect("equal"); ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
        ax.set_xlabel("$x/h$")
        ax.set_title(f"{name}   NMSE "
                     f"{float(m['nmse_snapshot_' + k]):.3f}", pad=3)
        if i == 0:
            ax.set_ylabel("$y/h$")
        else:
            ax.set_yticklabels([])
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]))
    cb.set_label(r"$|\hat{u}' - u'|\ (/u_\tau)$", labelpad=1)
    cb.outline.set_linewidth(LW["outline"])
    cb.ax.tick_params(labelsize=FS["tick"], width=LW["tick"])
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        q = f"{out_base}.{ext}"
        fig.savefig(q, **kw)
        print("wrote", q, flush=True)
    plt.close(fig)
    return {"abs_error_limit_A": A, "abs_error_percentile": abs_pct,
            "colormap": "magma", "plane_index_k": iz}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="explicit snapshot export to draw")
    ap.add_argument("--figdir", default=None,
                    help="directory holding recon3d_D*_snapshot*.npz")
    ap.add_argument("--dof", type=int, default=256)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--velocity-percentile", type=float, default=99.0)
    ap.add_argument("--error-percentile", type=float, default=99.0)
    ap.add_argument("--iso-percentile", type=float, default=90.0)
    ap.add_argument("--iso-zmax", type=float, default=0.40,
                    help="restrict the isosurface box to z/h <= this")
    ap.add_argument("--elev", type=float, default=24.0)
    ap.add_argument("--azim", type=float, default=-62.0)
    ap.add_argument("--iso-opacity", type=float, default=1.0)
    ap.add_argument("--iso-zoom", type=float, default=1.15)
    ap.add_argument("--iso-px", type=int, default=1500,
                    help="off-screen render width per 3-D panel")
    ap.add_argument("--abs-error-percentile", type=float, default=99.5)
    ap.add_argument("--iso-colour", choices=("z", "flat"), default="z",
                    help="ramp the isosurfaces by wall distance (default) "
                         "or use the old flat per-sign colours")
    ap.add_argument("--iso-z-clim", default="0.5",
                    help="top of the z/h colour scale: an explicit z/h "
                         "(default 0.5), 'unit' for the channel half-height "
                         "or 'slab' for the plotted slab")
    ap.add_argument("--skip-iso", action="store_true")
    ap.add_argument("--skip-abs", action="store_true")
    ap.add_argument("--q-percentile", type=float, default=90.0,
                    help="percentile of POSITIVE Q in the ground truth")
    ap.add_argument("--skip-q", action="store_true")
    a = ap.parse_args()

    npz = a.npz
    if npz is None:
        pat = os.path.join(a.figdir, f"recon3d_D{a.dof}_snapshot*.npz")
        hits = sorted(glob.glob(pat))
        if not hits:
            sys.exit(f"no export matching {pat}; run plot_recon3d.py first")
        npz = hits[0]
    print("reading", npz, flush=True)
    m = load(npz)
    os.makedirs(a.outdir, exist_ok=True)
    D = int(m["dof"])
    rep = {"source_npz": npz, "dof": D,
           "_dof_is": ("the RETAINED dimension the summary figure plots; "
                       "stored_rank below says what each method actually "
                       "holds, which differs for POD and OPINE because their "
                       "modes come in pairs"),
           "stored_rank": ({k: int(m[f"stored_rank_{k}"])
                            for k in ("pod", "opine", "cnn")}
                           if "stored_rank_pod" in m else
                           {"pod": D, "opine": D, "cnn": D}),
           "pair_divisor": int(m["pair_divisor"]) if "pair_divisor" in m else 1,
           "snapshot_index": int(m["snapshot_index"]),
           "opine_variant": str(m["opine_variant"]),
           "nmse_snapshot": {"POD": float(m["nmse_snapshot_pod"]),
                             "CNN": float(m["nmse_snapshot_cnn"]),
                             "OPINE": float(m["nmse_snapshot_opine"])},
           "nmse_split": {"POD": float(m["nmse_split_pod"]),
                          "CNN": float(m["nmse_split_cnn"]),
                          "OPINE": float(m["nmse_split_opine"])}}

    rep["figure1"] = figure1(
        m, os.path.join(a.outdir, f"fig1_recon3d_D{D}_uprime_wallplane"),
        a.velocity_percentile, a.error_percentile)
    if not a.skip_iso:
        rep["figure2"] = figure2(
            m, os.path.join(a.outdir, f"fig2_recon3d_D{D}_uprime_isosurface"),
            a.iso_percentile, a.iso_zmax, a.elev, a.azim,
            a.iso_opacity, a.iso_px, a.iso_zoom, a.iso_colour,
            a.iso_z_clim)
    if not a.skip_q and "truth_uvw" in m:
        rep["figure_q"] = figure_q(
            m, os.path.join(a.outdir,
                            f"fig3_recon3d_D{D}_qcriterion_p{a.q_percentile:g}"),
            a.q_percentile, a.iso_zmax, a.elev, a.azim, a.iso_opacity,
            a.iso_px, a.iso_zoom, a.iso_colour, a.iso_z_clim)
    elif not a.skip_q:
        print("no (u,v,w) in this export: skipping the Q-criterion figure",
              flush=True)
    if not a.skip_abs:
        rep["figure3_diagnostic"] = figure3(
            m, os.path.join(a.outdir, f"diag_recon3d_D{D}_uprime_abserror"),
            a.abs_error_percentile)

    p = os.path.join(a.outdir, f"paper_figures_D{D}_provenance.json")
    json.dump(rep, open(p, "w"), indent=1)
    print("wrote", p, flush=True)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
