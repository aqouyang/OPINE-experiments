#!/usr/bin/env python3
"""Pseudospectral 2-D Kolmogorov flow on a doubly-periodic square, via JAX-CFD.

Target equation (Cleary, Wang & Zaki, arXiv:2512.15470, eq. for vorticity):

    d_t omega + u . grad omega = (1/Re) laplacian(omega) - k_f cos(k_f y)

with Re := sqrt(chi*/k*^3)/nu, so in these code units the viscosity is exactly
1/Re and the vorticity forcing coefficient is exactly k_f.

Everything below is read off the installed JAX-CFD source rather than assumed;
see README.md for the audit.  The two facts that matter:

  * jax_cfd.base.forcings.kolmogorov_forcing(grid, scale=1, k=k_f) returns the
    VELOCITY forcing (sin(k_f y), 0).  NavierStokes2D applies it through
    spectral_curl_2d, i.e. d_x f_y - d_y f_x = -k_f cos(k_f y).  That is the
    target forcing term exactly, with no sign or amplitude fudge.
  * NavierStokes2D.linear_term = viscosity * laplacian - drag.  The target
    equation has no drag, so drag=0.  (jax_cfd's own ForcedNavierStokes2D
    helper hard-codes drag=0.1 for the Kochkov et al. setup and is therefore
    NOT used here.)

Domain: the paper says "square and doubly periodic" without giving the extent.
L = 2*pi is forced by the physics rather than chosen: kolmogorov_forcing builds
sin(k_f * y) from the physical coordinate, so the forcing is only periodic on
the domain when k_f * L is a multiple of 2*pi.  With integer k_f = 4 that means
L = 2*pi.  This is also what every JAX-CFD Kolmogorov example uses.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import jax_cfd.base as cfd
import jax_cfd.spectral as spectral
from jax_cfd.base import forcings, grids

DOMAIN_LENGTH = 2.0 * np.pi


@dataclasses.dataclass(frozen=True)
class KolmogorovConfig:
    re: float = 100.0
    k_f: int = 4
    n: int = 128
    domain_length: float = DOMAIN_LENGTH
    dt: float = None                  # set explicitly; see choose_dt
    burn_in_time: float = 50.0
    n_snapshots: int = 1000
    snapshot_interval: float = 1.0    # one advective time unit
    ic_max_velocity: float = 7.0
    ic_peak_wavenumber: int = 4
    smooth: bool = True               # 2/3-rule dealiasing
    drag: float = 0.0                 # target equation has no drag
    forcing_scale: float = 1.0
    x64: bool = True

    @property
    def viscosity(self):
        return 1.0 / self.re

    @property
    def steps_per_snapshot(self):
        s = self.snapshot_interval / self.dt
        n = int(round(s))
        if abs(s - n) > 1e-9:
            raise ValueError(
                f"snapshot_interval {self.snapshot_interval} is not an integer "
                f"multiple of dt {self.dt} (ratio {s})")
        return n

    @property
    def burn_in_steps(self):
        s = self.burn_in_time / self.dt
        n = int(round(s))
        if abs(s - n) > 1e-9:
            raise ValueError("burn_in_time is not an integer multiple of dt")
        return n


def make_grid(cfg):
    return grids.Grid((cfg.n, cfg.n),
                      domain=((0, cfg.domain_length), (0, cfg.domain_length)))


def make_equation(cfg, grid):
    """NavierStokes2D with drag=0 and the k_f Kolmogorov forcing.

    offsets=((0, 0), (0, 0)) puts both velocity components at cell centres.
    The staggered default (grid.cell_faces) belongs to the finite-volume
    solver; for the pseudospectral path the collocated choice is the one
    jax_cfd's own spectral Kolmogorov setup uses.
    """
    forcing_fn = lambda g: forcings.kolmogorov_forcing(
        g, scale=cfg.forcing_scale, k=cfg.k_f, offsets=((0, 0), (0, 0)))
    return spectral.equations.NavierStokes2D(
        viscosity=cfg.viscosity,
        grid=grid,
        drag=cfg.drag,
        smooth=cfg.smooth,
        forcing_fn=forcing_fn)


def make_step_fn(cfg, grid):
    """Carpenter-Kennedy low-storage RK4 explicit / Crank-Nicolson implicit."""
    return spectral.time_stepping.crank_nicolson_rk4(
        make_equation(cfg, grid), cfg.dt)


def advection_stable_dt(cfg, grid, max_velocity, cfl=0.5):
    """JAX-CFD's own advective stability limit, for reference."""
    return cfd.equations.stable_time_step(
        max_velocity=max_velocity, max_courant_number=cfl,
        viscosity=cfg.viscosity, grid=grid, implicit_diffusion=True)


def initial_vorticity_hat(cfg, grid, seed):
    """Deterministic divergence-free initial condition for one trajectory.

    jax_cfd.initial_conditions.filtered_velocity_field draws a random
    solenoidal field with a prescribed peak wavenumber and maximum speed, then
    projects it; we take its curl to get vorticity.
    """
    key = jax.random.PRNGKey(seed)
    v0 = cfd.initial_conditions.filtered_velocity_field(
        key, grid, cfg.ic_max_velocity, cfg.ic_peak_wavenumber)
    vort0 = cfd.finite_differences.curl_2d(v0).data
    return jnp.fft.rfftn(vort0)


def vorticity_forcing_field(cfg, grid):
    """The forcing term exactly as it enters the vorticity equation.

    Evaluated by calling the equation's own explicit_terms on zero vorticity:
    with omega = 0 the advection term vanishes identically, so what comes back
    is the forcing curl and nothing else.  This exercises the real code path
    rather than re-deriving it, so it can be compared against the analytic
    -k_f cos(k_f y).  Used by the audit in validate_pilot.py.
    """
    eq = make_equation(cfg, grid)
    zero_hat = jnp.zeros_like(grid.rfft_mesh()[0], dtype=complex)
    return jnp.fft.irfftn(eq.explicit_terms(zero_hat))
