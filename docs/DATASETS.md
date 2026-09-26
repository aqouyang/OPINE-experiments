# Datasets

None is distributed with this repository.

## 1. 2-D Kolmogorov flow — generated here

`KOLMOGOROV_ROOT` should hold `raw/`, `splits.json` and `results/`.

Fully reproducible from this repository. `provenance/kolmogorov/metadata.json`
records every setting; the short form:

| | |
|---|---|
| equation | $\partial_t\omega + u\cdot\nabla\omega = \nu\nabla^2\omega - k_f\cos(k_f y)$ |
| Re, $\nu$, $k_f$ | 100, 0.01, 4 |
| domain, grid | $2\pi\times2\pi$ doubly periodic, $128^2$ |
| solver | `jax_cfd.spectral.equations.NavierStokes2D`, 2/3 dealiasing |
| integrator | `crank_nicolson_rk4` |
| timestep | $1/512$; 512 steps per snapshot, so spacing is exactly one advective time unit |
| burn-in | 100 time units (51 200 steps), discarded |
| size | 100 trajectories x 1000 snapshots = 100 000 fields, float32, ~6.2 GB |
| split | **trajectory level** 80/10/10 -> 80 000 / 10 000 / 10 000 |

The split is at trajectory level on purpose: snapshots are one time unit apart
and the integral correlation time is 9.7 (dissipation) to 36 (energy), so a
snapshot-level split would leak.

```bash
python -m opine_experiments.kolmogorov.data_generation.generate_trajectories --output-dir $KOLMOGOROV_ROOT/raw
python -m opine_experiments.kolmogorov.data_generation.make_splits --raw-dir $KOLMOGOROV_ROOT/raw --output $KOLMOGOROV_ROOT/splits.json
python -m opine_experiments.kolmogorov.data_generation.validate_pilot --raw-dir $KOLMOGOROV_ROOT/raw --output-dir $KOLMOGOROV_ROOT/validation
```

`validate_pilot` reproduces the checks in
`provenance/kolmogorov/validation_report.json`: zero mean vorticity, stationary
after burn-in, spectrum decayed before the dealiasing cutoff.

## 2. y-z channel cross-sections — external DNS

`CHANNEL2D_DATA` should hold `train.npz`, `val.npz`, `test.npz` and
`preprocessing_yz_joint.json`.

Extracted from a $Re_\tau=180$ LESGO half-channel database that is **not ours
and is not redistributed here**. Obtain it from the authors of that database.

| | |
|---|---|
| source grid | $256\times128\times64$, float64, Fortran order |
| domain | $L_x=2\pi$, $L_y=\pi$, $L_z=1$; half channel, wall at $z=0$ |
| wall-normal grid | uniform, $z_k=(k+\tfrac12)/64$, $z^+=180z$ |
| sampling | DNS step 600 000 to 4 800 000 every 1000 -> 4201 snapshots |
| slice | fixed $x$ (ix = 0), three components -> shape $(3,128,64)$ |
| split | 2940 / 630 / 631, **random at snapshot level** |
| size | ~790 MB |

The split is random over snapshots here, unlike the other two datasets. No
decorrelation time at the 1000-step sampling interval is recorded, so we do not
claim the split is leak-free; it is stated as what it is.

Normalisation: per component and per wall-normal level, subtract the mean
profile and divide by the rms, both computed on the **training split only**.

## 3. 3-D minimal channel — external

`CHANNEL3D_ROOT` should hold `channel_retau180_nz64.npy` (15.7 GB, memory
mapped) and `channel_retau180_nz64_meta.npz`.

| | |
|---|---|
| domain | $\pi\times\pi/2\times1$, the minimal flow unit |
| grid | $32\times32\times64$, staggered in $z$ |
| $Re_\tau$, $\nu$, $u_\tau$ | 180, 1/180, 1 |
| walls | no-slip floor, stress-free lid |
| size | 20 000 snapshots, spacing 0.37 $h/u_\tau$ |
| split | **chronological** 16 000 / 2000 / 2000 |
| ambient dimension | 196 608 |

The split is chronological because adjacent snapshots at 0.37 $h/u_\tau$ are
strongly correlated.

**The solver that produced this dataset is not recorded** in its metadata. The
metadata describes the configuration only. We do not state a solver we cannot
verify.

Note on `w`: it lives on the w-grid whose first node is the wall, so
`w[:, 0]` is identically zero and its training sigma is floored at 1e-8,
leaving that slab exactly zero after normalisation.
