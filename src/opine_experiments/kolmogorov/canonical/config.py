"""Canonical configuration for the Kolmogorov-flow FINE experiments.

The canonical iResNet FINE model is exactly

    x -> f_theta -> P_Q -> f_theta^{-1} -> x_hat

with f_theta a 6-block iResNet and P_Q the POD projector.  At round 0 every
residual block's last conv is exact-zero-initialized, so f_theta = I and the
model is bit-for-bit identical to POD.

Only the parameters in CanonicalIResNetConfig are exposed on the training
interface.  Everything in FROZEN is part of the canonical setting and is not
a tuning knob -- changing one means you are no longer running the canonical
model and the result does not belong in the canonical benchmark table.

Nothing diagnostic lives here.  Probe lambda, decoder size, gradient
diagnostics and completion size are in
kolmogorov/diagnostics/config.py in the research repository.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalIResNetConfig:
    """The complete canonical training interface."""
    dataset: str              # root holding raw/, splits.json, results/
    dof: int                  # retained degrees of freedom (32/64/128/256)
    n_blocks: int = 6         # residual blocks in f_theta
    contraction_bound: float = 0.9   # per-conv spectral-norm cap
    lr: float = 1e-3
    rounds: int = 25
    batch_size: int = 64
    seed: int = 42


# Fixed part of the canonical setting.  Not exposed, not swept.
FROZEN = {
    "grid": 128,              # 128 x 128 vorticity field, 1 channel
    "hidden": 32,             # width inside each residual branch
    "lipschitz": 0.6,         # residual scale s; with contraction_bound=0.9
                              # this is the certified contraction setting
    "alpha": 0.03,            # subspace update rate
    "inner_epochs": 1,
    "grad_clip": 1.0,
    "inverse_iters": 40,      # Banach fixed-point steps for f^{-1}
    "grad_iters": 3,          # truncated Neumann terms in the implicit grad
    "subspace_power": 1,
    "latent_subsample": 20000,
    "diag_warm_iters": 30,
    "roundtrip_tol": 1e-8,
    "contraction_tol": 0.98,
    "num_workers": 8,
    "pad_mode": "circular_both",   # doubly-periodic domain
    "bottleneck": "unrestricted",
}

DOFS = (32, 64, 128, 256)


def dataset_paths(root):
    """Canonical layout of a dataset root."""
    return {
        "raw": os.path.join(root, "raw"),
        "splits": os.path.join(root, "splits.json"),
        "pod_basis": os.path.join(root, "results", "pod", "pod_basis.npz"),
    }


def canonical_nmse(recon, truth):
    """The project-wide metric: sum|recon-truth|^2 / sum|truth|^2."""
    return float(((recon - truth) ** 2).sum() / (truth ** 2).sum())
