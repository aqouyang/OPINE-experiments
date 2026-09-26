"""Alternative invertible monotonic activation layers for the FINE K-layer.

This module provides drop-in replacements for `MonotonicPiecewiseLinear`
(the baseline smooth_relu-hinge layer in models/one_d/FINE.py), used by the
activation-comparison experiment:

    baseline : MonotonicPiecewiseLinear  (cubic-smoothstep hinge; in FINE.py)
    leaky    : LeakyMonotonicPiecewiseLinear  (sharp piecewise-linear, C^0)
    spline   : RQSplineActivation  (rational-quadratic monotonic spline, C^1)

Design contract shared by all three (so they are apples-to-apples):
  * scalar map phi: [0,1] -> [0,1], applied pointwise to the signal;
  * strictly monotonic increasing  => invertible;
  * same normalization front-end ('minmax' or 'sigmoid') as the baseline;
  * forward(x)   = monotone_map(normalize(x))           (returns values in [0,1])
  * inverse(y)   = denormalize(monotone_map_inverse(y))
  * a `.inverse` attribute that is itself a callable nn.Module, matching the
    baseline's external `kan.inverse = Inverse...(kan)` pattern, so FINE.forward
    needs no change.

The leaky arm is C^0 (corners) and is the deliberate "non-smooth but invertible"
control; the spline arm is C^1 with an analytic inverse and is the clean
solution. See experiments/activation_comparison/CLAUDE.md for the rationale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from opine_experiments.models.one_d.FINE import MinMaxNormalizer, InvertibleSigmoidNorm


# ---------------------------------------------------------------------------
#  Shared normalization front-end helpers
# ---------------------------------------------------------------------------
def _make_normalizer(normalization):
    if normalization == "minmax":
        return ("minmax", MinMaxNormalizer())
    elif normalization == "sigmoid":
        return ("sigmoid", InvertibleSigmoidNorm())
    else:
        raise ValueError(f"unknown normalization: {normalization}")


def _normalize(kind, normalizer, x):
    if kind == "minmax":
        return normalizer.normalize(x)
    return normalizer(x)          # sigmoid: nn.Module.__call__ -> forward


def _denormalize(kind, normalizer, x):
    if kind == "minmax":
        return normalizer.reverse(x)
    return normalizer.inverse(x)  # sigmoid: logit


def _normalized_y_control(raw_y_control):
    """softplus + cumsum -> strictly increasing controls normalized to [0,1]."""
    y = torch.cumsum(F.softplus(raw_y_control), dim=0)
    y = y - y[0]
    y = y / y[-1]
    return y


# ===========================================================================
#  LEAKY arm: sharp (C^0) piecewise-linear monotonic interpolant
# ===========================================================================
class LeakyMonotonicPiecewiseLinear(nn.Module):
    """Exact piecewise-linear monotone map through (x_control, y_control).

    Same control-point parametrization as the baseline, but the corners are
    *sharp* (no smoothstep): the map is only C^0. Because softplus+cumsum makes
    y_control strictly increasing, every segment slope is strictly positive, so
    the map is strictly monotonic and exactly invertible (no Newton needed) ---
    this is the "leaky-ReLU-style" control: invertible but non-smooth.
    """

    def __init__(self, num_points, y_control_init, normalization="sigmoid"):
        super().__init__()
        assert num_points == len(y_control_init)
        self.num_points = num_points
        self.x_control = nn.Parameter(torch.linspace(0, 1, num_points),
                                      requires_grad=False)
        self.raw_y_control = nn.Parameter(y_control_init.clone())
        self.norm_kind, normalizer = _make_normalizer(normalization)
        if self.norm_kind == "minmax":
            self.normalizer = normalizer
        else:
            self.sigmoid = normalizer
        self.inverse = _LeakyInverse(self)

    def _normalizer(self):
        return self.normalizer if self.norm_kind == "minmax" else self.sigmoid

    def _pl(self, x, x_ctrl, y_ctrl):
        """Vectorized sharp piecewise-linear interpolation, x in [0,1]."""
        P = self.num_points
        xc = x.clamp(0.0, 1.0)
        # uniform grid => bin = floor(x * (P-1)), clamped to [0, P-2]
        idx = torch.floor(xc * (P - 1)).long().clamp(0, P - 2)
        x0 = x_ctrl[idx]
        x1 = x_ctrl[idx + 1]
        y0 = y_ctrl[idx]
        y1 = y_ctrl[idx + 1]
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (xc - x0)

    def forward(self, x):
        y_ctrl = _normalized_y_control(self.raw_y_control)
        xn = _normalize(self.norm_kind, self._normalizer(), x)
        return self._pl(xn, self.x_control, y_ctrl)


class _LeakyInverse(nn.Module):
    """Exact inverse of LeakyMonotonicPiecewiseLinear: swap (x,y) controls."""

    def __init__(self, model):
        super().__init__()
        # Store parent without registering as submodule (avoids circular graph)
        object.__setattr__(self, 'model', model)

    def forward(self, y):
        m = self.model
        y_ctrl = _normalized_y_control(m.raw_y_control)
        # invert the PL map: interpolate x_control as a function of y_ctrl
        P = m.num_points
        yc = y.clamp(0.0, 1.0)
        # y_ctrl is strictly increasing but NOT uniform -> use searchsorted
        idx = (torch.searchsorted(y_ctrl.contiguous(), yc.contiguous(),
                                  right=True) - 1).clamp(0, P - 2)
        y0 = y_ctrl[idx]
        y1 = y_ctrl[idx + 1]
        x0 = m.x_control[idx]
        x1 = m.x_control[idx + 1]
        slope = (x1 - x0) / (y1 - y0)
        x_in = x0 + slope * (yc - y0)
        return _denormalize(m.norm_kind, m._normalizer(), x_in.clamp(1e-6, 1 - 1e-6))


# ===========================================================================
#  SPLINE arm: monotonic rational-quadratic spline (Durkan et al., NeurIPS 2019)
# ===========================================================================
class RQSplineActivation(nn.Module):
    """Monotone rational-quadratic spline on [0,1] -> [0,1].

    C^1, strictly monotonic by construction (positive widths/heights and
    positive knot derivatives), with a closed-form inverse obtained by solving a
    quadratic. Reference: Durkan, Bekasov, Murray, Papamakarios,
    "Neural Spline Flows", NeurIPS 2019, eqs. (4)-(8).
    """

    def __init__(self, num_bins=19, normalization="sigmoid", min_bin=1e-3,
                 min_deriv=1e-3):
        super().__init__()
        self.num_bins = num_bins
        self.min_bin = min_bin
        self.min_deriv = min_deriv
        # near-identity init: uniform widths/heights, unit derivatives
        self.unnorm_widths = nn.Parameter(torch.zeros(num_bins))
        self.unnorm_heights = nn.Parameter(torch.zeros(num_bins))
        inv_softplus_1 = torch.log(torch.expm1(torch.tensor(1.0)))
        self.unnorm_derivs = nn.Parameter(torch.full((num_bins + 1,),
                                                      float(inv_softplus_1)))
        self.norm_kind, normalizer = _make_normalizer(normalization)
        if self.norm_kind == "minmax":
            self.normalizer = normalizer
        else:
            self.sigmoid = normalizer
        self.inverse = _RQSplineInverse(self)

    def _normalizer(self):
        return self.normalizer if self.norm_kind == "minmax" else self.sigmoid

    def _params(self):
        """Return knot x-edges, y-edges, and derivatives (all 1-D tensors)."""
        K = self.num_bins
        widths = F.softmax(self.unnorm_widths, dim=0)
        widths = self.min_bin + (1 - self.min_bin * K) * widths
        heights = F.softmax(self.unnorm_heights, dim=0)
        heights = self.min_bin + (1 - self.min_bin * K) * heights
        derivs = self.min_deriv + F.softplus(self.unnorm_derivs)
        knots_x = torch.cat([torch.zeros(1, device=widths.device),
                             torch.cumsum(widths, dim=0)])
        knots_y = torch.cat([torch.zeros(1, device=heights.device),
                             torch.cumsum(heights, dim=0)])
        knots_x = knots_x / knots_x[-1]
        knots_y = knots_y / knots_y[-1]
        return knots_x, knots_y, derivs

    def _gather_bin(self, t, knots_a):
        """bin index of each scalar in t against monotone edges knots_a."""
        K = self.num_bins
        idx = (torch.searchsorted(knots_a.contiguous(), t.contiguous(),
                                  right=True) - 1).clamp(0, K - 1)
        return idx

    def _spline(self, x, knots_x, knots_y, derivs):
        """Evaluate the RQS using one-hot gather (avoids IndexBackward scatter)."""
        K = self.num_bins
        xc = x.clamp(0.0, 1.0)
        orig_shape = xc.shape
        xc_flat = xc.reshape(-1)

        # Bin index (returns LongTensor, no grad)
        idx = (torch.searchsorted(knots_x[1:].contiguous(), xc_flat.contiguous(),
                                  right=False)).clamp(0, K - 1)

        # One-hot gather: avoids pathological IndexBackward0 scatter kernel
        oh = F.one_hot(idx, K).to(xc_flat.dtype)  # (N, K)

        # Per-bin quantities (K,)
        w_all = knots_x[1:] - knots_x[:-1]
        h_all = knots_y[1:] - knots_y[:-1]
        s_all = h_all / w_all
        x_k_all = knots_x[:-1]
        y_k_all = knots_y[:-1]
        d_k_all = derivs[:-1]
        d_kp_all = derivs[1:]

        # Gather via broadcasting (fast backward)
        w = (oh * w_all).sum(-1)
        h = (oh * h_all).sum(-1)
        s = (oh * s_all).sum(-1)
        x_k = (oh * x_k_all).sum(-1)
        y_k = (oh * y_k_all).sum(-1)
        d_k = (oh * d_k_all).sum(-1)
        d_kp = (oh * d_kp_all).sum(-1)

        xi = ((xc_flat - x_k) / w).clamp(0.0, 1.0)
        one_m = 1.0 - xi
        num = h * (s * xi * xi + d_k * xi * one_m)
        den = s + (d_kp + d_k - 2 * s) * xi * one_m
        return (y_k + num / den).reshape(orig_shape)

    def forward(self, x):
        knots_x, knots_y, derivs = self._params()
        xn = _normalize(self.norm_kind, self._normalizer(), x)
        return self._spline(xn, knots_x, knots_y, derivs)


class _RQSplineInverse(nn.Module):
    """Closed-form inverse of the RQ spline (solve the quadratic)."""

    def __init__(self, model):
        super().__init__()
        # Store parent without registering as submodule (avoids circular graph)
        object.__setattr__(self, 'model', model)

    def forward(self, y):
        m = self.model
        K = m.num_bins
        knots_x, knots_y, derivs = m._params()
        yc = y.clamp(0.0, 1.0)
        orig_shape = yc.shape
        yc_flat = yc.reshape(-1)

        # Bin index in y-space
        idx = (torch.searchsorted(knots_y[1:].contiguous(), yc_flat.contiguous(),
                                  right=False)).clamp(0, K - 1)

        # One-hot gather (avoids IndexBackward scatter)
        oh = F.one_hot(idx, K).to(yc_flat.dtype)

        w_all = knots_x[1:] - knots_x[:-1]
        h_all = knots_y[1:] - knots_y[:-1]
        s_all = h_all / w_all
        x_k_all = knots_x[:-1]
        y_k_all = knots_y[:-1]
        d_k_all = derivs[:-1]
        d_kp_all = derivs[1:]

        w = (oh * w_all).sum(-1)
        h = (oh * h_all).sum(-1)
        s = (oh * s_all).sum(-1)
        x_k = (oh * x_k_all).sum(-1)
        y_k = (oh * y_k_all).sum(-1)
        d_k = (oh * d_k_all).sum(-1)
        d_kp = (oh * d_kp_all).sum(-1)

        dy = yc_flat - y_k
        # quadratic a*xi^2 + b*xi + c = 0  (Durkan et al. eq. 6-8)
        a = h * (s - d_k) + dy * (d_kp + d_k - 2 * s)
        b = h * d_k - dy * (d_kp + d_k - 2 * s)
        c = -s * dy
        disc = (b * b - 4 * a * c).clamp(min=0.0)
        xi = 2 * c / (-b - torch.sqrt(disc))
        xi = xi.clamp(0.0, 1.0)
        x_in = x_k + xi * w
        return _denormalize(m.norm_kind, m._normalizer(),
                            x_in.clamp(1e-6, 1 - 1e-6).reshape(orig_shape))


# ===========================================================================
#  Factory used by FINE
# ===========================================================================
def make_kan_layer(activation, num_points=20, normalization="sigmoid"):
    """Build the K-layer module for the requested activation arm.

    'baseline' is built by FINE itself (kept there to avoid a circular import
    surprise); this factory handles the new arms.
    """
    if activation == "leaky":
        return LeakyMonotonicPiecewiseLinear(
            num_points, torch.ones(num_points), normalization=normalization)
    elif activation == "spline":
        return RQSplineActivation(
            num_bins=num_points - 1, normalization=normalization)
    elif activation == "realline_spline":
        from opine_experiments.models.one_d.real_line_spline import RealLineRQSpline
        return RealLineRQSpline(num_bins=num_points - 1, tail_bound=3.0)
    else:
        raise ValueError(f"make_kan_layer does not handle '{activation}'")
