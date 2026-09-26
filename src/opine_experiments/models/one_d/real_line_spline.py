"""
Identity-initialized monotone rational-quadratic spline on the real line.

K(x) = x at initialization.  For |x| <= B the map is a learnable RQ spline;
outside [-B, B] it continues as linear tails with slope matching the
boundary derivative, guaranteeing a strictly monotone C^1 bijection
R -> R.  The inverse is computed analytically (quadratic formula inside
the spline domain, linear outside).

Mathematical definition
-----------------------
Let phi: [0,1] -> [0,1] be the standard RQS with K bins on the unit
interval.  Define the scaled spline on [-B, B]:

    S(x) = -B + 2B * phi((x + B) / (2B))

Outside the interval:

    K(x) = S(-B) + d_left  * (x - (-B))   if x < -B
    K(x) = S( B) + d_right * (x -   B )   if x >  B

where d_left and d_right are the spline derivatives at the boundaries.

At initialization phi = identity, so S(x) = x and K(x) = x everywhere.

Parameters: unnorm_widths (K,), unnorm_heights (K,), unnorm_derivs (K+1,).
All zero-initialized so that the spline starts as identity.

Forward and inverse share the same parameter tensors — no copies, no detach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RealLineRQSpline(nn.Module):
    """Identity-initialized RQ spline on the full real line with linear tails.

    Parameters
    ----------
    num_bins : int
        Number of spline segments inside [-B, B].
    tail_bound : float
        Half-width B of the spline domain.  Outside [-B, B] the map is
        linear with slope matching the boundary derivative.
    min_bin : float
        Minimum bin width/height to prevent degenerate segments.
    min_deriv : float
        Minimum positive derivative at each knot.
    """

    def __init__(self, num_bins: int = 19, tail_bound: float = 3.0,
                 min_bin: float = 1e-3, min_deriv: float = 1e-3):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin = min_bin
        self.min_deriv = min_deriv

        # All-zeros → uniform bins → identity spline
        self.unnorm_widths = nn.Parameter(torch.zeros(num_bins))
        self.unnorm_heights = nn.Parameter(torch.zeros(num_bins))
        # Initialize derivs so that min_deriv + softplus(x) = 1.0 exactly
        target_sp = 1.0 - min_deriv
        inv_sp = float(torch.log(torch.expm1(torch.tensor(target_sp))))
        self.unnorm_derivs = nn.Parameter(
            torch.full((num_bins + 1,), inv_sp))

        self.inverse = _RealLineRQSplineInverse(self)

    def perturb(self, eps: float, seed: int = 0):
        """Add symmetric-breaking perturbation: theta = theta_id + eps * xi.

        xi is drawn from N(0, 1). The spline remains strictly monotone for
        small eps (the softmax/softplus ensure positivity regardless).
        """
        rng = torch.Generator(device=self.unnorm_widths.device)
        rng.manual_seed(seed)
        with torch.no_grad():
            self.unnorm_widths.add_(
                torch.randn(self.unnorm_widths.shape, generator=rng,
                            device=self.unnorm_widths.device) * eps)
            self.unnorm_heights.add_(
                torch.randn(self.unnorm_heights.shape, generator=rng,
                            device=self.unnorm_heights.device) * eps)
            self.unnorm_derivs.add_(
                torch.randn(self.unnorm_derivs.shape, generator=rng,
                            device=self.unnorm_derivs.device) * eps)

    def _params(self):
        """Compute knot positions and derivatives on [0, 1]."""
        K = self.num_bins
        widths = F.softmax(self.unnorm_widths, dim=0)
        widths = self.min_bin + (1.0 - self.min_bin * K) * widths
        heights = F.softmax(self.unnorm_heights, dim=0)
        heights = self.min_bin + (1.0 - self.min_bin * K) * heights
        derivs = self.min_deriv + F.softplus(self.unnorm_derivs)

        knots_x = torch.cat([torch.zeros(1, device=widths.device),
                             torch.cumsum(widths, dim=0)])
        knots_y = torch.cat([torch.zeros(1, device=heights.device),
                             torch.cumsum(heights, dim=0)])
        knots_x = knots_x / knots_x[-1]  # normalize to [0, 1]
        knots_y = knots_y / knots_y[-1]
        return knots_x, knots_y, derivs

    def forward(self, x):
        return _apply_real_line_rqs(
            x, self._params(), self.num_bins, self.tail_bound, forward=True)


class _RealLineRQSplineInverse(nn.Module):
    """Analytic inverse of RealLineRQSpline (same parameters, no copy)."""

    def __init__(self, parent):
        super().__init__()
        object.__setattr__(self, "parent", parent)

    def forward(self, y):
        p = self.parent
        return _apply_real_line_rqs(
            y, p._params(), p.num_bins, p.tail_bound, forward=False)


# -----------------------------------------------------------------------
#  Core evaluation shared by forward and inverse
# -----------------------------------------------------------------------

def _apply_real_line_rqs(x, params, K, B, forward):
    """Evaluate or invert the real-line RQS with linear tails.

    Parameters
    ----------
    x : Tensor
        Input values (arbitrary shape).
    params : (knots_x, knots_y, derivs) on [0, 1].
    K : int
        Number of bins.
    B : float
        Tail bound.
    forward : bool
        True for forward evaluation, False for inverse.
    """
    knots_x, knots_y, derivs = params

    # Scale knots from [0,1] to [-B, B]
    knots_x_scaled = -B + 2.0 * B * knots_x   # in [-B, B]
    knots_y_scaled = -B + 2.0 * B * knots_y

    # Boundary derivatives (for linear tails)
    d_left = derivs[0]
    d_right = derivs[-1]

    orig_shape = x.shape
    xf = x.reshape(-1)

    # Classify points
    if forward:
        inside = (xf >= -B) & (xf <= B)
        left = xf < -B
        right = xf > B
    else:
        # For the inverse, check against the y-domain boundaries
        y_left = knots_y_scaled[0]    # = -B at init
        y_right = knots_y_scaled[-1]  # =  B at init
        inside = (xf >= y_left) & (xf <= y_right)
        left = xf < y_left
        right = xf > y_right

    out = torch.empty_like(xf)

    # --- Inside the spline domain ---
    if inside.any():
        x_in = xf[inside]
        if forward:
            out[inside] = _rqs_forward_segment(
                x_in, knots_x_scaled, knots_y_scaled, derivs, K)
        else:
            out[inside] = _rqs_inverse_segment(
                x_in, knots_x_scaled, knots_y_scaled, derivs, K)

    # --- Linear tails ---
    if left.any():
        if forward:
            out[left] = knots_y_scaled[0] + d_left * (xf[left] - knots_x_scaled[0])
        else:
            out[left] = knots_x_scaled[0] + (xf[left] - knots_y_scaled[0]) / d_left

    if right.any():
        if forward:
            out[right] = knots_y_scaled[-1] + d_right * (xf[right] - knots_x_scaled[-1])
        else:
            out[right] = knots_x_scaled[-1] + (xf[right] - knots_y_scaled[-1]) / d_right

    return out.reshape(orig_shape)


def _rqs_forward_segment(x, knots_x, knots_y, derivs, K):
    """Forward RQS evaluation on the interior domain."""
    # Bin index via searchsorted on the scaled x-knots
    idx = (torch.searchsorted(knots_x[1:].contiguous(), x.contiguous(),
                              right=False)).clamp(0, K - 1)

    oh = F.one_hot(idx, K).to(x.dtype)

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

    xi = ((x - x_k) / w).clamp(0.0, 1.0)
    one_m = 1.0 - xi
    num = h * (s * xi * xi + d_k * xi * one_m)
    den = s + (d_kp + d_k - 2.0 * s) * xi * one_m
    return y_k + num / den


def _rqs_inverse_segment(y, knots_x, knots_y, derivs, K):
    """Inverse RQS evaluation on the interior domain."""
    idx = (torch.searchsorted(knots_y[1:].contiguous(), y.contiguous(),
                              right=False)).clamp(0, K - 1)

    oh = F.one_hot(idx, K).to(y.dtype)

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

    dy = y - y_k
    a = h * (s - d_k) + dy * (d_kp + d_k - 2.0 * s)
    b = h * d_k - dy * (d_kp + d_k - 2.0 * s)
    c = -s * dy
    disc = (b * b - 4.0 * a * c).clamp(min=0.0)
    xi = (2.0 * c / (-b - torch.sqrt(disc))).clamp(0.0, 1.0)
    return x_k + xi * w
