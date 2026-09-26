"""
FINE (Fourier-based Invertible Neural Encoder) model implementation.
This module contains the main FINE model and its components.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.nn.utils import spectral_norm


class MinMaxNormalizer:
    def __init__(self):
        self.min_val = 0.0
        self.max_val = 1.0

    def normalize(self, x):
        """ Apply min-max normalization """
        self.min_val = x.min()
        self.max_val = x.max()
        norm_x = (x - self.min_val) / (self.max_val - self.min_val)
        return norm_x

    def reverse(self, x):
        """ Reverse min-max normalization """
        return x * (self.max_val - self.min_val) + self.min_val


class InvertibleSigmoidNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.mean = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.std = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x):
        """ Maps input to (0,1) using a sigmoid transformation """
        normalized_x = (x - self.mean) / self.std
        return torch.sigmoid(normalized_x)

    def inverse(self, x):
        """ Inverts the transformation back to the original scale """
        logit_x = torch.log(x / (1 - x))
        return self.mean + self.std * logit_x


def smooth_relu(x, epsilon=0.01):
    """Smooth approximation of ReLU function"""
    t = ((x + epsilon) / (2 * epsilon)).clamp(0, 1)
    h = t * t * (3 - 2 * t)
    return x * h


class MonotonicPiecewiseLinear(nn.Module):
    def __init__(self, num_points, y_control_init, normalization='minmax'):
        super().__init__()
        assert num_points == len(y_control_init), "x and y must have the same length"

        self.x_control = nn.Parameter(torch.linspace(0, 1, num_points), requires_grad=False)
        self.raw_y_control = nn.Parameter(y_control_init)
        self.normalization = normalization
        if normalization == 'minmax':
            self.normalizer = MinMaxNormalizer()
        elif normalization == 'sigmoid':
            self.sigmoid = InvertibleSigmoidNorm()

    def forward(self, x):
        y_control = torch.cumsum(F.softplus(self.raw_y_control), dim=0)
        y_control = y_control - y_control[0]
        y_control = y_control / y_control[-1]

        output = torch.zeros_like(x)

        if self.normalization == 'minmax':
            x = self.normalizer.normalize(x)
        elif self.normalization == 'sigmoid':
            x = self.sigmoid(x)

        for i in range(len(self.x_control) - 1):
            slope = (y_control[i + 1] - y_control[i]) / (self.x_control[i + 1] - self.x_control[i])
            segment = smooth_relu(x - self.x_control[i]) - smooth_relu(x - self.x_control[i + 1])
            output += slope * segment

        return output


class InverseMonotonicPiecewiseLinear(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.x_control = model.x_control
        self.raw_y_control = model.raw_y_control
        if model.normalization == 'minmax':
            self.normalization = 'minmax'
            self.normalizer = model.normalizer
        elif model.normalization == 'sigmoid':
            self.normalization = 'sigmoid'
            self.sigmoid = model.sigmoid

    def forward(self, y):
        y_control = torch.cumsum(F.softplus(self.raw_y_control), dim=0)
        y_control = y_control - y_control[0]
        y_control = y_control / y_control[-1]

        output = torch.zeros_like(y)

        for i in range(len(self.x_control) - 1):
            slope = (self.x_control[i + 1] - self.x_control[i]) / (y_control[i + 1] - y_control[i])
            segment = smooth_relu(y - y_control[i]) - smooth_relu(y - y_control[i + 1])
            output += slope * segment

        if self.normalization == 'minmax':
            output = self.normalizer.reverse(output)
        elif self.normalization == 'sigmoid':
            output = self.sigmoid.inverse(output)
        return output


class InvertibleFourierFilter1D(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        freq_components = N // 2 + 1

        self.log_mag = nn.Parameter(torch.ones(freq_components))
        phase = nn.Parameter(torch.zeros(freq_components - 2) * 2 * torch.pi)
        self.register_buffer('phase',
                             torch.cat([torch.zeros(1), phase, torch.zeros(1)]))

    def _get_filter_fft(self):
        mag = torch.exp(self.log_mag)
        phase = torch.exp(1j * self.phase)
        filter_fft = mag * phase
        return filter_fft

    def forward(self, x):
        filter_fft = self._get_filter_fft().unsqueeze(0)
        x_fft = fft.rfft(x, dim=-1)
        y_fft = x_fft * filter_fft
        return fft.irfft(y_fft, n=self.N, dim=-1)

    def inverse(self, x):
        filter_fft = self._get_filter_fft()
        inverse_filter_fft = 1.0 / (filter_fft + 1e-6)
        inverse_filter_fft = inverse_filter_fft.unsqueeze(0)
        x_fft = fft.rfft(x, dim=-1)
        y_fft = x_fft * inverse_filter_fft
        return fft.irfft(y_fft, n=self.N, dim=-1)


class GaussianFourierFilter1D(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        freq_components = N // 2 + 1
        self.center = nn.Parameter(torch.tensor(float(freq_components // 2)))
        self.width = nn.Parameter(torch.tensor(float(freq_components // 4)))

    def _get_filter_fft(self):
        freq_indices = torch.arange(self.N // 2 + 1, device=self.center.device)
        gaussian_mag = torch.exp(-0.5 * ((freq_indices - self.center) / self.width) ** 2)
        return gaussian_mag

    def forward(self, x):
        filter_fft = self._get_filter_fft().unsqueeze(0)
        x_fft = fft.rfft(x, dim=-1)
        y_fft = x_fft * filter_fft
        return fft.irfft(y_fft, n=self.N, dim=-1)

    def inverse(self, x):
        filter_fft = self._get_filter_fft()
        inverse_filter_fft = 1.0 / (filter_fft + 1e-6)
        inverse_filter_fft = inverse_filter_fft.unsqueeze(0)
        x_fft = fft.rfft(x, dim=-1)
        y_fft = x_fft * inverse_filter_fft
        return fft.irfft(y_fft, n=self.N, dim=-1)


class GaussianMixtureFourierFilter1D(nn.Module):
    def __init__(self, N, K=2):
        super().__init__()
        self.N = N
        self.K = K
        freq_components = N // 2 + 1

        self.amplitudes = nn.Parameter(torch.ones(K))
        self.centers = nn.Parameter(torch.linspace(0, freq_components, K))
        self.widths = nn.Parameter(torch.ones(K) * (freq_components / 2))

    def _get_filter_fft(self):
        freq_indices = torch.arange(self.N // 2 + 1, device=self.centers.device)
        gaussians = [
            self.amplitudes[i] * torch.exp(-0.5 * ((freq_indices - self.centers[i]) / self.widths[i]) ** 2)
            for i in range(self.K)
        ]
        return torch.sum(torch.stack(gaussians), dim=0)

    def forward(self, x):
        filter_fft = self._get_filter_fft().unsqueeze(0)
        x_fft = fft.rfft(x, dim=-1)
        y_fft = x_fft * filter_fft
        return fft.irfft(y_fft, n=self.N, dim=-1)

    def inverse(self, x):
        filter_fft = self._get_filter_fft()
        inverse_filter_fft = 1.0 / (filter_fft + 1e-6)
        inverse_filter_fft = inverse_filter_fft.unsqueeze(0)
        x_fft = fft.rfft(x, dim=-1)
        y_fft = x_fft * inverse_filter_fft
        return fft.irfft(y_fft, n=self.N, dim=-1)


class FINE(nn.Module):
    """
    FINE (Fourier-based Invertible Neural Encoder) model.
    This model combines invertible layers (K) with Fourier-based convolutions (C)
    in a specified structure.
    """
    def __init__(self, input_dim, indices, normalization, structure, filter_type="Gaussian",
                 activation=None):
        super(FINE, self).__init__()
        self.input_dim = input_dim
        self.structure = structure
        self.indices = indices

        # Activation arm for the K-layer. If not given explicitly, fall back to
        # the FINE_ACTIVATION env var so the existing experiment entry points
        # (which do not pass `activation`) can be switched without edits.
        if activation is None:
            activation = os.environ.get("FINE_ACTIVATION", "baseline")
        self.activation = activation

        # Count the number of K and C in the structure
        self.K = structure.count('K')
        self.C = structure.count('C')

        # Define convolutional layers based on filter type
        if filter_type == "Gaussian":
            self.convs = nn.ModuleList([GaussianFourierFilter1D(input_dim) for _ in range(self.C)])
        elif filter_type == "GaussianMixture":
            self.convs = nn.ModuleList([GaussianMixtureFourierFilter1D(input_dim) for _ in range(self.C)])
        elif filter_type == "Full":
            self.convs = nn.ModuleList([InvertibleFourierFilter1D(input_dim) for _ in range(self.C)])

        # Define KAN layers (the invertible monotonic activation).
        if activation == "baseline":
            self.kans = nn.ModuleList([
                MonotonicPiecewiseLinear(20, torch.ones(20), normalization=normalization)
                for _ in range(self.K)
            ])
            # baseline keeps the external inverse-assignment pattern
            for kan in self.kans:
                kan.inverse = InverseMonotonicPiecewiseLinear(kan)
        elif activation in ("leaky", "spline"):
            # lazy import to avoid a circular import (activations imports FINE)
            from opine_experiments.models.one_d.activations import make_kan_layer
            self.kans = nn.ModuleList([
                make_kan_layer(activation, num_points=20, normalization=normalization)
                for _ in range(self.K)
            ])
            # leaky/spline layers create their own `.inverse` in __init__
        else:
            raise ValueError(f"unknown activation arm: {activation!r}")

    def forward(self, x):
        # Forward pass through the network
        kan_index = 0
        conv_index = 0
        for layer in self.structure:
            if layer == 'K':
                x = self.kans[kan_index](x)
                kan_index += 1
            elif layer == 'C':
                x = self.convs[conv_index](x)
                conv_index += 1

        # FFT truncation
        x = fft.rfft(x, dim=-1)
        info = x[:, self.indices]
        x_truncated = torch.zeros(x.shape[0], self.input_dim//2+1, device=x.device, dtype=x.dtype)
        x_truncated[:, self.indices] = info
        x = fft.irfft(x_truncated, n=self.input_dim, dim=-1)

        # Inverse pass
        for layer in reversed(self.structure):
            if layer == 'C':
                x = self.convs[conv_index - 1].inverse(x)
                conv_index -= 1
            elif layer == 'K':
                x = self.kans[kan_index - 1].inverse(x)
                kan_index -= 1

        return x

    def encode_pre_truncation(self, x):
        """Run the encoder (K/C layers) up to, but not including, the FFT
        truncation. Returns the signal whose rFFT is about to be truncated to
        `self.indices`. Used by the spectral-leakage diagnostic to measure how
        much energy a given activation pushes outside the retained modes."""
        kan_index = 0
        conv_index = 0
        for layer in self.structure:
            if layer == 'K':
                x = self.kans[kan_index](x)
                kan_index += 1
            elif layer == 'C':
                x = self.convs[conv_index](x)
                conv_index += 1
        return x


def ae_loss(recon_x, x):
    """Reconstruction loss for the autoencoder."""
    return F.mse_loss(recon_x, x, reduction='sum')
