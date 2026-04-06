"""
Stochastic interpolant paths for SiT (Scalable Interpolant Transformers).

Based on SiT (arXiv:2401.08740): https://github.com/willisma/SiT

Defines continuous-time interpolation paths x_t = alpha_t * x_1 + sigma_t * x_0,
where x_1 is data and x_0 ~ N(0, I) is noise. t goes from 0 (noise) to 1 (data).

Three path types:
- Linear: alpha_t = t, sigma_t = 1-t (equivalent to standard flow matching)
- GVP (trigonometric): alpha_t = sin(pi*t/2), sigma_t = cos(pi*t/2)
- VP (variance preserving): recovers the VP-SDE noise schedule
"""

import torch
import math
from typing import Tuple


def pad_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape t to be broadcastable with x."""
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class LinearPath:
    """Linear interpolant: alpha_t = t, sigma_t = 1 - t."""

    def coefficients(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (alpha_t, sigma_t, d_alpha_t, d_sigma_t)."""
        return t, 1 - t, torch.ones_like(t), -torch.ones_like(t)

    def plan(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
        """Compute interpolated sample x_t and velocity target u_t."""
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.coefficients(t)
        alpha_t = pad_t_like_x(alpha_t, x0)
        sigma_t = pad_t_like_x(sigma_t, x0)
        d_alpha_t = pad_t_like_x(d_alpha_t, x0)
        d_sigma_t = pad_t_like_x(d_sigma_t, x0)

        x_t = alpha_t * x1 + sigma_t * x0
        u_t = d_alpha_t * x1 + d_sigma_t * x0  # velocity = x1 - x0 for linear
        return x_t, u_t


class TrigonometricPath:
    """GVP / trigonometric interpolant: alpha_t = sin(pi*t/2), sigma_t = cos(pi*t/2)."""

    def coefficients(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_t = torch.sin(math.pi * t / 2)
        sigma_t = torch.cos(math.pi * t / 2)
        d_alpha_t = (math.pi / 2) * torch.cos(math.pi * t / 2)
        d_sigma_t = -(math.pi / 2) * torch.sin(math.pi * t / 2)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def plan(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.coefficients(t)
        alpha_t = pad_t_like_x(alpha_t, x0)
        sigma_t = pad_t_like_x(sigma_t, x0)
        d_alpha_t = pad_t_like_x(d_alpha_t, x0)
        d_sigma_t = pad_t_like_x(d_sigma_t, x0)

        x_t = alpha_t * x1 + sigma_t * x0
        u_t = d_alpha_t * x1 + d_sigma_t * x0
        return x_t, u_t


class VPPath:
    """Variance-preserving interpolant (recovers VP-SDE schedule)."""

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def coefficients(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = 1 - t  # reverse time for VP convention
        log_mean_coeff = -0.25 * s ** 2 * (self.sigma_max - self.sigma_min) - 0.5 * s * self.sigma_min
        alpha_t = torch.exp(log_mean_coeff)
        sigma_t = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))

        # Derivatives via chain rule
        d_log_mean = (0.5 * s * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min)
        d_alpha_t = alpha_t * d_log_mean
        d_sigma_t = -torch.exp(2 * log_mean_coeff) * d_log_mean / sigma_t
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def plan(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.coefficients(t)
        alpha_t = pad_t_like_x(alpha_t, x0)
        sigma_t = pad_t_like_x(sigma_t, x0)
        d_alpha_t = pad_t_like_x(d_alpha_t, x0)
        d_sigma_t = pad_t_like_x(d_sigma_t, x0)

        x_t = alpha_t * x1 + sigma_t * x0
        u_t = d_alpha_t * x1 + d_sigma_t * x0
        return x_t, u_t


def create_path(path_type: str = "linear"):
    """Factory for interpolant paths."""
    if path_type == "linear":
        return LinearPath()
    elif path_type in ("gvp", "trigonometric"):
        return TrigonometricPath()
    elif path_type == "vp":
        return VPPath()
    else:
        raise ValueError(f"Unknown path type: {path_type}. Choose from: linear, gvp, vp")
