"""
Transport framework for SiT training and sampling.

Wraps the interpolant path with training loss computation and ODE/SDE sampling.
Supports velocity, score, and noise prediction model types.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Callable
from nanodiffusion.sit.interpolant import create_path, pad_t_like_x


class Transport:
    """Training and sampling wrapper for stochastic interpolant models.

    Parameters
    ----------
    path_type : str
        Interpolant path: "linear", "gvp", or "vp".
    prediction : str
        What the model predicts: "velocity", "score", or "noise".
    loss_weight : str
        Loss weighting: "none", "velocity", or "likelihood".
    t_min : float
        Minimum time for sampling (avoids singularity for score/noise).
    """

    def __init__(
        self,
        path_type: str = "linear",
        prediction: str = "velocity",
        loss_weight: str = "none",
        t_min: float = 1e-3,
    ):
        self.path = create_path(path_type)
        self.prediction = prediction
        self.loss_weight = loss_weight
        # For velocity prediction, t=0 is safe; for score/noise, avoid singularity
        self.t_min = 0.0 if prediction == "velocity" else t_min

    def training_losses(self, model: nn.Module, x1: torch.Tensor, model_kwargs: Optional[dict] = None):
        """Compute training loss for a batch of data.

        Parameters
        ----------
        model : nn.Module
            Model that takes (t, x, **kwargs) and returns prediction.
        x1 : (N, C, H, W) data samples.
        model_kwargs : dict
            Additional kwargs passed to model (e.g., conditioning).

        Returns
        -------
        loss : scalar tensor
        """
        if model_kwargs is None:
            model_kwargs = {}

        batch_size = x1.shape[0]
        device = x1.device

        # Sample time and noise
        t = torch.rand(batch_size, device=device) * (1.0 - self.t_min) + self.t_min
        x0 = torch.randn_like(x1)

        # Compute interpolated sample and velocity target
        x_t, u_t = self.path.plan(t, x0, x1)

        # Model prediction
        model_output = model(t=t, x=x_t, **model_kwargs)
        if hasattr(model_output, "sample"):
            model_output = model_output.sample

        if self.prediction == "velocity":
            # Model directly predicts velocity
            target = u_t
            loss = self._weighted_mse(model_output, target, t)
        elif self.prediction == "noise":
            # Model predicts the noise x0
            target = x0
            loss = self._weighted_mse(model_output, target, t)
        elif self.prediction == "score":
            # Model predicts score = -x0 / sigma_t
            _, sigma_t, _, _ = self.path.coefficients(t)
            sigma_t = pad_t_like_x(sigma_t, x0)
            target = -x0 / sigma_t.clamp(min=1e-6)
            loss = self._weighted_mse(model_output, target, t)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction}")

        return loss

    def _weighted_mse(self, pred: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute (possibly weighted) MSE loss."""
        mse = ((pred - target) ** 2).flatten(1).mean(dim=1)  # (N,)

        if self.loss_weight == "none":
            return mse.mean()
        elif self.loss_weight == "velocity":
            # Weight by (drift_var / sigma_t)^2 for noise/score prediction
            _, sigma_t, _, d_sigma_t = self.path.coefficients(t)
            alpha_t, _, d_alpha_t, _ = self.path.coefficients(t)
            alpha_ratio = alpha_t / d_alpha_t.clamp(min=1e-6)
            drift_var = sigma_t ** 2 * alpha_ratio - sigma_t * d_sigma_t
            weight = (drift_var / sigma_t.clamp(min=1e-6)) ** 2
            return (weight * mse).mean()
        elif self.loss_weight == "likelihood":
            _, sigma_t, _, d_sigma_t = self.path.coefficients(t)
            alpha_t, _, d_alpha_t, _ = self.path.coefficients(t)
            alpha_ratio = alpha_t / d_alpha_t.clamp(min=1e-6)
            drift_var = sigma_t ** 2 * alpha_ratio - sigma_t * d_sigma_t
            weight = drift_var / sigma_t.clamp(min=1e-6) ** 2
            return (weight * mse).mean()
        else:
            raise ValueError(f"Unknown loss_weight: {self.loss_weight}")


def euler_ode_sample(
    model: nn.Module,
    shape: tuple,
    num_steps: int = 100,
    device: torch.device = None,
    seed: int = 0,
    prediction: str = "velocity",
    path_type: str = "linear",
) -> torch.Tensor:
    """Generate samples using Euler ODE integration.

    Integrates from t=0 (noise) to t=1 (data).

    Parameters
    ----------
    model : nn.Module
        Trained model.
    shape : tuple
        Output shape (N, C, H, W).
    num_steps : int
        Number of integration steps.
    device : torch.device
    seed : int
    prediction : str
        "velocity", "score", or "noise".
    path_type : str
        Interpolant path type.

    Returns
    -------
    samples : (N, C, H, W) in range [0, 1]
    """
    path = create_path(path_type)

    torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_val = i / num_steps
        t = torch.full((shape[0],), t_val, device=device)

        with torch.no_grad():
            model_output = model(t=t, x=x)
            if hasattr(model_output, "sample"):
                model_output = model_output.sample

        if prediction == "velocity":
            # Velocity IS the ODE drift
            drift = model_output
        elif prediction == "noise":
            # Convert noise prediction to velocity via the path
            alpha_t, sigma_t, d_alpha_t, d_sigma_t = path.coefficients(t)
            alpha_t = pad_t_like_x(alpha_t, x)
            sigma_t = pad_t_like_x(sigma_t, x)
            d_alpha_t = pad_t_like_x(d_alpha_t, x)
            d_sigma_t = pad_t_like_x(d_sigma_t, x)
            # x_t = alpha_t * x1 + sigma_t * x0
            # x1 = (x_t - sigma_t * x0) / alpha_t
            x0_pred = model_output
            x1_pred = (x - sigma_t * x0_pred) / alpha_t.clamp(min=1e-6)
            drift = d_alpha_t * x1_pred + d_sigma_t * x0_pred
        elif prediction == "score":
            alpha_t, sigma_t, d_alpha_t, d_sigma_t = path.coefficients(t)
            alpha_t = pad_t_like_x(alpha_t, x)
            sigma_t = pad_t_like_x(sigma_t, x)
            d_alpha_t = pad_t_like_x(d_alpha_t, x)
            d_sigma_t = pad_t_like_x(d_sigma_t, x)
            # score = -x0 / sigma_t => x0 = -score * sigma_t
            x0_pred = -model_output * sigma_t
            x1_pred = (x - sigma_t * x0_pred) / alpha_t.clamp(min=1e-6)
            drift = d_alpha_t * x1_pred + d_sigma_t * x0_pred
        else:
            raise ValueError(f"Unknown prediction type: {prediction}")

        x = x + drift * dt

    # Clamp and normalize to [0, 1]
    x = x.clamp(-1, 1) / 2 + 0.5
    return x
