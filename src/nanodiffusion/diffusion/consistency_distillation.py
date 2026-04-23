"""
Latent Consistency Distillation (LCD / LCM).

Distills a pre-trained diffusion model (teacher) into a consistency model (student)
that can generate high-quality samples in 1-4 steps.

Reference: https://arxiv.org/abs/2310.04378
"""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .base import BaseDiffusionAlgorithm
from .noise_scheduler import create_noise_schedule, forward_diffusion
from ..bookkeeping.mini_batch import MiniBatch


def compute_predicted_x0(x_t: torch.Tensor, t_indices: torch.Tensor,
                         noise_pred: torch.Tensor,
                         alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Convert noise prediction to x_0 prediction.

    x_0 = (x_t - sqrt(1 - alpha_cumprod_t) * eps) / sqrt(alpha_cumprod_t)
    """
    shape = (-1,) + (1,) * (x_t.ndim - 1)
    alpha_prod_t = alphas_cumprod[t_indices].view(*shape)
    return (x_t - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()


def ddim_step(x_t: torch.Tensor, t_from: torch.Tensor, t_to: torch.Tensor,
              noise_pred: torch.Tensor,
              alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """One DDIM deterministic step from t_from to t_to.

    Given x_{t_from} and predicted noise, compute x_{t_to} via DDIM update.
    """
    shape = (-1,) + (1,) * (x_t.ndim - 1)
    alpha_from = alphas_cumprod[t_from].view(*shape)
    alpha_to = alphas_cumprod[t_to].view(*shape)

    # Predict x_0
    pred_x0 = (x_t - (1 - alpha_from).sqrt() * noise_pred) / alpha_from.sqrt()
    # DDIM deterministic step to t_to
    x_t_to = alpha_to.sqrt() * pred_x0 + (1 - alpha_to).sqrt() * noise_pred
    return x_t_to


class ConsistencyDistillation(BaseDiffusionAlgorithm):
    """Latent Consistency Distillation algorithm.

    The student learns a consistency function that maps any noisy sample
    directly to the clean data, enabling generation in 1-4 steps.

    The training uses a teacher model (frozen pre-trained diffusion model)
    to provide ODE solutions, and trains the student to be self-consistent
    across adjacent timesteps.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        noise_schedule: Dict[str, torch.Tensor],
        device: str = "cuda:0",
        num_denoising_steps: int = 1000,
        num_ddim_timesteps: int = 50,
        guidance_scale_range: tuple = (3.0, 13.0),
        data_is_latent: bool = False,
        vae_scale_multiplier: float = 0.18215,
        conditional: bool = False,
        cond_drop_prob: float = 0.2,
        clip_sample_range: float = 2.0,
        vae: Optional[nn.Module] = None,
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.target_model = copy.deepcopy(student_model)
        # Freeze teacher and target
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        for p in self.target_model.parameters():
            p.requires_grad_(False)

        self.noise_schedule = noise_schedule
        self.alphas_cumprod = noise_schedule["alphas_cumprod"]
        self.device = device
        self.num_denoising_steps = num_denoising_steps
        self.num_ddim_timesteps = num_ddim_timesteps
        self.guidance_scale_range = guidance_scale_range
        self.data_is_latent = data_is_latent
        self.vae_scale_multiplier = vae_scale_multiplier
        self.conditional = conditional
        self.cond_drop_prob = cond_drop_prob
        self.clip_sample_range = clip_sample_range
        self.vae = vae

        # Build the sub-sampled timestep schedule (evenly spaced)
        # E.g., for 1000 steps and 50 ddim steps: [0, 20, 40, ..., 980]
        self.ddim_timesteps = torch.linspace(
            0, num_denoising_steps - 1, num_ddim_timesteps + 1,
            dtype=torch.long, device=device
        )

    def update_target_model(self, ema_decay: float):
        """Update target model with EMA of student."""
        for target_p, student_p in zip(
            self.target_model.parameters(), self.student_model.parameters()
        ):
            target_p.data.mul_(ema_decay).add_(student_p.data, alpha=1 - ema_decay)

    def _teacher_predict_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                               y: torch.Tensor = None,
                               guidance_scale: float = None) -> torch.Tensor:
        """Get noise prediction from frozen teacher, optionally with CFG."""
        n_T = self.num_denoising_steps
        with torch.no_grad():
            if y is not None and guidance_scale is not None and guidance_scale > 1.0:
                # Classifier-free guidance
                x_twice = torch.cat([x_t, x_t])
                t_twice = torch.cat([t, t])
                uncond_emb = self.teacher_model.get_null_cond_embed(
                    batch_size=x_t.shape[0]
                )
                emb_cat = torch.cat([uncond_emb.to(y.device), y])
                output = self.teacher_model(
                    t=t_twice / n_T, x=x_twice, y=emb_cat
                )
                if hasattr(output, "sample"):
                    output = output.sample
                uncond_pred, cond_pred = output.chunk(2)
                noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                output = self.teacher_model(t=t / n_T, x=x_t, y=y)
                if hasattr(output, "sample"):
                    output = output.sample
                noise_pred = output
        return noise_pred

    def _consistency_function(self, model: nn.Module, x_t: torch.Tensor,
                              t: torch.Tensor,
                              y: torch.Tensor = None,
                              p_uncond: float = None) -> torch.Tensor:
        """Apply the consistency function f_theta(x_t, t) -> predicted x_0.

        f(x_t, t) = c_skip(t) * x_t + c_out(t) * model(x_t, t)

        where the model predicts noise, and we convert to x_0 prediction.
        The c_skip and c_out are chosen so that:
        - When alpha_cumprod_t ~= 1 (t ~= 0), f(x_t, t) ~= x_t (boundary condition)
        """
        n_T = self.num_denoising_steps

        inputs = {"x": x_t, "t": t / n_T}
        if y is not None:
            inputs["y"] = y
            inputs["p_uncond"] = p_uncond

        output = model(**inputs)
        if hasattr(output, "sample"):
            output = output.sample

        # Convert noise prediction to x_0 prediction
        pred_x0 = compute_predicted_x0(x_t, t, output, self.alphas_cumprod)

        if self.clip_sample_range > 0:
            pred_x0 = pred_x0.clamp(-self.clip_sample_range, self.clip_sample_range)

        return pred_x0

    def prepare_training_examples(self, batch: MiniBatch):
        """Prepare consistency distillation training examples.

        Returns a dict of inputs for the loss computation rather than
        the standard (inputs, targets) pair, since the consistency loss
        requires running both student and target models.

        Returns:
            A dict with keys:
                - x_0: clean data
                - y: conditioning embeddings (or None)
                - p_uncond: conditioning dropout probability
        """
        x_0 = batch.x.to(self.device)
        if self.data_is_latent and self.vae_scale_multiplier is not None:
            x_0 = x_0 * self.vae_scale_multiplier

        y = None
        p_uncond = None
        if self.conditional and batch.text_emb is not None:
            y = batch.text_emb.to(self.device)
            p_uncond = self.cond_drop_prob

        return {"x_0": x_0, "y": y, "p_uncond": p_uncond}

    def consistency_distillation_loss(
        self, x_0: torch.Tensor, y: torch.Tensor = None,
        p_uncond: float = None,
    ) -> torch.Tensor:
        """Compute the consistency distillation loss.

        1. Sample adjacent timestep pair (t_n, t_{n+1}) from the DDIM schedule
        2. Forward-diffuse x_0 to get x_{t_{n+1}}
        3. Use teacher to do one DDIM step: x_{t_{n+1}} -> x_hat_{t_n}
        4. Student predicts x_0 from x_{t_{n+1}}
        5. Target (EMA) predicts x_0 from x_hat_{t_n}
        6. Loss = ||student_pred - target_pred||^2
        """
        batch_size = x_0.shape[0]

        # Sample random index into DDIM schedule (excluding last)
        idx = torch.randint(
            0, len(self.ddim_timesteps) - 1, (batch_size,), device=self.device
        )
        t_n = self.ddim_timesteps[idx]          # earlier timestep
        t_n1 = self.ddim_timesteps[idx + 1]     # later timestep (noisier)

        # Forward diffuse x_0 to x_{t_{n+1}}
        noise = torch.randn_like(x_0)
        x_t_n1, _ = forward_diffusion(x_0, t_n1, self.noise_schedule, noise)

        # Sample guidance scale for this batch (LCM uses random w per batch)
        w_min, w_max = self.guidance_scale_range
        w = torch.FloatTensor(1).uniform_(w_min, w_max).item()

        # Teacher predicts noise at t_{n+1}, then DDIM step to t_n
        teacher_noise_pred = self._teacher_predict_noise(
            x_t_n1, t_n1, y=y, guidance_scale=w if y is not None else None
        )
        x_hat_t_n = ddim_step(x_t_n1, t_n1, t_n, teacher_noise_pred, self.alphas_cumprod)

        # Student: f_theta(x_{t_{n+1}}, t_{n+1}) -> predicted x_0
        student_pred = self._consistency_function(
            self.student_model, x_t_n1, t_n1, y=y, p_uncond=p_uncond
        )

        # Target: f_{theta^-}(x_hat_{t_n}, t_n) -> predicted x_0 (no grad)
        with torch.no_grad():
            target_pred = self._consistency_function(
                self.target_model, x_hat_t_n.detach(), t_n, y=y, p_uncond=p_uncond
            )

        # Huber loss (more robust than pure MSE for consistency distillation)
        loss = nn.functional.huber_loss(student_pred, target_pred.detach(), delta=1.0)
        return loss

    def sample(self, x_T: torch.Tensor, y: torch.Tensor = None,
               num_inference_steps: int = 4, guidance_scale: float = None,
               seed: int = 0, quiet: bool = False) -> torch.Tensor:
        """Generate samples using the consistency model in few steps.

        Args:
            x_T: Initial noise tensor.
            y: Conditioning embeddings (optional).
            num_inference_steps: Number of denoising steps (1-4 recommended).
            guidance_scale: Not used for the student (guidance is baked in).
            seed: Random seed.
            quiet: Suppress progress bar.
        """
        torch.manual_seed(seed)
        n_T = self.num_denoising_steps

        # Build inference timestep schedule (evenly spaced, decreasing)
        timesteps = torch.linspace(
            n_T - 1, 0, num_inference_steps + 1, dtype=torch.long, device=self.device
        )

        x_t = x_T
        model = self.target_model  # Use target (EMA) model for inference

        pbar = tqdm(range(num_inference_steps), disable=quiet)
        for i in pbar:
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = torch.full(
                (x_t.shape[0],), t_cur, device=self.device, dtype=torch.long
            )

            # Predict x_0 via consistency function
            with torch.no_grad():
                pred_x0 = self._consistency_function(model, x_t, t_batch, y=y)

            if i < num_inference_steps - 1 and t_next > 0:
                # Add noise back to get x_{t_next} for the next step
                t_next_batch = torch.full(
                    (x_t.shape[0],), t_next, device=self.device, dtype=torch.long
                )
                noise = torch.randn_like(x_t)
                alpha_next = self.alphas_cumprod[t_next_batch].view(-1, *([1]*(x_t.ndim-1)))
                x_t = alpha_next.sqrt() * pred_x0 + (1 - alpha_next).sqrt() * noise
            else:
                x_t = pred_x0

            if not quiet:
                pbar.set_postfix({"std": x_t.std().item()})

        # Decode if using VAE
        if self.vae is not None:
            x_t = x_t / self.vae_scale_multiplier
            decoded = self.vae.decode(x_t)
            x_t = decoded.sample if hasattr(decoded, "sample") else decoded
            x_t = (x_t / 2 + 0.5).clamp(0, 1)
        else:
            x_t = (x_t / 2 + 0.5).clamp(0, 1)

        return x_t
