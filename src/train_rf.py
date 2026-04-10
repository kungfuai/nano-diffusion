"""
Rectified Flow (minRF) training for nano-diffusion.

Implements rectified flow training with logit-normal time sampling,
as described in "Improving Training of Rectified Flows" and minRF.

Key differences from standard CFM (train_cfm.py):
- Logit-normal time sampling: t = sigmoid(N(mean, std)) biases timesteps
  toward the middle of [0,1], improving training efficiency.
- Simple Euler integration for sampling (no torchdyn dependency needed).

Usage:
    python src/train_rf.py --dataset cifar10 --net dit_s2 --total_steps 100000

References:
- https://arxiv.org/abs/2405.20320
- https://github.com/cloneofsimo/minRF
"""

import argparse
import copy
from typing import Union
import os
import torch
import torch.optim as optim
from dataclasses import dataclass

from nanodiffusion.models.factory import create_model, choices
from nanodiffusion.optimizers.lr_schedule import get_cosine_schedule_with_warmup
from nanodiffusion.datasets import load_data
from nanodiffusion.plan.ot import OTPlanSampler
from nanodiffusion.bookkeeping.wandb_utils import load_model_from_wandb

from train_cfm import (
    ConditionalFlowMatcher,
    TrainingConfig,
    FlowMatchingModelComponents,
    training_loop,
)


class RectifiedFlowMatcher(ConditionalFlowMatcher):
    """Rectified Flow matcher with logit-normal time sampling.

    Instead of t ~ U(0,1), samples t = sigmoid(N(mean, std)), which biases
    timesteps toward the middle of [0,1] for improved training.

    Reference: "Improving Training of Rectified Flows" (arXiv:2405.20320)
    """

    def __init__(self, sigma: Union[float, int] = 0.0,
                 logit_normal_mean: float = 0.0,
                 logit_normal_std: float = 1.0):
        super().__init__(sigma)
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            # Logit-normal sampling: t = sigmoid(N(mean, std))
            nt = torch.randn(x0.shape[0], device=x0.device)
            t = torch.sigmoid(nt * self.logit_normal_std + self.logit_normal_mean)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)


class ExactOptimalTransportRectifiedFlowMatcher(RectifiedFlowMatcher):
    """Rectified Flow with OT coupling and logit-normal time sampling."""

    def __init__(self, sigma: Union[float, int] = 0.0,
                 logit_normal_mean: float = 0.0,
                 logit_normal_std: float = 1.0):
        super().__init__(sigma, logit_normal_mean, logit_normal_std)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)


def generate_samples_euler(model, device, num_samples=8, resolution=32,
                           in_channels=3, seed=0, num_steps=50):
    """Generate samples via simple Euler integration (no torchdyn dependency).

    Integrates the learned velocity field from t=0 (noise) to t=1 (data).
    """
    model.eval()
    with torch.no_grad():
        torch.manual_seed(seed)
        x = torch.randn(num_samples, in_channels, resolution, resolution, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.ones(num_samples, device=device) * t_val
            v = model(t=t, x=x)
            v = v.sample if hasattr(v, "sample") else v
            x = x + v * dt
        x = x.clip(-1, 1) / 2 + 0.5
    return x


@dataclass
class RFTrainingConfig(TrainingConfig):
    """Training config extended with rectified flow parameters."""
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0


def create_rf_model_components(config: RFTrainingConfig) -> FlowMatchingModelComponents:
    """Create model components with a RectifiedFlowMatcher."""
    device = torch.device(config.device)
    denoising_model = create_model(
        net=config.net, in_channels=config.in_channels, resolution=config.resolution
    )
    denoising_model = denoising_model.to(device)
    ema_model = copy.deepcopy(denoising_model) if config.use_ema else None
    optimizer = optim.AdamW(
        denoising_model.parameters(),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps, lr_min=config.lr_min,
    )

    rf_kwargs = dict(
        sigma=0,
        logit_normal_mean=config.logit_normal_mean,
        logit_normal_std=config.logit_normal_std,
    )
    if config.plan == "ot":
        FM = ExactOptimalTransportRectifiedFlowMatcher(**rf_kwargs)
    elif config.plan == "simple":
        FM = RectifiedFlowMatcher(**rf_kwargs)
    else:
        raise ValueError(f"Unknown plan: {config.plan}")

    if config.init_from_wandb_run_path and config.init_from_wandb_file:
        load_model_from_wandb(
            denoising_model, config.init_from_wandb_run_path, config.init_from_wandb_file
        )

    return FlowMatchingModelComponents(denoising_model, ema_model, optimizer, lr_scheduler, FM)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Rectified Flow training for images")
    # Dataset
    parser.add_argument("-d", "--dataset", type=str, default="cifar10")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=32)
    # Model
    parser.add_argument("--net", type=str, choices=choices(), default="unet_small")
    parser.add_argument("--plan", type=str, choices=["ot", "simple"], default="simple")
    parser.add_argument("--num_denoising_steps", type=int, default=100)
    # RF-specific
    parser.add_argument("--logit_normal_mean", type=float, default=0.0,
                        help="Mean of logit-normal time distribution")
    parser.add_argument("--logit_normal_std", type=float, default=1.0,
                        help="Std of logit-normal time distribution")
    # Training
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr_min", type=float, default=2e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # EMA
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_beta", type=float, default=0.999)
    # Logging
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=1500)
    parser.add_argument("--save_every", type=int, default=60000)
    parser.add_argument("--validate_every", type=int, default=1500)
    parser.add_argument("--fid_every", type=int, default=6000)
    parser.add_argument("--num_samples_for_logging", type=int, default=8)
    parser.add_argument("--num_samples_for_fid", type=int, default=1000)
    parser.add_argument("--num_real_samples_for_fid", type=int, default=10000)
    # Misc
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_dir", type=str, default="logs/train_rf")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--data_is_latent", action="store_true")
    parser.add_argument("--watch_model", action="store_true")
    parser.add_argument("--init_from_wandb_run_path", type=str, default=None)
    parser.add_argument("--init_from_wandb_file", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = RFTrainingConfig(**vars(args))

    train_dataloader, val_dataloader = load_data(config)
    model_components = create_rf_model_components(config)

    num_examples_trained = training_loop(
        model_components, train_dataloader, val_dataloader, config
    )
    print(f"Training completed. Total examples trained: {num_examples_trained}")


if __name__ == "__main__":
    main()
