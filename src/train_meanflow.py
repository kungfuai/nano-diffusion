"""
MeanFlow training pipeline for one-step image generation.

MeanFlow learns an average velocity field u(z_t, r, t) instead of an instantaneous
velocity v_t(z_t). At inference, a single forward pass generates images:
    x_0 = z_1 - u(z_1, r=0, t=1)   where z_1 ~ N(0, I)

The training signal comes from the MeanFlow identity:
    u = v - (t-r) * du/dt
computed efficiently using Jacobian-vector products (JVP).

Usage:
    python src/train_meanflow.py -d cifar10 --net dit_t1 --resolution 32
    python src/train_meanflow.py -d cifar10 --net unet_small --resolution 32 --sample_steps 5

References:
    - "Mean Flows for One-step Generative Modeling" (Geng et al., 2025)
    - https://arxiv.org/abs/2505.13447
"""

import argparse
import copy
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import torch
import torch.optim as optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

try:
    import wandb
except ImportError:
    print("wandb not installed, skipping")

try:
    from cleanfid import fid
except ImportError:
    print("clean-fid not installed, skipping")

from nanodiffusion.meanflow import (
    MeanFlowModelWrapper,
    compute_meanflow_loss,
    generate_samples_meanflow,
)
from nanodiffusion.models.factory import create_model, choices
from nanodiffusion.optimizers.lr_schedule import get_cosine_schedule_with_warmup
from nanodiffusion.datasets import load_data
from nanodiffusion.bookkeeping.cfm_bookkeeping import (
    log_training_step,
    compute_fid,
    save_model,
)
from nanodiffusion.cfm.cfm_training_loop import (
    update_ema_model,
    save_final_models,
    save_model,
    precompute_fid_stats_for_real_images,
)
from nanodiffusion.bookkeeping.wandb_utils import load_model_from_wandb


@dataclass
class MeanFlowTrainingConfig:
    # Dataset
    dataset: str
    resolution: int
    val_split: float = 0.1
    in_channels: int = 3

    # Model architecture
    net: str = "dit_t1"

    # MeanFlow-specific
    flow_ratio: float = 0.5  # fraction of samples where r=t (instantaneous velocity)
    time_dist: str = "lognorm"  # "lognorm" or "uniform"
    time_mu: float = -0.4  # logit-normal mean
    time_sigma: float = 1.0  # logit-normal std
    adaptive_loss_gamma: float = 0.5  # adaptive loss power
    sample_steps: int = 1  # sampling steps (1 = one-step generation)

    # Training loop and optimizer
    total_steps: int = 100000
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_min: float = 1e-6
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # EMA
    use_ema: bool = False
    ema_beta: float = 0.9999

    # Logging and evaluation
    log_every: int = 100
    sample_every: int = 1000
    save_every: int = 50000
    validate_every: int = 10000
    fid_every: int = 10000
    num_samples_for_fid: int = 1000
    num_samples_for_logging: int = 8
    num_real_samples_for_fid: int = 10000

    # Infrastructure
    device: str = "cuda"
    logger: str = "wandb"
    cache_dir: str = f"{os.path.expanduser('~')}/.cache"
    checkpoint_dir: str = "logs/train_meanflow"
    min_steps_for_final_save: int = 100
    watch_model: bool = False
    init_from_wandb_run_path: str = None
    init_from_wandb_file: str = None
    random_flip: bool = False

    def __post_init__(self):
        self.checkpoint_dir = (
            f"{self.checkpoint_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )


@dataclass
class MeanFlowModelComponents:
    model: MeanFlowModelWrapper
    ema_model: Optional[MeanFlowModelWrapper]
    optimizer: Optimizer
    lr_scheduler: Any
    # Expose base model for saving/loading compatibility
    denoising_model: Module = None

    def __post_init__(self):
        self.denoising_model = self.model


def create_meanflow_model_components(config: MeanFlowTrainingConfig) -> MeanFlowModelComponents:
    device = torch.device(config.device)
    base_model = create_model(
        net=config.net,
        in_channels=config.in_channels,
        resolution=config.resolution,
    )
    model = MeanFlowModelWrapper(base_model).to(device)

    ema_model = copy.deepcopy(model) if config.use_ema else None

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps,
        lr_min=config.lr_min,
    )

    if config.init_from_wandb_run_path and config.init_from_wandb_file:
        load_model_from_wandb(
            model, config.init_from_wandb_run_path, config.init_from_wandb_file
        )

    return MeanFlowModelComponents(model, ema_model, optimizer, lr_scheduler)


def train_step(model, x, optimizer, config, device):
    optimizer.zero_grad()
    x = x.to(device)

    loss, mse = compute_meanflow_loss(
        model, x, device,
        flow_ratio=config.flow_ratio,
        time_dist=config.time_dist,
        time_mu=config.time_mu,
        time_sigma=config.time_sigma,
        gamma=config.adaptive_loss_gamma,
    )

    loss.backward()

    if config.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

    optimizer.step()
    return loss, mse


def generate_and_log_samples(components, config, step=None, seed=0):
    device = torch.device(config.device)
    model = components.model
    n = config.num_samples_for_logging

    sampled = generate_samples_meanflow(
        model, device, n,
        resolution=config.resolution,
        in_channels=config.in_channels,
        sample_steps=config.sample_steps,
        seed=seed,
    )
    images = (sampled * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")

    if config.logger == "wandb":
        wandb.log({
            "num_batches_trained": step,
            "test_samples": [wandb.Image(img) for img in images],
        })
    else:
        grid = make_grid(sampled, nrow=4, normalize=False)
        save_image(grid, Path(config.checkpoint_dir) / f"samples_step_{step}.png")

    if config.use_ema and components.ema_model is not None:
        ema_sampled = generate_samples_meanflow(
            components.ema_model, device, n,
            resolution=config.resolution,
            in_channels=config.in_channels,
            sample_steps=config.sample_steps,
            seed=seed,
        )
        ema_images = (ema_sampled * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")

        if config.logger == "wandb":
            wandb.log({
                "num_batches_trained": step,
                "ema_test_samples": [wandb.Image(img) for img in ema_images],
            })
        else:
            grid = make_grid(ema_sampled, nrow=4, normalize=False)
            save_image(grid, Path(config.checkpoint_dir) / f"ema_samples_step_{step}.png")


def compute_and_log_fid(components, config):
    device = torch.device(config.device)
    batch_size = config.batch_size * 2
    num_batches = (config.num_samples_for_fid + batch_size - 1) // batch_size
    generated = []

    for i in range(num_batches):
        n = min(batch_size, config.num_samples_for_fid - len(generated))
        batch = generate_samples_meanflow(
            components.model, device, n,
            resolution=config.resolution,
            in_channels=config.in_channels,
            sample_steps=config.sample_steps,
            seed=i,
        )
        generated.append(batch)

    generated = torch.cat(generated, dim=0)
    fid_score = compute_fid(None, generated, device, config.dataset, config.resolution)
    print(f"FID Score: {fid_score:.4f}")

    log_dict = {"fid": fid_score}

    if config.use_ema and components.ema_model is not None:
        ema_generated = []
        for i in range(num_batches):
            n = min(batch_size, config.num_samples_for_fid - len(ema_generated))
            batch = generate_samples_meanflow(
                components.ema_model, device, n,
                resolution=config.resolution,
                in_channels=config.in_channels,
                sample_steps=config.sample_steps,
                seed=i,
            )
            ema_generated.append(batch)
        ema_generated = torch.cat(ema_generated, dim=0)
        ema_fid = compute_fid(None, ema_generated, device, config.dataset, config.resolution)
        print(f"EMA FID Score: {ema_fid:.4f}")
        log_dict["ema_fid"] = ema_fid

    if config.logger == "wandb":
        wandb.log(log_dict)


def training_loop(components, train_dataloader, val_dataloader, config):
    print(f"Training MeanFlow on {config.device}")
    device = torch.device(config.device)
    model = components.model.to(device)
    ema_model = components.ema_model

    if config.dataset not in ["cifar10"]:
        precompute_fid_stats_for_real_images(
            train_dataloader, config,
            Path(config.cache_dir) / "real_images_for_fid",
        )

    if config.logger == "wandb":
        project_name = os.getenv("WANDB_PROJECT") or "nano-diffusion"
        print(f"Logging to W&B project: {project_name}")
        params = sum(p.numel() for p in model.parameters())
        run_params = asdict(config)
        run_params["model_parameters"] = params
        run_params["algorithm"] = "meanflow"
        wandb.init(project=project_name, config=run_params)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    num_examples = 0

    while step < config.total_steps:
        for x, _ in train_dataloader:
            if step >= config.total_steps:
                break

            num_examples += x.shape[0]
            x = x.to(device)

            model.train()
            loss, mse = train_step(model, x, components.optimizer, config, device)
            model.eval()

            components.lr_scheduler.step()

            with torch.no_grad():
                if config.use_ema and ema_model is not None:
                    update_ema_model(ema_model, model, config.ema_beta)

                if step % config.log_every == 0:
                    log_training_step(step, num_examples, loss, components.optimizer, config.logger)
                    if config.logger == "wandb":
                        wandb.log({"mse": mse.item()})

                if step % config.sample_every == 0:
                    generate_and_log_samples(components, config, step=step, seed=0)

                if step % config.save_every == 0 and step > 0:
                    save_model(
                        model, checkpoint_dir / f"model_step_{step}.pth", config.logger
                    )
                    if config.use_ema and ema_model is not None:
                        save_model(
                            ema_model, checkpoint_dir / f"ema_model_step_{step}.pth",
                            config.logger,
                        )

                if step % config.fid_every == 0 and step > 0:
                    compute_and_log_fid(components, config)

            step += 1

    if step > config.min_steps_for_final_save:
        save_final_models(components, config)

    return num_examples


def parse_arguments():
    parser = argparse.ArgumentParser(description="MeanFlow training for one-step image generation")
    parser.add_argument("-d", "--dataset", type=str, default="cifar10")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--net", type=str, choices=choices(), default="dit_t1")

    # MeanFlow-specific
    parser.add_argument("--flow_ratio", type=float, default=0.5,
                        help="Fraction of samples with r=t (instantaneous velocity learning)")
    parser.add_argument("--time_dist", type=str, choices=["lognorm", "uniform"], default="lognorm")
    parser.add_argument("--time_mu", type=float, default=-0.4, help="Logit-normal mean")
    parser.add_argument("--time_sigma", type=float, default=1.0, help="Logit-normal std")
    parser.add_argument("--adaptive_loss_gamma", type=float, default=0.5)
    parser.add_argument("--sample_steps", type=int, default=1,
                        help="Sampling steps (1 = one-step generation)")

    # Training
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr_min", type=float, default=2e-6)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_beta", type=float, default=0.9999)

    # Logging
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=1500)
    parser.add_argument("--save_every", type=int, default=60000)
    parser.add_argument("--validate_every", type=int, default=1500)
    parser.add_argument("--fid_every", type=int, default=6000)
    parser.add_argument("--num_samples_for_fid", type=int, default=1000)
    parser.add_argument("--num_samples_for_logging", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_dir", type=str, default="logs/train_meanflow")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--watch_model", action="store_true")
    parser.add_argument("--init_from_wandb_run_path", type=str, default=None)
    parser.add_argument("--init_from_wandb_file", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_arguments()
    config = MeanFlowTrainingConfig(**vars(args))

    train_dataloader, val_dataloader = load_data(config)
    components = create_meanflow_model_components(config)

    num_examples = training_loop(components, train_dataloader, val_dataloader, config)
    print(f"Training completed. Total examples trained: {num_examples}")


if __name__ == "__main__":
    main()
