import copy
from dataclasses import dataclass
import torch
from torch import nn, optim
from typing import Dict, Optional, Any

from src.config.diffusion_training_config import DiffusionTrainingConfig
from src.models.factory import create_model
from src.optimizers.lr_schedule import get_cosine_schedule_with_warmup
from src.diffusion import create_noise_schedule
from src.bookkeeping.wandb_utils import load_model_from_wandb


@dataclass
class DiffusionModelComponents:
    denoising_model: nn.Module
    ema_model: Optional[nn.Module]
    optimizer: optim.Optimizer
    lr_scheduler: Any
    noise_schedule: Dict[str, torch.Tensor]


def create_diffusion_model_components(
    config: DiffusionTrainingConfig,
) -> DiffusionModelComponents:
    device = torch.device(config.device)
    denoising_model = create_model(
        net=config.net, in_channels=config.in_channels, resolution=config.resolution
    )
    denoising_model = denoising_model.to(device)
    # ema_model = create_ema_model(denoising_model, config.ema_beta) if config.use_ema else None
    ema_model = copy.deepcopy(denoising_model) if config.use_ema else None
    optimizer = optim.AdamW(
        denoising_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps,
        lr_min=config.lr_min,
    )
    noise_schedule = create_noise_schedule(config.num_denoising_steps, device)

    if config.init_from_wandb_run_path and config.init_from_wandb_file:
        load_model_from_wandb(
            denoising_model,
            config.init_from_wandb_run_path,
            config.init_from_wandb_file,
        )

    return DiffusionModelComponents(
        denoising_model, ema_model, optimizer, lr_scheduler, noise_schedule
    )