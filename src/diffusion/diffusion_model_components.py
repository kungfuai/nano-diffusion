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
from src.diffusion.base import Diffusion
from src.diffusion.ddpm import DDPM
from src.diffusion.vdm import VDM


@dataclass
class DiffusionModelComponents:
    denoising_model: nn.Module
    ema_model: Optional[nn.Module]
    optimizer: optim.Optimizer
    lr_scheduler: Any
    noise_schedule: Dict[str, torch.Tensor]
    diffusion: Diffusion
    vae: Optional[nn.Module] = None


def create_vae_if_data_is_latent(config: DiffusionTrainingConfig) -> nn.Module:
    vae = None
    if config.data_is_latent:
        from diffusers import AutoencoderKL

        if config.vae_use_fp16:
            vae = AutoencoderKL.from_pretrained(config.vae_model_name, torch_dtype=torch.float16).to(config.device)
        else:
            vae = AutoencoderKL.from_pretrained(config.vae_model_name).to(config.device)
    return vae


def create_diffusion_model_components(
    config: DiffusionTrainingConfig,
    denoising_model: Optional[nn.Module] = None,
) -> DiffusionModelComponents:
    device = torch.device(config.device)
    if denoising_model is None: # if no denoising model is provided, create one
        denoising_model = create_model(
            net=config.net, in_channels=config.in_channels, resolution=config.resolution, cond_embed_dim=config.cond_embed_dim
        )
        config.net = denoising_model.name if hasattr(denoising_model, "name") else denoising_model.__class__.__name__
    denoising_model = denoising_model.to(device)
    if config.compile:
        denoising_model = torch.compile(denoising_model)
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
    vae = create_vae_if_data_is_latent(config)
    if config.diffusion_algo == "ddpm":
        diffusion = DDPM(denoising_model=denoising_model, noise_schedule=noise_schedule, config=config, vae=vae)
    elif config.diffusion_algo == "vdm":
        diffusion = VDM(denoising_model=denoising_model, config=config, vae=vae)
    else:
        raise ValueError(f"Invalid diffusion algorithm: {config.diffusion_algo}")

    if config.init_from_wandb_run_path and config.init_from_wandb_file:
        load_model_from_wandb(
            denoising_model,
            config.init_from_wandb_run_path,
            config.init_from_wandb_file,
        )

    return DiffusionModelComponents(
        denoising_model, ema_model, optimizer, lr_scheduler, noise_schedule, diffusion, vae
    )

