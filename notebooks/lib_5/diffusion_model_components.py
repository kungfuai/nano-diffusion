import copy
from dataclasses import dataclass
import torch
from torch import nn, optim
from typing import Dict, Optional, Any

from config import TrainingConfig
from model import create_denoiser_model
from noise_scheduler import create_noise_schedule
from diffusion import Diffusion
from ddpm import DDPM
from vdm import VDM


@dataclass
class DiffusionModelComponents:
    denoising_model: nn.Module
    ema_model: Optional[nn.Module]
    optimizer: optim.Optimizer
    noise_schedule: Dict[str, torch.Tensor]
    diffusion: Diffusion
    vae: Optional[nn.Module] = None


def create_vae_if_data_is_latent(config: TrainingConfig) -> nn.Module:
    vae = None
    if config.data_is_latent:
        from diffusers import AutoencoderKL

        if config.vae_use_fp16:
            vae = AutoencoderKL.from_pretrained(config.vae_model_name, torch_dtype=torch.float16).to(config.device)
        else:
            vae = AutoencoderKL.from_pretrained(config.vae_model_name).to(config.device)
    return vae


def create_diffusion_model_components(
    config: TrainingConfig,
    denoising_model: Optional[nn.Module] = None,
) -> DiffusionModelComponents:
    device = torch.device(config.device)
    if denoising_model is None:  # if no denoising model is provided, create one based on config.net
        denoising_model = create_denoiser_model(
            config=config, device=device
        )
    else:
        config.net = denoising_model.name if hasattr(denoising_model, "name") else denoising_model.__class__.__name__
    denoising_model = denoising_model.to(device)
    if config.compile:
        denoising_model = torch.compile(denoising_model)
    ema_model = None
    optimizer = optim.AdamW(
        denoising_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    noise_schedule = create_noise_schedule(config.num_denoising_steps, device)
    vae = create_vae_if_data_is_latent(config)
    if config.diffusion_algo == "ddpm":
        diffusion = DDPM(denoising_model=denoising_model, noise_schedule=noise_schedule, config=config, vae=vae)
    elif config.diffusion_algo == "vdm":
        diffusion = VDM(denoising_model=denoising_model, config=config, vae=vae)
    else:
        raise ValueError(f"Invalid diffusion algorithm: {config.diffusion_algo}")


    return DiffusionModelComponents(
        denoising_model, ema_model, optimizer, noise_schedule, diffusion, vae
    )

