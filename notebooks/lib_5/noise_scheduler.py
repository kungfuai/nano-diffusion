import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

from config import TrainingConfig
from mini_batch import MiniBatch
from diffusion import Diffusion


def create_noise_schedule(n_T: int, device: torch.device, beta_min: float=0.0001, beta_max: float=0.02) -> Dict[str, torch.Tensor]:
    betas = torch.linspace(beta_min, beta_max, n_T).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)

    return {
        "alphas": alphas,
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "num_denoising_steps": n_T,
    }


def obsolete_sample_timesteps(num_examples: int, config: TrainingConfig) -> torch.Tensor:
    """
    Sample timesteps for the forward diffusion process.
    """
    return torch.randint(0, config.num_denoising_steps, (num_examples,), device=config.device).long()


def forward_diffusion(x_0, t, noise_schedule, noise=None):
    """
    Applies forward diffusion to input data.
    
    Args:
        x_0: Clean input data
        t: Timestep
        noise_schedule: Dictionary containing pre-computed noise parameters
        noise: Optional pre-generated noise
        
    Returns:
        x_t: Noised version of input
        noise: The noise that was added
    """
    t_shape = (-1,) + (1,) * (x_0.ndim - 1)
    _ts = t.view(*t_shape)
    if noise is None:
        noise = torch.randn_like(x_0)
    assert _ts.max() < len(noise_schedule["alphas_cumprod"]), f"t={_ts.max()} is larger than the length of noise_schedule: {len(noise_schedule['alphas_cumprod'])}"
    alpha_prod_t = noise_schedule["alphas_cumprod"][_ts]
    x_t = (alpha_prod_t ** 0.5) * x_0 + ((1 - alpha_prod_t) ** 0.5) * noise
    return x_t, noise


def obsolete_forward_diffusion_for_vdm(x_0: Tensor, t: Tensor, beta_a: float, beta_b: float, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Apply forward diffusion process to add noise to images.

    This function implements a non-standard forward diffusion process.
    It uses a beta distribution to sample noise levels, and uses beta, and 1-beta to scale the noise and signal levels.
    
    Args:
        x_0: The original images to which noise is added.
        t: The noise levels to apply to the images.
        beta_a: The alpha parameter for the noise level distribution.
        beta_b: The beta parameter for the noise level distribution.
        noise: Optional noise to use instead of generating new noise.
    
    Returns:
        x_noisy: The images with noise added.
        noise: The noise added to the images.
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    
    noise_level = torch.tensor(
        np.random.beta(beta_a, beta_b, len(x_0)), 
        device=x_0.device
    )
    signal_level = 1 - noise_level
    x_noisy = noise_level.view(-1, 1, 1, 1) * noise + signal_level.view(-1, 1, 1, 1) * x_0
    
    return x_noisy.float(), noise


def unconditional_denoising_step(denoising_model, x_t, t, noise_schedule, clip_sample=True, clip_sample_range=1.0):
    """
    This is the backward diffusion step, with the effect of denoising.
    """
    if isinstance(t, int):
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device)
    else:
        t_tensor = t
    with torch.no_grad():
        model_output = denoising_model(t=t_tensor, x=x_t)
    if hasattr(model_output, "sample"):
        model_output = model_output.sample

    # Extract relevant values from noise_schedule
    alpha_prod_t = noise_schedule["alphas_cumprod"][t_tensor]
    # deal with t=0 case where t can be a tensor
    alpha_prod_t_prev = torch.where(t_tensor > 0,
                                    noise_schedule["alphas_cumprod"][t_tensor - 1],
                                    torch.ones_like(t_tensor, device=x_t.device))

    # Reshape alpha_prod_t_prev for proper broadcasting
    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
    alpha_prod_t_prev = alpha_prod_t_prev.view(-1, 1, 1, 1)

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # Compute the previous sample mean
    pred_original_sample = (x_t - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

    if clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -clip_sample_range, clip_sample_range)

    # Compute the coefficients for pred_original_sample and current sample
    pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

    # Compute the previous sample
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * x_t

    # Add noise
    variance = torch.zeros_like(x_t)
    variance_noise = torch.randn_like(x_t)

    # Handle t=0 case where t can be a tensor
    non_zero_mask = (t_tensor != 0).float().view(-1, 1, 1, 1)
    variance = non_zero_mask * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t)
    variance = torch.clamp(variance, min=1e-20)

    pred_prev_sample = pred_prev_sample + (variance ** 0.5) * variance_noise

    return pred_prev_sample


def denoising_step(denoising_model, x_t, y, t, noise_schedule, clip_sample=True, clip_sample_range=1.0, guidance_scale=7.5, mask=None):
    """
    This is the backward diffusion step, with the effect of denoising.

    Implements classifier-free guidance by conditioning on both conditional (e.g. text prompt) embeddings and unconditional (null) embeddings.
    """
    num_denoising_steps = noise_schedule["num_denoising_steps"]
    
    if isinstance(t, int):
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device)
    else:
        t_tensor = t


    # Create unconditional embeddings (zeros) and concatenate with text embeddings
    if y is not None:
        # Double the batch - first half conditioned on text, second half unconditioned
        x_twice = torch.cat([x_t] * 2)
        t_twice = torch.cat([t_tensor] * 2)
        uncond_embeddings = denoising_model.get_null_cond_embed(batch_size=x_t.shape[0])
        embeddings_cat = torch.cat([uncond_embeddings.to(y.device), y])
        with torch.no_grad():
            model_output = denoising_model(t=t_twice / num_denoising_steps, x=x_twice, y=embeddings_cat)
        # Split predictions and perform guidance
        noise_pred_uncond, noise_pred_text = model_output.chunk(2)
        adjustment = guidance_scale * (noise_pred_text - noise_pred_uncond)
        model_output = noise_pred_uncond + adjustment
    else:
        # unconditional denoising
        with torch.no_grad():
            model_output = denoising_model(t=t_tensor / num_denoising_steps, x=x_t)
    
    if hasattr(model_output, "sample"):
        model_output = model_output.sample

    # Extract relevant values from noise_schedule
    alpha_prod_t = noise_schedule["alphas_cumprod"][t_tensor]
    # deal with t=0 case where t can be a tensor
    alpha_prod_t_prev = torch.where(t_tensor > 0,
                                    noise_schedule["alphas_cumprod"][t_tensor - 1],
                                    torch.ones_like(t_tensor, device=x_t.device))

    # Reshape alpha_prod_t_prev for proper broadcasting
    additional_shape = (1,) * (x_t.ndim - 1)
    alpha_prod_t = alpha_prod_t.view(-1, *additional_shape)
    alpha_prod_t_prev = alpha_prod_t_prev.view(-1, *additional_shape)

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # Compute the previous sample mean
    pred_original_sample = (x_t - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    if clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -clip_sample_range, clip_sample_range)

    # Compute the coefficients for pred_original_sample and current sample
    pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

    # Compute the previous sample
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * x_t

    # Add noise
    variance = torch.zeros_like(x_t)
    variance_noise = torch.randn_like(x_t)

    # Handle t=0 case where t can be a tensor
    non_zero_mask = (t_tensor != 0).float().view(-1, *additional_shape)
    variance = non_zero_mask * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t)
    variance = torch.clamp(variance, min=1e-20)

    pred_prev_sample = pred_prev_sample + (variance ** 0.5) * variance_noise

    return pred_prev_sample


def denoising_step_direct(
    denoising_model,
    x_t,
    t,
    noise_schedule,
    clip_sample=True,
    clip_sample_range=1.0,
):
    """
    This is the backward diffusion step, with the effect of denoising.
    """
    if isinstance(t, int):
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device)
    else:
        t_tensor = t

    with torch.no_grad():
        eps_theta = denoising_model(t=t_tensor, x=x_t)
    if hasattr(eps_theta, "sample"):
        eps_theta = eps_theta.sample

    # Extract alphas from noise schedule
    alpha_t = noise_schedule["alphas"][t_tensor]
    alpha_t_cumprod = noise_schedule["alphas_cumprod"][t_tensor]

    # Reshape for broadcasting
    view_shape = (-1,) + (1,) * (x_t.ndim - 1)
    alpha_t = alpha_t.view(*view_shape)
    alpha_t_cumprod = alpha_t_cumprod.view(*view_shape)

    # Calculate epsilon factor
    eps_factor = (1 - alpha_t) / (1 - alpha_t_cumprod).sqrt()

    # Calculate mean for reverse process
    mean = (1 / torch.sqrt(alpha_t)) * (x_t - eps_factor * eps_theta)

    # Add noise scaled by variance for non-zero timesteps
    noise = torch.randn_like(x_t)
    beta_t = 1 - alpha_t
    variance = beta_t * (1 - alpha_t_cumprod / alpha_t) / (1 - alpha_t_cumprod)
    variance = torch.clamp(variance, min=1e-20)  # Add clamp to prevent numerical instability
    
    # Mask out noise for t=0 timesteps
    non_zero_mask = (t_tensor > 0).float().view(*view_shape)
    noise_scale = torch.sqrt(variance) * non_zero_mask
    
    pred_prev_sample = mean + noise_scale * noise

    # Apply clipping
    if clip_sample:
        pred_prev_sample = torch.clamp(pred_prev_sample, -clip_sample_range, clip_sample_range)

    return pred_prev_sample


def generate_samples_by_denoising(
        denoising_model, x_T, y, noise_schedule, n_T, guidance_scale=4.5,
        clip_sample=True, clip_sample_range=1.0, seed=0,
        method="one_stop", quiet=False):
    """
    This is the generation process.

    y: the conditional embeddings (i.e. the embeddings of the prompt).
        If y is None, then the model is run in unconditional mode.
    """
    torch.manual_seed(seed)

    x_t = x_T
    pbar = tqdm(range(n_T - 1, -1, -1)) if not quiet else range(n_T - 1, -1, -1)
    for t in pbar:
        if method == "direct":
            # x_t = denoising_step_direct(denoising_model, x_t, t, y, noise_schedule, clip_sample, clip_sample_range)
            raise NotImplementedError("Direct denoising step no longer supported")
        else:
            x_t = denoising_step(denoising_model, x_t, y, t, noise_schedule, clip_sample, clip_sample_range, guidance_scale=guidance_scale)
        if not quiet:
            pbar.set_postfix({"std": x_t.std().item()})

    return x_t

def generate_samples_by_denoising_impainting(
        denoising_model, x_T, y, noise_schedule, n_T, guidance_scale=4.5,
        clip_sample=True, clip_sample_range=1.0, seed=0,
        method="one_stop", quiet=False, mask=None):
    """
    Args:
        mask: Optional binary mask where 1 indicates known pixels to maintain
    """
    torch.manual_seed(seed)

    x_t = x_T.clone()
    alphas_cumprod = noise_schedule['alphas_cumprod']

    pbar = tqdm(range(n_T - 1, -1, -1)) if not quiet else range(n_T - 1, -1, -1)
    random_noise = torch.randn_like(x_T)
    for t in pbar:
        # Handle masked regions if mask is provided
        if mask is not None:
            # Calculate noise level for this timestep
            signal_level = torch.sqrt(alphas_cumprod[t])
            noise_level = torch.sqrt(1 - alphas_cumprod[t])
            
            # Add appropriate noise to known regions
            x_input = x_t
            x_known = x_T * mask  # Use original values from x_T
            x_input = x_input * (1 - mask) + (noise_level * random_noise + 
                                            signal_level * x_known) * mask
        else:
            x_input = x_t

        if method == "direct":
            raise NotImplementedError("Direct denoising step no longer supported")
        else:
            # Use x_input instead of x_t for model input
            x_t = denoising_step(denoising_model, x_input, y, t, noise_schedule, 
                               clip_sample, clip_sample_range, guidance_scale=guidance_scale)

        if not quiet:
            pbar.set_postfix({"std": x_t.std().item()})

    return x_t


def compute_validation_loss(
    denoising_model: nn.Module,
    val_dataloader: DataLoader,
    diffusion: Diffusion,
    config: TrainingConfig,
) -> float:
    total_loss = 0
    num_batches = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in val_dataloader:
            batch = MiniBatch.from_dataloader_batch(batch).to(config.device)
            inputs, targets = diffusion.prepare_training_examples(batch)
            predictions = denoising_model(**inputs)
            predictions = predictions.sample if hasattr(predictions, "sample") else predictions

            loss = criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

