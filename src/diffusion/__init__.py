import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

from ..config.diffusion_training_config import DiffusionTrainingConfig


def create_noise_schedule(n_T: int, device: torch.device, beta_min: float=0.0001, beta_max: float=0.02) -> Dict[str, torch.Tensor]:
    betas = torch.linspace(beta_min, beta_max, n_T).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)

    return {
        "alphas": alphas,
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
    }


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


def denoising_step(denoising_model, x_t, t, noise_schedule, clip_sample=True, clip_sample_range=1.0):
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


def conditional_denoising_step(denoising_model, x_t, text_embeddings, t, noise_schedule, clip_sample=True, clip_sample_range=1.0, guidance_scale=7.5):
    """
    This is the backward diffusion step, with the effect of denoising.

    Implements classifier-free guidance by conditioning on both text embeddings and unconditional (null) embeddings.
    """
    
    if isinstance(t, int):
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device)
    else:
        t_tensor = t

    # Double the batch - first half conditioned on text, second half unconditioned
    x_twice = torch.cat([x_t] * 2)
    t_twice = torch.cat([t_tensor] * 2)
    
    # Create unconditional embeddings (zeros) and concatenate with text embeddings
    if text_embeddings is not None:
        uncond_embeddings = denoising_model.get_null_cond_embed(batch_size=x_t.shape[0])
        embeddings_cat = torch.cat([uncond_embeddings.to(text_embeddings.device), text_embeddings])
    else:
        embeddings_cat = None

    with torch.no_grad():
        model_output = denoising_model(t=t_twice, x=x_twice, y=embeddings_cat)
    if hasattr(model_output, "sample"):
        model_output = model_output.sample

    # Split predictions and perform guidance
    noise_pred_uncond, noise_pred_text = model_output.chunk(2)
    adjustment = guidance_scale * (noise_pred_text - noise_pred_uncond)
    # print("avg adjustment", adjustment.cpu().numpy().mean(), "guidance scale", guidance_scale)
    model_output = noise_pred_uncond + adjustment

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


def generate_samples_by_denoising(denoising_model, x_T, noise_schedule, n_T, device, clip_sample=True, clip_sample_range=1.0, seed=0, method="one_stop", quiet=False):
    """
    This is the generation process.
    """
    torch.manual_seed(seed)

    x_t = x_T.to(device)
    pbar = tqdm(range(n_T - 1, -1, -1)) if not quiet else range(n_T - 1, -1, -1)
    for t in pbar:
        if method == "direct":
            x_t = denoising_step_direct(denoising_model, x_t, t, noise_schedule, clip_sample, clip_sample_range)
        else:
            x_t = denoising_step(denoising_model, x_t, t, noise_schedule, clip_sample, clip_sample_range)
        if not quiet:
            pbar.set_postfix({"std": x_t.std().item()})

    # print("raw x_t range", x_t.min(), x_t.max())
    x_t = (x_t / 2 + 0.5).clamp(0, 1)
    # print("after clamp", x_t.min(), x_t.max())
    return x_t


def generate_conditional_samples_by_denoising(denoising_model, x_T, text_embeddings, noise_schedule, n_T, device, clip_sample=True, clip_sample_range=1.0, seed=0, method="one_stop", quiet=False, guidance_scale=7.5):
    """
    Generate latent samples by denoising. Optionally, text embeddings are provided.
    """
    torch.manual_seed(seed)

    x_t = x_T.to(device)
    pbar = tqdm(range(n_T - 1, -1, -1)) if not quiet else range(n_T - 1, -1, -1)
    for t in pbar:
        if method == "direct":
            raise NotImplementedError("Direct denoising step no longer supported")
        
        x_t = conditional_denoising_step(denoising_model, x_t, text_embeddings, t, noise_schedule, clip_sample, clip_sample_range, guidance_scale)
        
        if not quiet:
            pbar.set_postfix({"std": x_t.std().item()})

    # x_t = (x_t / 2 + 0.5).clamp(0, 1)
    print(f"x_t: min={x_t.min()}, max={x_t.max()}, l2 norm={x_t.norm(dim=1).mean().item()}, mean={x_t.mean()}, std={x_t.std()}")
    return x_t


def compute_validation_loss(
    denoising_model: nn.Module,
    val_dataloader: DataLoader,
    noise_schedule: Dict[str, torch.Tensor],
    n_T: int,
    device: torch.device,
) -> float:
    total_loss = 0
    num_batches = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for x, _ in val_dataloader:
            x = x.to(device)
            t = torch.randint(0, n_T, (x.shape[0],)).to(device)
            noise = torch.randn(x.shape).to(device)
            x_t, true_noise = forward_diffusion(x, t, noise_schedule, noise=noise)

            predicted_noise = denoising_model(t=t, x=x_t)
            if hasattr(predicted_noise, "sample"):
                predicted_noise = predicted_noise.sample

            loss = criterion(predicted_noise, true_noise)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def compute_validation_loss_for_latents(
    denoising_model: nn.Module,
    val_dataloader: DataLoader,
    noise_schedule: Dict[str, torch.Tensor],
    n_T: int,
    device: torch.device,
    config: DiffusionTrainingConfig,
) -> float:
    total_loss = 0
    num_batches = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in val_dataloader:
            x = batch['image_emb']
            x = x.float().to(device)
            x = x * config.vae_scale_factor
            if 'text_emb' in batch:
                text_emb = batch['text_emb'].float().to(device)
                text_emb = text_emb.reshape(text_emb.shape[0], -1)
            else:
                text_emb = None
            t = torch.randint(0, n_T, (x.shape[0],)).to(device)
            noise = torch.randn(x.shape).to(device)
            x_t, true_noise = forward_diffusion(x, t, noise_schedule, noise=noise)

            predicted_noise = denoising_model(t=t, x=x_t, y=text_emb, p_uncond=0.1)  # TODO: this is hardcoded. The syntax is also wrong.
            if hasattr(predicted_noise, "sample"):
                predicted_noise = predicted_noise.sample

            loss = criterion(predicted_noise, true_noise)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def generate_images(
    model: nn.Module,
    noise_schedule: Dict[str, torch.Tensor],
    n_T: int,
    device: torch.device,
    batch_size: int,
    resolution: int = 32,
    in_channels: int = 3,
) -> torch.Tensor:
    with torch.no_grad():
        x = torch.randn(batch_size, in_channels, resolution, resolution).to(device)
        samples = generate_samples_by_denoising(model, x, noise_schedule, n_T, device)
    return samples