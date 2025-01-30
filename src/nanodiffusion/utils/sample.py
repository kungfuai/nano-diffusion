import torch
import numpy as np
from typing import Callable, Dict
from torchvision.utils import make_grid, save_image


def threshold_sample(sample: torch.Tensor, dynamic_thresholding_ratio=0.995, sample_max_value=1.0) -> torch.Tensor:
    """
    "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
    prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
    s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
    pixels from saturation at each step. We find that dynamic thresholding results in significantly better
    photorealism as well as better image-text alignment, especially when using very large guidance weights."

    https://arxiv.org/abs/2205.11487
    """
    dtype = sample.dtype
    batch_size, channels, *remaining_dims = sample.shape

    if dtype not in (torch.float32, torch.float64):
        sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

    # Flatten sample for doing quantile calculation along each image
    sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

    abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

    s = torch.quantile(abs_sample, dynamic_thresholding_ratio, dim=1)
    s = torch.clamp(
        s, min=1, max=sample_max_value
    )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
    s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
    sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

    sample = sample.reshape(batch_size, channels, *remaining_dims)
    sample = sample.to(dtype)

    return sample


def denoise_and_compare(
        model: torch.nn.Module, 
        images: torch.Tensor, 
        forward_diffusion: Callable, 
        denoising_step: Callable, 
        noise_schedule: Dict, 
        n_T: int,  # timesteps for diffusion
        device: str,
    ):
    torch.manual_seed(10)
    model.eval()
    with torch.no_grad():
        # Add noise to the images
        t = torch.randint(0, n_T, (images.shape[0],), device=device)
        x_t, _ = forward_diffusion(images, t, noise_schedule)
        
        # Denoise the images
        pred_noise = model(x_t, t)
        if hasattr(pred_noise, "sample"):
            pred_noise = pred_noise.sample
        pred_previous_images = denoising_step(model, x_t, t, noise_schedule)
        # Compute the predicted original images using the correct formula
        alpha_t = noise_schedule["alphas"][t][:, None, None, None]
        alpha_t_cumprod = noise_schedule["alphas_cumprod"][t][:, None, None, None]
        pred_original_images = (
            x_t - ((1 - alpha_t) / (1 - alpha_t_cumprod).sqrt()) * pred_noise) / (alpha_t / (1 - alpha_t_cumprod).sqrt())
    model.train()
    return x_t, pred_original_images


def save_comparison_grid(original, denoised, ema_denoised, step, prefix, output_dir):
    # Combine images into a grid
    grid_images = torch.cat([original, denoised], dim=0)
    if ema_denoised is not None:
        grid_images = torch.cat([grid_images, ema_denoised], dim=0)
    
    grid = make_grid(grid_images, nrow=original.shape[0], normalize=True, scale_each=True)
    save_image(grid, output_dir / f"{prefix}_comparison_step{step}.png")