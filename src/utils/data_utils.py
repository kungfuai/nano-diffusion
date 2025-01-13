import torch
from ..config.diffusion_training_config import DiffusionTrainingConfig as TrainingConfig


def scale_input(x: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    """
    Scale the input tensor by the vae scale factor if the data is latent.

    The latent embeddings from VAE are typically too large, so we scale them down (e.g. by a factor of 0.18215)
    so that it is closer to a unit Gaussian.

    The generated latents using the denoiser will need to be scaled up by the same factor
    before being passed to the VAE decoder.
    """
    if config.data_is_latent and config.vae_scale_multiplier is not None:
        return x * config.vae_scale_multiplier
    return x