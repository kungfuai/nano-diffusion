import numpy as np
import torch
import torchvision.transforms.functional as F
from ..config.diffusion_training_config import DiffusionTrainingConfig as TrainingConfig


class SquarePad:
    """
    Borrowed from https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/5
    """
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')
    

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