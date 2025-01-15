from typing import Tuple, Dict
import torch
from ..bookkeeping.mini_batch import MiniBatch


class BaseDiffusionAlgorithm:
    """
    An interface for all diffusion algorithms.
    """
    def prepare_training_examples(self, batch: MiniBatch, **kwargs) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Prepare a training example for the denoising model.

        This is used in the traiining step or a validation step.
        """
        ...
    
    def sample(self, x_T, y = None, guidance_scale: float = None, seed: int = None, **kwargs):
        """
        Sample from the diffusion model.
        """
        ...
    
    def forward_diffusion(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        Add noise to the input data according to a noise schedule.
        """
        ...

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        This is an alias for `forward_diffusion`.
        """
        return self.forward_diffusion(x_0, t, noise)


Diffusion = BaseDiffusionAlgorithm