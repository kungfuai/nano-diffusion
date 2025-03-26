from typing import Tuple, Dict
import torch
from ..bookkeeping.mini_batch import MiniBatch


class BaseDiffusionAlgorithm:
    """
    A common interface for diffusion algorithms.
    """
    def prepare_training_examples(self, batch: MiniBatch, **kwargs) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Prepare a training example for the denoising model by adding noise to the input data.

        This is used in the training step.

        For diffusion, this is the forward diffusion process.
        """
        ...
    
    def sample(self, x_T, y = None, guidance_scale: float = None, seed: int = None, **kwargs):
        """
        Sample from the denoising model. For diffusion, this is the reverse diffusion process.
        """
        ...


Diffusion = BaseDiffusionAlgorithm
