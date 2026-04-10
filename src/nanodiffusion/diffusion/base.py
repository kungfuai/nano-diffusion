from typing import Tuple, Dict
import torch
from ..bookkeeping.mini_batch import MiniBatch


class BaseDiffusionAlgorithm:
    """
    A common interface for diffusion algorithms.
    """
    def prepare_training_examples(self, batch: MiniBatch, **kwargs) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Prepare the per-step supervised training problem for the denoising model.

        The full sample in `batch` is the final target we eventually want to generate.
        Training does not supervise the whole multi-step generation process at once.
        Instead, it samples one generation turn (for example, one denoising step),
        constructs the model inputs for that turn, and returns the target for that
        turn.

        This is conceptually similar to autoregressive training, where a full
        sequence is converted into next-token prediction examples. Here, a full
        sample is converted into one-step denoising or generation examples.

        For diffusion algorithms, this usually means running the forward
        noising process to create `x_t` and the corresponding training target.
        """
        ...
    
    def sample(self, x_T, y = None, guidance_scale: float = None, seed: int = None, **kwargs):
        """
        Sample from the denoising model. For diffusion, this is the reverse diffusion process.
        """
        ...


Diffusion = BaseDiffusionAlgorithm
