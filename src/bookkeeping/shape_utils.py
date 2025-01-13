import argparse
from typing import List
from .mini_batch import MiniBatch
from ..config.diffusion_training_config import DiffusionTrainingConfig as TrainingConfig


def parse_shape(s: str) -> List[int]:
    try:
        return [int(x) for x in s.split(',')]
    except:
        raise argparse.ArgumentTypeError("Data shape must be comma-separated integers (e.g., '3,32,32')")


def check_model_input_shapes(mini_batch: MiniBatch, config: TrainingConfig):
    """Validates inputs match configuration"""
    if config.use_text_conditioning and mini_batch.text_emb is None:
        raise ValueError("Text conditioning enabled but no text embeddings provided")
    
    if config.use_latent_diffusion:
        expected_shape = (config.batch_size, config.latent_channels, 
                         config.latent_size, config.latent_size)
        if mini_batch.x.shape != expected_shape:
            raise ValueError(f"Expected latent shape {expected_shape}, got {mini_batch.x.shape}")