"""
Configuration for MDLM (Masked Diffusion Language Model) training.
"""

from dataclasses import dataclass, field
from datetime import datetime
import os


@dataclass
class MDLMTrainingConfig:
    # Dataset
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    tokenizer: str = "gpt2"
    seq_length: int = 128
    max_train_examples: int = None  # None = use all available
    val_split: float = 0.05

    # Model architecture
    net: str = "dit_text_s"  # dit_text_t, dit_text_s, dit_text_b
    hidden_size: int = None  # Override default for the model size
    depth: int = None  # Override default
    num_heads: int = None  # Override default
    dropout: float = 0.1
    time_conditioning: bool = True

    # MDLM-specific settings
    sampling_eps: float = 1e-3  # Minimum time value (avoid t=0)
    antithetic_sampling: bool = True  # Stratified time sampling

    # Sampling settings
    sampling_steps: int = 128
    sampling_strategy: str = "ddpm_cache"  # 'ddpm_cache' or 'topk'
    sampling_temperature: float = 1.0

    # Training loop and optimizer
    compile: bool = False
    fp16: bool = False
    total_steps: int = 50000
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    lr_min: float = 1e-6
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # EMA
    use_ema: bool = True
    ema_beta: float = 0.9999
    ema_start_step: int = 0

    # Logging and evaluation
    log_every: int = 50
    sample_every: int = 2000
    num_samples_for_logging: int = 4
    validate_every: int = 1000
    save_every: int = 10000

    # Accelerator
    device: str = "cuda:0"

    # Logging
    logger: str = None  # "wandb" or None
    checkpoint_dir: str = "logs/mdlm_train"
    min_steps_for_final_save: int = 100
    cache_dir: str = field(default_factory=lambda: f"{os.path.expanduser('~')}/.cache")

    def update_checkpoint_dir(self):
        self.checkpoint_dir = f"{self.checkpoint_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    def __post_init__(self):
        self.update_checkpoint_dir()
