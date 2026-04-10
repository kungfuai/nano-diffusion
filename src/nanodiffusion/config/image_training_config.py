from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Protocol


class SupportsImageDataConfig(Protocol):
    dataset: str
    resolution: int
    val_split: float
    batch_size: int
    random_flip: bool
    data_is_latent: bool


@dataclass
class ImageTrainingConfig:
    dataset: str
    resolution: int
    val_split: float = 0.1
    in_channels: int = 3
    device: str = "cuda:0"
    logger: str = "none"
    cache_dir: str = field(default_factory=lambda: f"{os.path.expanduser('~')}/.cache")
    checkpoint_dir: str = "logs/train"
    min_steps_for_final_save: int = 100
    watch_model: bool = False
    init_from_wandb_run_path: str = None
    init_from_wandb_file: str = None
    random_flip: bool = False
    data_is_latent: bool = False

    def update_checkpoint_dir(self):
        self.checkpoint_dir = (
            f"{self.checkpoint_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )

    def __post_init__(self):
        self.update_checkpoint_dir()
