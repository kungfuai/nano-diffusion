from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 256  # batch size
    learning_rate: float = 5e-4  # initial learning rate
    weight_decay: float = 1e-6  # weight decay
    num_denoising_steps: int = 1000  # number of timesteps