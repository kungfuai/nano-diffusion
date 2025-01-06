from dataclasses import dataclass
from datetime import datetime
import os


@dataclass
class DiffusionTrainingConfig:
    # Dataset
    dataset: str  # dataset name
    resolution: int  # resolution of the image

    # Model architecture
    # TODO: the current assumption is that the data dimension is [in_channels, resolution, resolution]
    #   This can be more flexible.
    in_channels: int = 3  # number of input channels
    resolution: int = 64  # resolution of the image
    net: str = "unet_small"  # network architecture
    num_denoising_steps: int = 1000  # number of timesteps

    # VAE (tokenizer) for compression
    vae_use_fp16: bool = True  # use fp16 for the VAE
    vae_model_name: str = "madebyollin/sdxl-vae-fp16-fix"  # VAE model name
    vae_scale_factor: float = 0.18215  # scale factor for the VAE encoding outputs (so that the std is close to 1)

    # Text conditioning
    text_drop_prob: float = 1  # probability of dropping text conditioning during training

    # Training loop and optimizer
    total_steps: int = 100000  # total number of training steps
    batch_size: int = 16  # batch size
    learning_rate: float = 1e-4  # initial learning rate
    weight_decay: float = 0.0  # weight decay
    lr_min: float = 1e-6  # minimum learning rate
    warmup_steps: int = 1000  # number of warmup steps

    # Logging and evaluation
    log_every: int = 50  # log every N steps
    sample_every: int = 1000  # sample every N steps
    num_samples_for_logging: int = 8  # number of samples for periodical generation and logging
    validate_every: int = 1000  # compute validation loss every N steps
    save_every: int = 10000  # save model every N steps
    fid_every: int = 5000  # compute FID every N steps
    num_samples_for_fid: int = 1000  # number of samples for FID
    num_real_samples_for_fid: int = 100  # number of real samples for FID when not using CIFAR

    # Sampling
    clip_sample_range: float = 1.0  # range for clipping sample. If 0 or less, no clipping

    # Regularization
    max_grad_norm: float = -1  # maximum norm for gradient clipping
    use_loss_mean: bool = False  # use loss.mean() instead of just loss
    use_ema: bool = False  # use EMA for the model
    ema_beta: float = 0.9999  # EMA decay factor
    ema_start_step: int = 0  # step to start EMA update

    # Accelerator
    device: str = "cuda"  # device to use for training

    # Logging
    logger: str = "wandb"  # logging method
    cache_dir: str = f"{os.path.expanduser('~')}/.cache" # cache directory in the home directory, same across runs
    checkpoint_dir: str = "logs/train"  # checkpoint directory
    min_steps_for_final_save: int = 100  # minimum steps for final save
    watch_model: bool = False  # watch the model with wandb
    init_from_wandb_run_path: str = (
        None  # resume model from a wandb run path "user/project/run_id"
    )
    init_from_wandb_file: str = None  # resume model from a wandb file "path/to/file"

    # Data augmentation
    random_flip: bool = False  # randomly flip images horizontally

    def update_checkpoint_dir(self):
        # Update the checkpoint directory use a timestamp
        self.checkpoint_dir = f"{self.checkpoint_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    def __post_init__(self):
        self.update_checkpoint_dir()
