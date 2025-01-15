from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import List, Tuple


def diffusion_algo_choices():
    return ["ddpm", "vdm"]

@dataclass
class DiffusionTrainingConfig:
    # Dataset
    dataset: str  # dataset name
    data_is_latent: bool = False  # whether the data is already in latent space
    in_channels: int = 3  # number of input channels
    resolution: int = 64  # resolution of the image
    data_shape: List[int] = None  # data shape (e.g., [3, 64, 64] for RGB images, [16, 3, 64, 64] for video). When specified, this overrides in_channels and resolution.

    # Model architecture
    net: str = "unet_small"  # network architecture

    # Denoising and diffusion settings
    diffusion_algo: str = field(default="ddpm", metadata={"choices": diffusion_algo_choices()})  # denoising algorithm
    num_denoising_steps: int = 1000  # number of timesteps
    vdm_beta_a: float = 1.0  # alpha parameter for the noise level distribution
    vdm_beta_b: float = 2.5  # beta parameter for the noise level distribution

    # VAE (tokenizer) for compression
    vae_use_fp16: bool = False  # use fp16 for the VAE
    vae_model_name: str = "madebyollin/sdxl-vae-fp16-fix"  # VAE model name
    vae_scale_multiplier: float = 0.18215  # scale multiplier for the VAE encoding outputs (so that the std is close to 1)

    # Conditioning
    conditional: bool = False  # whether to use conditional training
    cond_embed_dim: int = 768  # dimension of the conditioning embedding (before the projection layer)
    cond_drop_prob: float = 0.2  # probability of dropping conditioning during training
    guidance_scale: float = 4.5  # guidance scale for classifier-free guidance
    
    # Training loop and optimizer
    compile: bool = False  # whether to compile the model
    accelerator: bool = False  # whether to use the accelerator utility
    fp16: bool = False  # whether to use fp16 mixed precision for training
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
    log_grad_norm: bool = False  # whether to log the gradient norm

    # Sampling
    clip_sample_range: float = 2.0  # range for clipping sample. If 0 or less, no clipping

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

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if self.resolution is not None:
            return (self.in_channels, self.resolution, self.resolution)
        return tuple(self.data_shape)
    
    def __post_init__(self):
        self.update_checkpoint_dir()

        if self.resolution is not None:
            if self.in_channels is None:
                raise ValueError("If resolution is specified, in_channels must also be specified")
        elif self.data_shape is None:
            raise ValueError("Must specify either resolution or data_shape")
        
        if self.data_shape is not None:
            print(f"Overriding in_channels and resolution from data_shape: {self.data_shape}")
            # Override the in_channels and resolution from the data_shape
            self.in_channels = self.data_shape[0]
            if len(self.data_shape) == 3:
                self.resolution = self.data_shape[1]
            else:
                self.resolution = None
        
            # Assert the resolution, in_channels, and data_shape are consistent
            if len(self.data_shape) == 3:
                assert self.in_channels == self.data_shape[0], f"in_channels {self.in_channels} does not match data_shape {self.data_shape}"
                assert self.resolution == self.data_shape[1], f"resolution {self.resolution} does not match data_shape {self.data_shape}"
                assert self.resolution == self.data_shape[2], f"resolution {self.resolution} does not match data_shape {self.data_shape}"
            # TODO: Add other assertions here
        
        # Process accelerator
        if self.fp16:
            if not self.accelerator:
                raise ValueError("fp16 is only supported with accelerator. Please pass in --accelerator as an argument when using --fp16.")
