from dataclasses import dataclass
import os
from typing import List, Tuple
from datetime import datetime


@dataclass
class TrainingConfig:
    dataset: str = "mj_latents"  # dataset name
    val_split: float = 0.1  # fraction of the dataset to use for validation
    data_is_latent: bool = True  # whether the data is already in latent space
    in_channels: int = 4  # number of input channels
    resolution: int = 64  # resolution
    data_shape: List[int] = None  # data shape (e.g., [3, 64, 64] for RGB images, [16, 3, 64, 64] for video). When specified, this overrides in_channels and resolution.
    caption_column: str = None  # the column that contains the captions
    checkpoint_dir: str = "checkpoints"  # directory for checkpoints
    cache_dir: str = f"{os.path.expanduser('~')}/.cache" # cache directory in the home directory, same across runs
    logger: str = None  # "wandb"  # logger

    diffusion_algo: str = "vdm"  # diffusion algorithm

    # Conditioning
    conditional: bool = True  # whether to use conditional training
    cond_embed_dim: int = 768  # dimension of the conditioning embedding (before the projection layer)
    cond_drop_prob: float = 0.2  # probability of dropping conditioning during training
    guidance_scale: float = 4.5  # guidance scale for classifier-free guidance
    text_embed_dim: int = 768  # text embedding dimension

    # VAE (tokenizer) for compression
    vae_use_fp16: bool = False  # use fp16 for the VAE
    vae_model_name: str = "madebyollin/sdxl-vae-fp16-fix"  # VAE model name
    vae_scale_multiplier: float = 0.18215  # scale multiplier for the VAE encoding outputs (so that the std is close to 1)

    random_flip: bool = False  # random flip

    # Training loop and optimizer
    compile: bool = False  # whether to compile the model
    accelerator: bool = False  # whether to use the accelerator utility
    fp16: bool = False  # whether to use fp16 mixed precision for training
    batch_size: int = 256  # batch size
    learning_rate: float = 5e-4  # initial learning rate
    lr_min: float = 1e-6  # minimum learning rate
    weight_decay: float = 1e-6  # weight decay
    num_denoising_steps: int = 1000  # number of timesteps
    warmup_steps: int = 1000  # number of warmup steps
    total_steps: int = 10000  # total number of training steps
    device: str = "cuda"  # device

    # Bookkeeping
    log_every: int = 20  # log every n steps
    validate_every: int = 500  # validate every n steps
    sample_every: int = 2000  # sample every n steps
    save_every: int = 10000  # save every n steps
    fid_every: int = 5000  # compute FID every n steps
    num_samples_for_fid: int = 1000  # number of samples for FID
    num_real_samples_for_fid: int = 100  # number of real samples for FID when not using CIFAR
    num_samples_for_logging: int = 8  # number of samples for logging
    log_grad_norm: bool = False  # whether to log the gradient norm
    min_steps_for_final_save: int = 100  # minimum steps for final save

    # Sampling
    clip_sample_range: float = 1.0  # clip sample range
    n_samples: int = 8  # number of samples to generate

    max_grad_norm: float = -1  # maximum norm for gradient clipping

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
