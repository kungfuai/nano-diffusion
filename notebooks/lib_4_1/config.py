from dataclasses import dataclass


@dataclass
class TrainingConfig:
    dataset: str = "reese-green/afhq64_captions_64k"  # dataset name
    caption_column: str = "caption_blip2-opt-2.7b"  # the column that contains the captions
    checkpoint_dir: str = "checkpoints"  # directory for checkpoints
    logger: str = None  # "wandb"  # logger

    batch_size: int = 256  # batch size
    learning_rate: float = 5e-4  # initial learning rate
    weight_decay: float = 1e-6  # weight decay
    num_denoising_steps: int = 1000  # number of timesteps
    device: str = "cuda"  # device
    resolution: int = 64  # resolution
    text_embed_dim: int = 768  # text embedding dimension
    text_drop_prob: float = 0.2  # text drop probability
    random_flip: bool = False  # random flip

    # Bookkeeping
    log_every: int = 20  # log every n steps
    validate_every: int = 500  # validate every n steps
    sample_every: int = 2000  # sample every n steps
    save_every: int = 10000  # save every n steps

    # Sampling
    clip_sample_range: float = 1.0  # clip sample range
    n_samples: int = 8  # number of samples to generate