"""
A minimal training pipeline for diffusion.

## Main components

1. Forward Diffusion Process:

Adds progressively more Gaussian noise to the data according to a noise schedule β.

2. Reverse Process:

A neural network is trained to predict the noise at each time step and is used to denoise the data in the reverse direction, step by step.

3. Loss Function:

The training is supervised by minimizing the difference (mean squared error) between the actual noise added in the forward process and the predicted noise.

4. Noise Schedule:

The variance β at each timestep defines how much noise to add. It can be a fixed schedule, e.g., linearly increasing over time.

"""

import argparse

from src.config.diffusion_training_config import DiffusionTrainingConfig as TrainingConfig, diffusion_algo_choices
from src.datasets import load_data
from src.diffusion.diffusion_model_components import create_diffusion_model_components
from src.diffusion.diffusion_training_loop import training_loop
from src.models.factory import choices
from src.bookkeeping import parse_shape
from src.bookkeeping.mini_batch import collate_fn_for_latents


def parse_arguments():
    parser = argparse.ArgumentParser(description="DDPM training")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="cifar10",
        help="Dataset to use: e.g. cifar10, flowers, celeb, pokemon, or any huggingface dataset that has an image field.",
    )
    parser.add_argument(
        "--data_is_latent",
        action="store_true",
        help="Whether the data is already in latent space",
    )
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Whether the model is conditional",
    )
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--resolution", type=int, default=32, help="Resolution of the image. Only used for unet.")
    parser.add_argument(
        "--data_shape",
        type=parse_shape,  # Custom type function
        help="Comma-separated input shape (e.g., '3,32,32' for RGB images, '16,3,64,64' for video). When specified, this overrides in_channels and resolution.",
    )
    parser.add_argument(
        "--cond_embed_dim",
        type=int,
        default=768,
        help="Dimension of the conditioning embedding. This assume there is only one conditioning embedding (e.g. text prompt).",
    )
    parser.add_argument(
        "--logger",
        type=str,
        choices=["wandb", "none"],
        default="none",
        help="Logging method",
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=choices(),
        default="unet_small",
        help="Network architecture",
    )
    parser.add_argument(
        "--num_denoising_steps",
        type=int,
        default=1000,
        help="Number of timesteps in the diffusion process",
    )
    parser.add_argument(
        "-a",
        "--diffusion_algo",
        type=str,
        choices=diffusion_algo_choices(),
        default="ddpm",
        help="Denoising algorithm",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 for training",
    )
    parser.add_argument(
        "--accelerator",
        action="store_true",
        help="Use the accelerator utility",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of the dataset to use for validation",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1200, help="Number of warmup steps"
    )
    parser.add_argument(
        "--total_steps", type=int, default=10100, help="Total number of training steps"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument(
        "--lr_min", type=float, default=2e-6, help="Minimum learning rate"
    )
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    parser.add_argument(
        "--sample_every", type=int, default=1500, help="Sample every N steps"
    )
    parser.add_argument(
        "--save_every", type=int, default=60000, help="Save model every N steps"
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=1500,
        help="Compute validation loss every N steps",
    )
    parser.add_argument(
        "--fid_every", type=int, default=6000, help="Compute FID every N steps"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )
    parser.add_argument(
        "--clip_sample_range",
        type=float,
        default=1.0,
        help="Range for clipping sample",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=-1,
        help="Maximum norm for gradient clipping",
    )
    parser.add_argument(
        "--use_loss_mean",
        action="store_true",
        help="Use loss.mean() instead of just loss",
    )
    parser.add_argument(
        "--watch_model", action="store_true", help="Use wandb to watch the model"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use Exponential Moving Average (EMA) for the model",
    )
    parser.add_argument(
        "--ema_beta", type=float, default=0.999, help="EMA decay factor"
    )
    parser.add_argument(
        "--ema_start_step", type=int, default=2000, help="Step to start EMA update"
    )
    parser.add_argument(
        "--random_flip", action="store_true", help="Randomly flip images horizontally"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="logs/train", help="Checkpoint directory"
    )
    parser.add_argument(
        "--init_from_wandb_run_path",
        type=str,
        default=None,
        help="Resume from a wandb run path (run id)",
    )
    parser.add_argument(
        "--init_from_wandb_file",
        type=str,
        default=None,
        help="Resume from a wandb file (a model checkpoint inside the run)",
    )
    parser.add_argument(
        "--num_samples_for_fid",
        type=int,
        default=1000,
        help="Number of samples to use for FID computation",
    )
    parser.add_argument(
        "--num_real_samples_for_fid",
        type=int,
        default=10000,
        help="Number of real samples to use for FID computation",
    )
    parser.add_argument(
        "--num_samples_for_logging",
        type=int,
        default=8,
        help="Number of samples to use for logging",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    config = TrainingConfig(**vars(args))

    train_dataloader, val_dataloader = load_data(config, collate_fn=collate_fn_for_latents if config.data_is_latent else None)
    model_components = create_diffusion_model_components(config)

    num_examples_trained = training_loop(
        model_components, train_dataloader, val_dataloader, config
    )

    print(f"Training completed. Total examples trained: {num_examples_trained}")


if __name__ == "__main__":
    """
    Main variables of the recipe:
        - data_is_latent: whether the data is already in latent space.
            If true, a VAE decoder will be needed to decode the generated latents to the raw data space (e.g. images).
        - conditional: whether the model is conditional.
            If true, the model will require an additional input `y` for conditioning (e.g. embedding of a text prompt).
        - diffusion algorithm: DDPM, VDM
            - parameters of the forward diffusion process: num_denoising_steps, noise_schedule
            - parameters of the sampler: num_denoising_steps, guidance_scale, clip_sample, clip_sample_range
        - network architecture: unet_small, unet, unet_big, dit_s2, etc.
        - other common deep learning hyperparameters: learning rate, batch size
    """
    main()
