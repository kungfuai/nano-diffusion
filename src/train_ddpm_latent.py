"""
Train a DDPM model on image latents and text embeddings.
"""
import argparse
import torch
import numpy as np
from src.config.diffusion_training_config import DiffusionTrainingConfig as TrainingConfig
from src.datasets import load_data
from src.diffusion.diffusion_model_components import create_latent_diffusion_model_components
from src.diffusion.latent_diffusion_training_loop import training_loop
from src.models.factory import choices
from src.bookkeeping.mini_batch import MiniBatch


def parse_arguments():
    parser = argparse.ArgumentParser(description="DDPM training for image latents and text embeddings")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="zzsi/afhq64_16k_latents_sdxl_blip2",
        help="Latent dataset to use.",
    )
    parser.add_argument("--in_channels", type=int, default=4, help="Number of input channels. For Stable Diffusion VAEs, this should be 4.")
    parser.add_argument("--resolution", type=int, default=8, help="Resolution of the image latents.")
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
        "--cond_embed_dim",
        type=int,
        default=None,
        help="Dimension of the conditioning embedding (before the projection layer). This is required when training a conditional model.",
    )
    parser.add_argument(
        "--cond_drop_prob",
        type=float,
        default=0.2,
        help="Probability of dropping conditioning during training",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1200, help="Number of warmup steps"
    )
    parser.add_argument(
        "--total_steps", type=int, default=100000, help="Total number of training steps"
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
        "--save_every", type=int, default=50000, help="Save model every N steps"
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=1500,
        help="Compute validation loss every N steps",
    )
    parser.add_argument(
        "--fid_every", type=int, default=10000, help="Compute FID every N steps"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )
    parser.add_argument(
        "--clip_sample_range",
        type=float,
        default=2.0,
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
        help="Use loss.mean() instead of just loss. This will be deprecated.",
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
        "--random_flip", action="store_true", help="Randomly flip images horizontally. Do not use it for image latents."
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


def collate_fn(batch):
    assert "image_emb" in batch[0], f"Data must be a dict that contains 'image_emb'. Got {type(batch[0])}"
    # print(batch[0]['image_emb'])
    # assert np.array(batch[0]['image_emb']).shape == (4, 8, 8), f"Image emb shape is {np.array(batch[0]['image_emb']).shape}. Expected (4, 8, 8)."
    data = {
        'image_emb': torch.stack([torch.from_numpy(np.array(item['image_emb'])) for item in batch]),
    }
    if "text_emb" in batch[0]:
        data["text_emb"] = torch.stack([torch.from_numpy(np.array(item["text_emb"])) for item in batch])
    return data


def main():
    args = parse_arguments()
    config = TrainingConfig(**vars(args))

    train_dataloader, val_dataloader = load_data(config, collate_fn=collate_fn)
    model_components = create_latent_diffusion_model_components(config)

    num_examples_trained = training_loop(
        model_components, train_dataloader, val_dataloader, config
    )

    print(f"Training completed. Total examples trained: {num_examples_trained}")


if __name__ == "__main__":
    main()
