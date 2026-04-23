"""
Consistency Distillation training for nano-diffusion.

Distills a pre-trained diffusion model (teacher) into a consistency model (student)
that can generate high-quality samples in 1-4 denoising steps.

Usage:
    # Distill from a trained DDPM checkpoint:
    python -m train_distill --teacher_checkpoint logs/train/final_model.pth \
        --dataset cifar10 --net unet_small --total_steps 10000

    # Distill with latent diffusion:
    python -m train_distill --teacher_checkpoint logs/train/final_model.pth \
        --dataset my_latent_dataset --data_is_latent --net dit_s2 \
        --resolution 32 --in_channels 4

Reference: https://arxiv.org/abs/2310.04378
"""

import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from nanodiffusion.bookkeeping.mini_batch import MiniBatch, collate_fn_for_latents
from nanodiffusion.config.diffusion_training_config import DiffusionTrainingConfig
from nanodiffusion.datasets import load_data
from nanodiffusion.diffusion.consistency_distillation import ConsistencyDistillation
from nanodiffusion.diffusion.noise_scheduler import create_noise_schedule
from nanodiffusion.models.factory import choices, create_model
from nanodiffusion.optimizers.lr_schedule import get_cosine_schedule_with_warmup
from nanodiffusion.bookkeeping import parse_shape


def parse_arguments():
    parser = argparse.ArgumentParser(description="Consistency Distillation Training")

    # Teacher model
    parser.add_argument(
        "--teacher_checkpoint", type=str, required=True,
        help="Path to pre-trained teacher model checkpoint (.pth)",
    )

    # Dataset
    parser.add_argument("-d", "--dataset", type=str, default="cifar10")
    parser.add_argument("--data_is_latent", action="store_true")
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument(
        "--data_shape", type=parse_shape,
        help="Comma-separated input shape (e.g., '3,32,32')",
    )
    parser.add_argument("--cond_embed_dim", type=int, default=768)
    parser.add_argument("--val_split", type=float, default=0.1)

    # Model
    parser.add_argument("--net", type=str, choices=choices(), default="unet_small")

    # Diffusion settings
    parser.add_argument("--num_denoising_steps", type=int, default=1000)

    # Distillation-specific
    parser.add_argument(
        "--num_ddim_timesteps", type=int, default=50,
        help="Number of DDIM sub-steps for the teacher ODE solver schedule",
    )
    parser.add_argument(
        "--guidance_scale_min", type=float, default=3.0,
        help="Min classifier-free guidance scale (sampled per batch)",
    )
    parser.add_argument(
        "--guidance_scale_max", type=float, default=13.0,
        help="Max classifier-free guidance scale (sampled per batch)",
    )
    parser.add_argument(
        "--target_ema_decay", type=float, default=0.95,
        help="EMA decay for the target model",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=4,
        help="Number of steps for sample generation during evaluation",
    )

    # Training
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--accelerator", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr_min", type=float, default=2e-6)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--clip_sample_range", type=float, default=2.0)

    # Logging
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=1500)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--checkpoint_dir", type=str, default="logs/distill")
    parser.add_argument("--num_samples_for_logging", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--random_flip", action="store_true")

    return parser.parse_args()


def load_teacher_model(checkpoint_path: str, net: str, in_channels: int,
                       resolution: int, cond_embed_dim: int,
                       device: str) -> Module:
    """Load the pre-trained teacher model from a checkpoint."""
    model = create_model(
        net=net, in_channels=in_channels, resolution=resolution,
        cond_embed_dim=cond_embed_dim,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded teacher model from {checkpoint_path}")
    return model


def distillation_training_loop(
    student_model: Module,
    teacher_model: Module,
    consistency_distillation: ConsistencyDistillation,
    optimizer: Optimizer,
    lr_scheduler,
    train_dataloader: DataLoader,
    args,
):
    """Main training loop for consistency distillation."""
    device = args.device
    checkpoint_dir = Path(args.checkpoint_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    if args.logger == "wandb":
        import wandb
        import os
        project_name = os.getenv("WANDB_PROJECT") or "nano-diffusion"
        wandb.init(
            project=project_name,
            config={
                "method": "consistency_distillation",
                "teacher_checkpoint": args.teacher_checkpoint,
                "net": args.net,
                "dataset": args.dataset,
                "total_steps": args.total_steps,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_ddim_timesteps": args.num_ddim_timesteps,
                "target_ema_decay": args.target_ema_decay,
                "num_inference_steps": args.num_inference_steps,
            },
        )

    accelerator = None
    if args.accelerator:
        from accelerate import Accelerator
        accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
        student_model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
            student_model, train_dataloader, optimizer, lr_scheduler
        )

    step = 0
    num_examples_trained = 0

    while step < args.total_steps:
        for batch in train_dataloader:
            if step >= args.total_steps:
                break

            batch = MiniBatch.from_dataloader_batch(batch)
            num_examples_trained += batch.num_examples

            student_model.train()

            context = accelerator.accumulate() if accelerator else nullcontext()
            with context:
                optimizer.zero_grad()

                # Compute consistency distillation loss
                examples = consistency_distillation.prepare_training_examples(batch)
                loss = consistency_distillation.consistency_distillation_loss(
                    x_0=examples["x_0"],
                    y=examples["y"],
                    p_uncond=examples["p_uncond"],
                )

                if accelerator:
                    accelerator.backward(loss)
                else:
                    loss.backward()

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(), args.max_grad_norm
                    )

                optimizer.step()

            lr_scheduler.step()

            # Update target model EMA
            with torch.no_grad():
                consistency_distillation.update_target_model(args.target_ema_decay)

            student_model.eval()

            # Logging
            with torch.no_grad():
                if args.log_every > 0 and step % args.log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"Step: {step}, Examples: {num_examples_trained}, "
                        f"Loss: {loss.item():.4f}, LR: {lr:.6f}"
                    )
                    if args.logger == "wandb":
                        import wandb
                        wandb.log({
                            "loss": loss.item(),
                            "learning_rate": lr,
                            "num_examples_trained": num_examples_trained,
                            "num_batches_trained": step,
                        }, step=step)

                if args.sample_every > 0 and step % args.sample_every == 0 and step > 0:
                    generate_distilled_samples(
                        consistency_distillation, args, step, checkpoint_dir,
                    )

                if args.save_every > 0 and step % args.save_every == 0 and step > 0:
                    save_path = checkpoint_dir / f"student_model_step_{step}.pth"
                    torch.save(student_model.state_dict(), save_path)
                    print(f"Student model saved at {save_path}")

                    target_path = checkpoint_dir / f"target_model_step_{step}.pth"
                    torch.save(
                        consistency_distillation.target_model.state_dict(), target_path
                    )
                    print(f"Target model saved at {target_path}")

            step += 1

    # Final save
    if step > 100:
        final_path = checkpoint_dir / "final_student_model.pth"
        torch.save(student_model.state_dict(), final_path)
        print(f"Final student model saved at {final_path}")

        final_target_path = checkpoint_dir / "final_target_model.pth"
        torch.save(consistency_distillation.target_model.state_dict(), final_target_path)
        print(f"Final target (EMA) model saved at {final_target_path}")

    if accelerator:
        accelerator.end_training()

    return num_examples_trained


def generate_distilled_samples(
    consistency_distillation: ConsistencyDistillation, args, step: int,
    checkpoint_dir: Path,
):
    """Generate and log samples from the distilled model."""
    device = args.device
    n_samples = args.num_samples_for_logging
    in_channels = args.in_channels
    resolution = args.resolution

    if args.data_shape:
        data_dim = tuple(args.data_shape)
    else:
        data_dim = (in_channels, resolution, resolution)

    torch.manual_seed(0)
    x_T = torch.randn(n_samples, *data_dim).to(device)

    sampled_images = consistency_distillation.sample(
        x_T, num_inference_steps=args.num_inference_steps, seed=0, quiet=True,
    )

    if args.logger == "wandb":
        import wandb
        images_np = (
            (sampled_images * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")
        )
        wandb.log({
            "distilled_samples_step": step,
            "distilled_samples": [wandb.Image(img) for img in images_np],
        })
    else:
        grid = make_grid(sampled_images, nrow=4, normalize=False)
        save_image(grid, checkpoint_dir / f"distilled_sample_step_{step}.png")
        print(f"Saved distilled samples at step {step}")


def main():
    args = parse_arguments()

    device = args.device

    # Build config for data loading
    config = DiffusionTrainingConfig(
        dataset=args.dataset,
        data_is_latent=args.data_is_latent,
        conditional=args.conditional,
        in_channels=args.in_channels,
        resolution=args.resolution,
        data_shape=args.data_shape,
        cond_embed_dim=args.cond_embed_dim,
        val_split=args.val_split,
        batch_size=args.batch_size,
        random_flip=args.random_flip,
        device=device,
    )

    # Load data
    collate_fn = collate_fn_for_latents if args.data_is_latent else None
    train_dataloader, val_dataloader = load_data(config, collate_fn=collate_fn)

    # Load teacher
    cond_dim = args.cond_embed_dim if args.conditional else None
    teacher_model = load_teacher_model(
        args.teacher_checkpoint, args.net, args.in_channels,
        args.resolution, cond_dim, device,
    )

    # Create student (same architecture, initialized from teacher)
    student_model = create_model(
        net=args.net, in_channels=args.in_channels,
        resolution=args.resolution, cond_embed_dim=cond_dim,
    ).to(device)
    student_model.load_state_dict(teacher_model.state_dict())
    print("Student model initialized from teacher weights")

    if args.compile:
        student_model = torch.compile(student_model)

    # Create noise schedule
    noise_schedule = create_noise_schedule(args.num_denoising_steps, torch.device(device))

    # Create VAE if needed
    vae = None
    if args.data_is_latent:
        from nanodiffusion.diffusion.diffusion_model_components import create_vae_if_data_is_latent
        vae = create_vae_if_data_is_latent(config)

    # Create consistency distillation algorithm
    consistency_distillation = ConsistencyDistillation(
        student_model=student_model,
        teacher_model=teacher_model,
        noise_schedule=noise_schedule,
        device=device,
        num_denoising_steps=args.num_denoising_steps,
        num_ddim_timesteps=args.num_ddim_timesteps,
        guidance_scale_range=(args.guidance_scale_min, args.guidance_scale_max),
        data_is_latent=args.data_is_latent,
        vae_scale_multiplier=config.vae_scale_multiplier,
        conditional=args.conditional,
        cond_drop_prob=config.cond_drop_prob,
        clip_sample_range=args.clip_sample_range,
        vae=vae,
    )

    # Optimizer and scheduler (only for student)
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
        lr_min=args.lr_min,
    )

    # Run distillation
    num_examples = distillation_training_loop(
        student_model=student_model,
        teacher_model=teacher_model,
        consistency_distillation=consistency_distillation,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        args=args,
    )

    print(f"Distillation completed. Total examples trained: {num_examples}")


if __name__ == "__main__":
    main()
