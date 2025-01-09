from dataclasses import asdict
from pathlib import Path
from typing import Callable
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from ..config.diffusion_training_config import DiffusionTrainingConfig
from ..diffusion import generate_conditional_samples_by_denoising, compute_validation_loss_for_latents
from ..diffusion.diffusion_model_components import LatentDiffusionModelComponents
from ..eval.fid import compute_fid


# TODO: generate_samples_by_denoising should use text_emb if it's provided


class LatentDiffusionBookkeeping:
    def __init__(self, config: DiffusionTrainingConfig, model_components: LatentDiffusionModelComponents): #, denoising_model: nn.Module, noise_schedule: Dict):
        self.config = config
        self.model_components = model_components
        self.num_examples_trained = 0

    def set_up_logger(self):
        # create directory for checkpoints
        print(f"Creating checkpoint directory: {self.config.checkpoint_dir}")
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Setting up logger: {self.config.logger}")
        params = sum(p.numel() for p in self.model_components.denoising_model.parameters())
        if self.config.logger == "wandb":
            import wandb

            project_name = os.getenv("WANDB_PROJECT") or "nano-diffusion"
            print(f"Logging to Weights & Biases project: {project_name}")
            run_params = asdict(self.config)
            run_params["model_parameters"] = params
            wandb.init(project=project_name, config=run_params)
            if self.config.watch_model:
                print("  Watching model gradients (can be slow)")
                wandb.watch(self.model_components.denoising_model)
        
    def run_callbacks(self, config: DiffusionTrainingConfig, model_components: LatentDiffusionModelComponents, step: int, loss: float, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader, val_dataloader: DataLoader):
        self.num_examples_trained += config.batch_size

        with torch.no_grad():
            if step % config.log_every == 0:
                log_training_step(
                    step, self.num_examples_trained, loss, optimizer, config.logger
                )

            if step % config.validate_every == 0 and step > 0 and val_dataloader:
                validate_and_log(
                    compute_validation_loss_for_latents,
                    model_components,
                    val_dataloader,
                    config,
                )

            if step % config.sample_every == 0:
                generate_and_log_samples(model_components, config, step, val_dataloader)

            if step % config.save_every == 0 and step > 0:
                save_checkpoints(model_components, step, config)

            if step % config.fid_every == 0 and step > 0:
                compute_and_log_fid(model_components, config, train_dataloader, val_dataloader)


def log_training_step(
    step: int,
    num_examples_trained: int,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    logger: str,
):
    lr = optimizer.param_groups[0]["lr"]
    if logger == "wandb":
        import wandb

        wandb.log(
            {
                "num_examples_trained": num_examples_trained,
                "num_batches_trained": step,
                "loss": loss.item(),
                "learning_rate": lr,
            }
        )

    print(
        f"Step: {step}, Examples: {num_examples_trained}, Loss: {loss.item():.4f}, LR: {lr:.6f}"
    )


def validate_and_log(
    compute_validation_loss_for_latents: Callable,
    model_components: LatentDiffusionModelComponents,
    val_dataloader: DataLoader,
    config: DiffusionTrainingConfig,
):
    val_loss = compute_validation_loss_for_latents(
        denoising_model=model_components.denoising_model,
        val_dataloader=val_dataloader,
        noise_schedule=model_components.noise_schedule,
        n_T=config.num_denoising_steps,
        device=config.device,
        config=config,
    )
    if config.use_ema:
        ema_val_loss = compute_validation_loss_for_latents(
            denoising_model=model_components.ema_model,
            val_dataloader=val_dataloader,
            noise_schedule=model_components.noise_schedule,
            n_T=config.num_denoising_steps,
            device=config.device,
            config=config,
        )

    if config.logger == "wandb":
        import wandb
        
        log_dict = {"val_loss": val_loss}
        if config.use_ema:
            log_dict["ema_val_loss"] = ema_val_loss
        wandb.log(log_dict)
    else:
        print(f"Validation Loss: {val_loss:.4f}")
        if config.use_ema:
            print(f"EMA Validation Loss: {ema_val_loss:.4f}")


def generate_and_log_samples(
    model_components: LatentDiffusionModelComponents, config: DiffusionTrainingConfig, step: int = None,
    val_dataloader: DataLoader = None,
):
    device = torch.device(config.device)
    denoising_model = model_components.denoising_model
    ema_model = model_components.ema_model
    noise_schedule = model_components.noise_schedule

    # Generate random noise
    n_samples = config.num_samples_for_logging
    data_dim = [config.in_channels, config.resolution, config.resolution]
    x = torch.randn(n_samples, *data_dim).to(device)

    # Optionally, prepare the text embeddings
    if val_dataloader:
        # TODO: This assumes n_samples <= batch_size. Get multiple batches until we have enough samples.
        it = iter(val_dataloader)
        batch = next(it)
        if "text_embeddings" in batch or "text_emb" in batch:
            text_embeddings = batch["text_embeddings"] if "text_embeddings" in batch else batch["text_emb"]
            n_samples = min(n_samples, text_embeddings.shape[0])
            text_embeddings = text_embeddings[:n_samples].float().to(device)
            text_embeddings = text_embeddings.reshape(text_embeddings.shape[0], -1)
        else:
            text_embeddings = None

    # Sample using the main model
    sampled_latents = generate_conditional_samples_by_denoising(
        denoising_model, x, text_embeddings, noise_schedule, config.num_denoising_steps, device,
        clip_sample=config.clip_sample_range > 0,
        clip_sample_range=config.clip_sample_range,
        guidance_scale=config.guidance_scale,
    )
    sampled_latents = sampled_latents / config.vae_scale_factor
    sampled_images = model_components.vae.decode(sampled_latents).sample
    print(f"sampled_images: min={sampled_images.min()}, max={sampled_images.max()}, std={sampled_images.std()}")
    # sampled_images = (sampled_images / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
    sampled_images = (sampled_images - sampled_images.min()) / (sampled_images.max() - sampled_images.min())
    images_processed = (
        (sampled_images * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")
    )

    if config.logger == "wandb":
        import wandb

        wandb.log(
            {
                "test_samples_step": step,
                "test_samples": [wandb.Image(img) for img in images_processed],
            }
        )
    else:
        grid = make_grid(sampled_images, nrow=4, normalize=True) # , value_range=(-1, 1))
        save_image(
            grid, Path(config.checkpoint_dir) / f"generated_sample_step_{step}.png"
        )

    if config.use_ema:
        ema_sampled_latents = generate_conditional_samples_by_denoising(
            ema_model, x, text_embeddings, noise_schedule, config.num_denoising_steps, device
        )
        ema_sampled_latents = ema_sampled_latents / config.vae_scale_factor
        ema_sampled_images = model_components.vae.decode(ema_sampled_latents).sample
        ema_images_processed = (
            (ema_sampled_images * 255)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .round()
            .astype("uint8")
        )

        if config.logger == "wandb":
            for i in range(ema_images_processed.shape[0]):
                wandb.log(
                    {
                        "test_samples_step": step,
                        "ema_test_samples": [
                            wandb.Image(img) for img in ema_images_processed
                        ],
                    }
                )
        else:
            grid = make_grid(
                ema_sampled_images, nrow=4, normalize=True, value_range=(-1, 1)
            )
            save_image(
                grid,
                Path(config.checkpoint_dir) / f"ema_generated_sample_step_{step}.png",
            )


def save_model(model: nn.Module, path: Path, logger: str):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")
    if logger == "wandb":
        import wandb

        wandb.save(str(path))


def save_checkpoints(
    model_components: LatentDiffusionModelComponents, step: int, config: DiffusionTrainingConfig
):
    save_model(
        model_components.denoising_model,
        Path(config.checkpoint_dir) / f"model_checkpoint_step_{step}.pth",
        config.logger,
    )
    if config.use_ema:
        save_model(
            model_components.ema_model,
            Path(config.checkpoint_dir) / f"ema_model_checkpoint_step_{step}.pth",
            config.logger,
        )


def compute_and_log_fid(
    model_components: LatentDiffusionModelComponents,
    config: DiffusionTrainingConfig,
    train_dataloader: DataLoader = None,
    val_dataloader: DataLoader = None,
):
    print(f"Generating samples and computing FID. This can be slow.")
    device = torch.device(config.device)
    
    if config.dataset in ["cifar10"]:
        # No need to get real images, as the stats are already computed.
        real_images = None
    

    def batch_generator():
        for batch in val_dataloader:
            yield batch
        for batch in train_dataloader:
            yield batch

    batch_size = config.batch_size * 2  # Adjust this value based on your GPU memory
    num_batches = (config.num_samples_for_fid + batch_size - 1) // batch_size
    generated_images = []

    count = 0
    batch_gen = batch_generator()
    for i in range(num_batches):
        data_batch = next(batch_gen)
        if "text_embeddings" in data_batch:
            text_embeddings = data_batch["text_embeddings"].float().to(device)
            text_embeddings = text_embeddings.reshape(text_embeddings.shape[0], -1)
        else:
            text_embeddings = None
        current_batch_size = min(batch_size, config.num_samples_for_fid - len(generated_images))
        x_t = torch.randn(current_batch_size, config.in_channels, config.resolution, config.resolution).to(device)
        batch_latents = generate_conditional_samples_by_denoising(model_components.denoising_model, x_t, text_embeddings, model_components.noise_schedule, config.num_denoising_steps, device=device, seed=i)
        batch_latents = batch_latents / config.vae_scale_factor
        batch_images = model_components.vae.decode(batch_latents).sample
        generated_images.append(batch_images)
        count += current_batch_size
        print(f"Generated {count} out of {config.num_samples_for_fid} images")

    generated_images = torch.cat(generated_images, dim=0) # [:config.num_samples_for_fid]
    
    real_images = None
    fid_score = compute_fid(real_images, generated_images, device, config.dataset, config.resolution)
    print(f"FID Score: {fid_score:.4f}")

    batch_gen = batch_generator()
    if config.use_ema:
        ema_generated_images = []
        count = 0
        for i in range(num_batches):
            current_batch_size = min(batch_size, config.num_samples_for_fid - len(ema_generated_images))
            data_batch = next(batch_gen)
            if "text_embeddings" in data_batch:
                text_embeddings = data_batch["text_embeddings"].float().to(device)
                text_embeddings = text_embeddings.reshape(text_embeddings.shape[0], -1)
            else:
                text_embeddings = None
            batch_latents = generate_conditional_samples_by_denoising(model_components.ema_model, x_t, text_embeddings, model_components.noise_schedule, config.num_denoising_steps, device=device, seed=i)
            batch_latents = batch_latents / config.vae_scale_factor
            batch_images = model_components.vae.decode(batch_latents).sample
            ema_generated_images.append(batch_images)
            count += current_batch_size
            print(f"EMA Generated {count} out of {config.num_samples_for_fid} images")
        ema_generated_images = torch.cat(ema_generated_images, dim=0) # [:config.num_samples_for_fid]
        ema_fid_score = compute_fid(real_images, ema_generated_images, device, config.dataset, config.resolution)
        print(f"EMA FID Score: {ema_fid_score:.4f}")

    if config.logger == "wandb":
        import wandb

        log_dict = {"fid": fid_score}
        if config.use_ema:
            log_dict["ema_fid"] = ema_fid_score
        wandb.log(log_dict)

