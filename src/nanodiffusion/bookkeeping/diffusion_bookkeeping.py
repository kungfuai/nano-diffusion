from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterator
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from ..config.diffusion_training_config import DiffusionTrainingConfig
from ..diffusion import compute_validation_loss
from ..diffusion.diffusion_model_components import DiffusionModelComponents
from ..eval.fid import compute_fid
from ..bookkeeping.mini_batch import MiniBatch


class DiffusionBookkeeping:
    def __init__(self, config: DiffusionTrainingConfig, model_components: DiffusionModelComponents):
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
        
    def run_callbacks(self, config: DiffusionTrainingConfig, step: int, loss: float, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader, val_dataloader: DataLoader, grad_norm: float = None):
        # TODO: instead of passing in the optimizer, only pass in lr=optimizer.param_groups[0]["lr"].
        self.num_examples_trained += config.batch_size
        model_components = self.model_components

        with torch.no_grad():
            if (config.log_every or 0) > 0 and step % config.log_every == 0:
                log_training_step(
                    step, self.num_examples_trained, loss, optimizer, config.logger, grad_norm
                )

            if (config.validate_every or 0) > 0 and step % config.validate_every == 0 and step > 0 and val_dataloader:
                validate_and_log(
                    model_components,
                    val_dataloader,
                    config,
                )

            if (config.sample_every or 0) > 0 and step % config.sample_every == 0:
                generate_and_log_samples(model_components, config, step, val_dataloader if config.conditional else None, seed=0)

            if (config.save_every or 0) > 0 and step % config.save_every == 0:
                save_checkpoints(model_components, step, config)

            if (config.fid_every or 0) > 0 and step % config.fid_every == 0:
                compute_and_log_fid(model_components, config, train_dataloader, val_dataloader)


def log_training_step(
    step: int,
    num_examples_trained: int,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    logger: str,
    grad_norm: float = None,
):
    lr = optimizer.param_groups[0]["lr"]
    if logger == "wandb":
        import wandb

        log_dict = {
            "num_examples_trained": num_examples_trained,
            "num_batches_trained": step,
            "loss": loss.item(),
            "learning_rate": lr,
        }
        if grad_norm:
            log_dict["grad_norm"] = grad_norm

        wandb.log(log_dict, step=step)
        
    print(
        f"Step: {step}, Examples: {num_examples_trained}, Loss: {loss.item():.4f}, LR: {lr:.6f}"
    )


def validate_and_log(
    model_components: DiffusionModelComponents,
    val_dataloader: DataLoader,
    config: DiffusionTrainingConfig,
):
    val_loss = compute_validation_loss(
        denoising_model=model_components.denoising_model,
        val_dataloader=val_dataloader,
        diffusion=model_components.diffusion,
        config=config,
    )
    if config.use_ema:
        ema_val_loss = compute_validation_loss(
            denoising_model=model_components.ema_model,
            val_dataloader=val_dataloader,
            diffusion=model_components.diffusion,
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
    model_components: DiffusionModelComponents, config: DiffusionTrainingConfig, step: int = None,
    val_dataloader: DataLoader = None, seed: int = None,
):
    device = torch.device(config.device)

    # Generate random noise
    n_samples = config.num_samples_for_logging
    data_dim = config.input_shape
    torch.manual_seed(seed)
    x = torch.randn(n_samples, *data_dim).to(device)
    y = None

    if val_dataloader:
        # TODO: This assumes n_samples <= batch_size. Get multiple batches until we have enough samples.
        it = iter(val_dataloader)
        batch = next(it)
        batch = MiniBatch.from_dataloader_batch(batch)
        inputs, _ = model_components.diffusion.prepare_training_examples(batch)
        y = inputs.get("y")
        if config.conditional:
            assert y is not None, "Conditional model requires y"
        assert len(y) >= n_samples, f"Not enough samples in a validation batch. Need to have at least {n_samples} samples. But the batch has {len(y)} samples."
        y = y[:n_samples]
    else:
        assert not config.conditional, "Conditional model requires val_dataloader. In generate_and_log_samples()."

    # Sample using the main model
    print(f"Sampling a {x.shape} array in {config.num_denoising_steps} steps. Initial avg: {x.mean().item()}")
    if y is not None:
        print(f"y shape: {y.shape}, y avg: {y.mean().item()}")
    if y is not None and config.conditional:
        print(f"Sampling with guidance scale {config.guidance_scale}")
        sampled_x = model_components.diffusion.sample(x, y, guidance_scale=config.guidance_scale, seed=seed)
    else:
        sampled_x = model_components.diffusion.sample(x, seed=seed)
        
    images_processed = (
        (sampled_x * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")
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
        grid = make_grid(sampled_x, nrow=4, normalize=True, value_range=(-1, 1))
        save_image(
            grid, Path(config.checkpoint_dir) / f"generated_sample_step_{step}.png"
        )

    if config.use_ema:
        ema_sampled_images = model_components.diffusion.sample(x)
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
    model_components: DiffusionModelComponents, step: int, config: DiffusionTrainingConfig
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
    model_components: DiffusionModelComponents,
    config: DiffusionTrainingConfig,
    train_dataloader: DataLoader = None,  # TODO: delete this
    val_dataloader: DataLoader = None,
):
    print(f"Generating samples and computing FID. This can be slow.")
    device = torch.device(config.device)
    diffusion = model_components.diffusion

    if config.dataset in ["cifar10"]:
        # No need to get real images, as the stats are already computed.
        real_images = None
    
    batch_size = config.batch_size
    num_batches = (config.num_samples_for_fid + batch_size - 1) // batch_size
    generated_images = []

    def batch_generator() -> Iterator[MiniBatch]:
        while True:
            for batch in val_dataloader:
                batch = MiniBatch.from_dataloader_batch(batch).to(device)
                if batch.num_examples < batch_size:
                    continue
                yield batch
            for batch in train_dataloader:
                batch = MiniBatch.from_dataloader_batch(batch).to(device)
                if batch.num_examples < batch_size:
                    continue
                yield batch
    
    count = 0
    batch_gen = batch_generator()
    for i in range(num_batches):
        data_batch = next(batch_gen)
        inputs, _ = diffusion.prepare_training_examples(data_batch)
        torch.manual_seed(i)
        x_T = torch.randn(batch_size, *config.input_shape).to(device)
        y = inputs.get("y")
        if y is not None and config.conditional:
            batch_sampled_x = diffusion.sample(x_T, y, guidance_scale=config.guidance_scale, seed=i)
        else:
            batch_sampled_x = diffusion.sample(x_T, seed=i)
        
        current_batch_size = min(batch_size, config.num_samples_for_fid - len(generated_images))
        batch_images = batch_sampled_x[:current_batch_size]
        generated_images.append(batch_images)
        count += current_batch_size
        print(f"Generated {count} out of {config.num_samples_for_fid} images")

    generated_images = torch.cat(generated_images, dim=0) # [:config.num_samples_for_fid]
    
    real_images = None
    fid_score = compute_fid(real_images, generated_images, device, config.dataset, config.resolution)
    print(f"FID Score: {fid_score:.4f}")

    if config.use_ema:
        ema_generated_images = []
        count = 0
        batch_gen = batch_generator()
        for i in range(num_batches):
            current_batch_size = min(batch_size, config.num_samples_for_fid - len(ema_generated_images))
            data_batch = next(batch_gen)
            inputs, _ = diffusion.prepare_training_examples(data_batch)
            x_T = torch.randn(batch_size, *config.input_shape).to(device)
            y = inputs.get("y")
            if y is not None and config.conditional:
                batch_sampled_x = diffusion.sample(x_T, y, guidance_scale=config.guidance_scale)
            else:
                batch_sampled_x = diffusion.sample(x_T)
            batch_images = batch_sampled_x[:current_batch_size]
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


def denoise_and_compare(
        model: torch.nn.Module, 
        images: torch.Tensor, 
        forward_diffusion: Callable, 
        denoising_step: Callable, 
        noise_schedule: Dict, 
        n_T: int,  # timesteps for diffusion
        device: str,
    ):
    torch.manual_seed(10)
    model.eval()
    with torch.no_grad():
        # Add noise to the images
        t = torch.randint(0, n_T, (images.shape[0],), device=device)
        x_t, _ = forward_diffusion(images, t, noise_schedule)
        
        # Denoise the images
        pred_noise = model(x_t, t)
        if hasattr(pred_noise, "sample"):
            pred_noise = pred_noise.sample
        pred_previous_images = denoising_step(model, x_t, t, noise_schedule)
        # Compute the predicted original images using the correct formula
        alpha_t = noise_schedule["alphas"][t][:, None, None, None]
        alpha_t_cumprod = noise_schedule["alphas_cumprod"][t][:, None, None, None]
        pred_original_images = (
            x_t - ((1 - alpha_t) / (1 - alpha_t_cumprod).sqrt()) * pred_noise) / (alpha_t / (1 - alpha_t_cumprod).sqrt())
    model.train()
    return x_t, pred_original_images


def log_denoising_results(
    model_components: DiffusionModelComponents,
    config: DiffusionTrainingConfig,
    step: int,
    train_dataloader: DataLoader,
):
    device = torch.device(config.device)
    denoising_model = model_components.denoising_model
    ema_model = model_components.ema_model
    noise_schedule = model_components.noise_schedule

    # Get a batch of real images
    real_images, _ = next(iter(train_dataloader))
    real_images = real_images[:8].to(device)  # Use 8 images for visualization

    # Denoise and compare using real images
    denoised, _ = denoise_and_compare(
        denoising_model, real_images, noise_schedule, config.num_denoising_steps, device
    )

    # Create grids
    real_grid = make_grid(real_images, nrow=4, normalize=True, value_range=(-1, 1))
    denoised_grid = make_grid(denoised, nrow=4, normalize=True, value_range=(-1, 1))

    if config.logger == "wandb":
        wandb.log(
            {
                "real_images": wandb.Image(real_grid),
                "denoised_images": wandb.Image(denoised_grid),
            }
        )
    else:
        save_image(
            real_grid, Path(config.checkpoint_dir) / f"real_images_step_{step}.png"
        )
        save_image(
            denoised_grid,
            Path(config.checkpoint_dir) / f"denoised_images_step_{step}.png",
        )
        # You might want to save the MSE list to a file here

    if config.use_ema:
        ema_denoised, _ = denoise_and_compare(
            ema_model, real_images, noise_schedule, config.num_denoising_steps, device
        )
        ema_denoised_grid = make_grid(
            ema_denoised, nrow=4, normalize=True, value_range=(-1, 1)
        )

        if config.logger == "wandb":
            import wandb

            wandb.log(
                {
                    "num_batches_trained": step,
                    "ema_denoised_images": wandb.Image(ema_denoised_grid),
                }
            )
        else:
            save_image(
                ema_denoised_grid,
                Path(config.checkpoint_dir) / f"ema_denoised_images_step_{step}.png",
            )
