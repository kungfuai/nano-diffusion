from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from .config import TrainingConfig
from .diffusion import forward_diffusion, generate_samples_by_denoising


class Bookkeeping:
    def __init__(self, config: TrainingConfig, denoising_model: nn.Module, noise_schedule: Dict):
        self.config = config
        self.denoising_model = denoising_model
        self.noise_schedule = noise_schedule
        self.num_examples_trained = 0

    def set_up_logger(self):
        # create directory for checkpoints
        print(f"Creating checkpoint directory: {self.config.checkpoint_dir}")
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Setting up logger: {self.config.logger}")
        if self.config.logger == "wandb":
            import wandb

            project_name = os.getenv("WANDB_PROJECT") or "nano-diffusion"
            print(f"Logging to Weights & Biases project: {project_name}")
            params = sum(p.numel() for p in self.denoising_model.parameters())
            run_params = asdict(self.config)
            run_params["model_parameters"] = params
            wandb.init(project=project_name, config=run_params)
        
    def run_callbacks(self, config: TrainingConfig, step: int, loss: float, optimizer: torch.optim.Optimizer, val_dataloader: DataLoader):
        self.num_examples_trained += config.batch_size
        denoising_model = self.denoising_model
        noise_schedule = self.noise_schedule

        with torch.no_grad():
            if step % config.log_every == 0:
                log_training_step(
                    step, self.num_examples_trained, loss, optimizer, config.logger
                )

            if step % config.validate_every == 0 and step > 0 and val_dataloader:
                validate_and_log(
                    compute_validation_loss=compute_validation_loss,
                    denoising_model=self.denoising_model,
                    noise_schedule=self.noise_schedule,
                    val_dataloader=val_dataloader,
                    config=config,
                )

            if step % config.sample_every == 0:
                generate_and_log_samples(denoising_model, noise_schedule, config, step)

            if step % config.save_every == 0 and step > 0:
                save_checkpoints(denoising_model, step, config)


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



def compute_validation_loss(
    denoising_model: nn.Module,
    val_dataloader: DataLoader,
    noise_schedule: Dict[str, torch.Tensor],
    n_T: int,
    device: torch.device,
) -> float:
    total_loss = 0
    num_batches = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for x, _ in val_dataloader:
            x = x.to(device)
            t = torch.randint(0, n_T, (x.shape[0],)).to(device)
            noise = torch.randn(x.shape).to(device)
            x_t, true_noise = forward_diffusion(x, t, noise_schedule, noise=noise)

            predicted_noise = denoising_model(t=t, x=x_t)
            if hasattr(predicted_noise, "sample"):
                predicted_noise = predicted_noise.sample

            loss = criterion(predicted_noise, true_noise)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def validate_and_log(
    compute_validation_loss: Callable,
    denoising_model: nn.Module,
    noise_schedule: dict,
    val_dataloader: DataLoader,
    config: TrainingConfig,
):
    val_loss = compute_validation_loss(
        denoising_model,
        val_dataloader,
        noise_schedule,
        config.num_denoising_steps,
        config.device,
    )

    if config.logger == "wandb":
        import wandb

        log_dict = {"val_loss": val_loss}
        wandb.log(log_dict)
    else:
        print(f"Validation Loss: {val_loss:.4f}")



def generate_and_log_samples(
    denoising_model: nn.Module, noise_schedule: Dict[str, torch.Tensor], config: TrainingConfig, step: int = None
):
    device = torch.device(config.device)

    # Generate random noise
    # TODO: make this a config parameter
    n_samples = 8
    # TODO: data_dim = [3, 32, 32] is hardcoded
    x = torch.randn(n_samples, 3, 32, 32).to(device)

    # Sample using the main model
    sampled_images = generate_samples_by_denoising(
        denoising_model, x, noise_schedule, config.num_denoising_steps, device,
        clip_sample=config.clip_sample_range > 0,
        clip_sample_range=config.clip_sample_range,
    )
    images_processed = (
        (sampled_images * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")
    )

    if config.logger == "wandb":
        import wandb
        wandb.log(
            {
                "num_batches_trained": step,
                "test_samples": [wandb.Image(img) for img in images_processed],
            }
        )
    else:
        grid = make_grid(sampled_images, nrow=4, normalize=True, value_range=(-1, 1))
        save_image(
            grid, Path(config.checkpoint_dir) / f"generated_sample_step_{step}.png"
        )



def save_model(model: nn.Module, path: Path, logger: str):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")
    if logger == "wandb":
        import wandb

        wandb.save(str(path))


def save_checkpoints(
    denoising_model: nn.Module, step: int, config: TrainingConfig
):
    save_model(
        denoising_model,
        Path(config.checkpoint_dir) / f"model_checkpoint_step_{step}.pth",
        config.logger,
    )
