from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from config import TrainingConfig
from diffusion_model_components import DiffusionModelComponents
from mini_batch import MiniBatch
from fid import compute_fid
from diffusion import Diffusion


class Bookkeeping:
    def __init__(self, config: TrainingConfig, model_components: DiffusionModelComponents):
        self.config = config
        self.model_components = model_components
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
        
    def run_callbacks(self, config: TrainingConfig, step: int, loss: float, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader, val_dataloader: DataLoader, grad_norm: float = None):
        self.num_examples_trained += config.batch_size
        model_components = self.model_components

        with torch.no_grad():
            if (config.log_every or 0) > 0 and step % config.log_every == 0:
                log_training_step(
                    step, self.num_examples_trained, loss, optimizer, config.logger,
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
    diffusion: Diffusion,
    config: TrainingConfig,
) -> float:
    total_loss = 0
    num_batches = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in val_dataloader:
            batch = MiniBatch.from_dataloader_batch(batch).to(config.device)
            inputs, targets = diffusion.prepare_training_examples(batch)
            predictions = denoising_model(**inputs)
            predictions = predictions.sample if hasattr(predictions, "sample") else predictions

            loss = criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def validate_and_log(
    model_components: DiffusionModelComponents,
    val_dataloader: DataLoader,
    config: TrainingConfig,
):
    val_loss = compute_validation_loss(
        denoising_model=model_components.denoising_model,
        val_dataloader=val_dataloader,
        diffusion=model_components.diffusion,
        config=config,
    )

    if config.logger == "wandb":
        import wandb
        
        log_dict = {"val_loss": val_loss}
        wandb.log(log_dict)
    else:
        print(f"Validation Loss: {val_loss:.4f}")


def generate_and_log_samples(
    model_components: DiffusionModelComponents, config: TrainingConfig, step: int = None,
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
    print(f"Sampling a {x.shape} array in {config.num_denoising_steps} steps with guidance scale {config.guidance_scale}. Initial avg: {x.mean().item()}")
    if y is not None:
        print(f"y shape: {y.shape}, y avg: {y.mean().item()}")
    if y is not None and config.conditional:
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


def generate_and_log_samples_old(
    denoising_model: nn.Module, noise_schedule: Dict[str, torch.Tensor], config: TrainingConfig, step: int = None
):
    device = torch.device(config.device)

    # Generate random noise
    # TODO: make this a config parameter
    n_samples = 8
    # TODO: data_dim is hardcoded
    data_dim = [3, config.resolution, config.resolution]
    x = torch.randn(n_samples, *data_dim).to(device)

    # Sample using the main model
    sampled_images = generate_samples_by_denoising(
        denoising_model=denoising_model,
        x_T=x,
        y=None,
        noise_schedule=noise_schedule,
        n_T=config.num_denoising_steps,
        device=device,
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
    model_components: DiffusionModelComponents, step: int, config: TrainingConfig
):
    save_model(
        model_components.denoising_model,
        Path(config.checkpoint_dir) / f"model_checkpoint_step_{step}.pth",
        config.logger,
    )


def compute_and_log_fid(
    model_components: DiffusionModelComponents,
    config: TrainingConfig,
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

    if config.logger == "wandb":
        import wandb

        log_dict = {"fid": fid_score}
        wandb.log(log_dict)