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
import copy
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import tempfile
from datetime import datetime
from typing import Callable, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from torchvision.utils import save_image, make_grid
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    print("wandb not installed, skipping")

try:
    from cleanfid import fid
except ImportError:
    print("clean-fid not installed, skipping")


from src.datasets.celeb_dataset import CelebDataset
from src.datasets.flowers_dataset import FlowersDataset
from src.datasets.hugging_face_dataset import HuggingFaceDataset
from src.datasets.pokemon_dataset import PokemonDataset
from src.models.factory import create_model
from src.utils.sample import denoise_and_compare
from src.optimizers.lr_schedule import get_cosine_schedule_with_warmup


def forward_diffusion(x_0, t, noise_schedule, noise=None):
    _ts = t.view(-1, 1, 1, 1)
    if noise is None:
        noise = torch.randn_like(x_0)
    assert _ts.max() < len(
        noise_schedule["alphas_cumprod"]
    ), f"t={_ts.max()} is larger than the length of noise_schedule: {len(noise_schedule['alphas_cumprod'])}"
    alpha_prod_t = noise_schedule["alphas_cumprod"][_ts]
    x_t = (alpha_prod_t**0.5) * x_0 + ((1 - alpha_prod_t) ** 0.5) * noise
    return x_t, noise


def denoising_step(
    denoising_model,
    x_t,
    t,
    noise_schedule,
    clip_sample=True,
    clip_sample_range=1.0,
):
    """
    About clipping, see discussion here: https://github.com/hojonathanho/diffusion/issues/5

    And explanation here: https://github.com/taehoon-yoon/Diffusion/blob/master/src/diffusion.py#L93
    """
    return denoising_step_one_stop(denoising_model, x_t, t, noise_schedule, clip_sample, clip_sample_range)


def denoising_step_direct(
    denoising_model,
    x_t,
    t,
    noise_schedule,
    clip_sample=True,
    clip_sample_range=1.0,
):
    """
    This is the backward diffusion step, with the effect of denoising.

    Given the current state x_t, predict the previous state x_{t-1}.

        x_t --> x_{t-1}

    Similar to a direct flight, there is no additional stop in the middle.
    """
    if isinstance(t, int):
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device)
    else:
        t_tensor = t

    with torch.no_grad():
        eps_theta = denoising_model(t=t_tensor, x=x_t)
    if hasattr(eps_theta, "sample"):
        eps_theta = eps_theta.sample

    # Extract alphas from noise schedule
    alpha_t = noise_schedule["alphas"][t_tensor]
    alpha_t_cumprod = noise_schedule["alphas_cumprod"][t_tensor]

    # Reshape for broadcasting
    view_shape = (-1,) + (1,) * (x_t.ndim - 1)
    alpha_t = alpha_t.view(*view_shape)
    alpha_t_cumprod = alpha_t_cumprod.view(*view_shape)

    # Calculate epsilon factor
    eps_factor = (1 - alpha_t) / (1 - alpha_t_cumprod).sqrt()

    # Calculate mean for reverse process
    mean = (1 / torch.sqrt(alpha_t)) * (x_t - eps_factor * eps_theta)

    # Apply clipping to mean if requested
    if clip_sample:
        mean = torch.clamp(mean, -clip_sample_range, clip_sample_range)

    # Add noise scaled by variance for non-zero timesteps
    noise = torch.randn_like(x_t)
    beta_t = 1 - alpha_t
    variance = beta_t * (1 - alpha_t_cumprod / alpha_t) / (1 - alpha_t_cumprod)
    variance = torch.clamp(variance, min=1e-20)  # Add clamp to prevent numerical instability
    
    # Mask out noise for t=0 timesteps
    non_zero_mask = (t_tensor > 0).float().view(*view_shape)
    noise_scale = torch.sqrt(variance) * non_zero_mask
    
    pred_prev_sample = mean + noise_scale * noise

    return pred_prev_sample


def denoising_step_one_stop(
    denoising_model,
    x_t,
    t,
    noise_schedule,
    clip_sample=True,
    clip_sample_range=1.0,
):
    """
    This is the backward diffusion step, with the effect of denoising.

    The procedure is similar to a flight with one stop in the middle.

        1. x_t --> predicted x_0
        2. clip the predicted x_0 to desired range
        3. predicted x_0 --> predicted x_{t-1}

    To clip the x_0 to out desired range, we cannot directly apply (11) to sample x_{t-1}, rather we have to
    calculate predicted x_0 using (4) and then calculate mu in (7) using that predicted x_0. Which is exactly
    same calculation except for clipping. (See https://github.com/taehoon-yoon/Diffusion/blob/master/src/diffusion.py#L93)
    """
    if isinstance(t, int):
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device)
    else:
        t_tensor = t
    with torch.no_grad():
        model_output = denoising_model(t=t_tensor, x=x_t)
    if hasattr(model_output, "sample"):
        model_output = model_output.sample

    # Extract relevant values from noise_schedule
    alpha_prod_t = noise_schedule["alphas_cumprod"][t_tensor]
    # deal with t=0 case where t can be a tensor
    alpha_prod_t_prev = torch.where(
        t_tensor > 0,
        noise_schedule["alphas_cumprod"][t_tensor - 1],
        torch.ones_like(t_tensor, device=x_t.device),
    )

    # Reshape alpha_prod_t_prev for proper broadcasting
    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
    alpha_prod_t_prev = alpha_prod_t_prev.view(-1, 1, 1, 1)

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # Compute the previous sample mean
    pred_original_sample = (x_t - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

    if clip_sample:
        pred_original_sample = torch.clamp(
            pred_original_sample, -clip_sample_range, clip_sample_range
        )

    # Compute the coefficients for pred_original_sample and current sample
    pred_original_sample_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

    # Compute the previous sample
    pred_prev_sample = (
        pred_original_sample_coeff * pred_original_sample + current_sample_coeff * x_t
    )

    # Add noise
    variance = torch.zeros_like(x_t)
    variance_noise = torch.randn_like(x_t)

    # Handle t=0 case where t can be a tensor
    non_zero_mask = (t_tensor != 0).float().view(-1, 1, 1, 1)
    variance = non_zero_mask * (
        (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
    )
    variance = torch.clamp(variance, min=1e-20)

    pred_prev_sample = pred_prev_sample + (variance**0.5) * variance_noise

    return pred_prev_sample


def generate_samples_by_denoising(
    denoising_model,
    x_T,
    noise_schedule,
    n_T,
    device,
    clip_sample=True,
    clip_sample_range=1.0,
    seed=0,
):
    """
    This is the generation process.
    """
    torch.manual_seed(seed)

    x_t = x_T.to(device)
    for t in range(n_T - 1, -1, -1):
        x_t = denoising_step(
            denoising_model,
            x_t,
            t,
            noise_schedule,
            clip_sample,
            clip_sample_range,
        )

    # print("raw x_t range", x_t.min(), x_t.max())
    x_t = (x_t / 2 + 0.5).clamp(0, 1)
    # print("after clamp", x_t.min(), x_t.max())
    return x_t


def compute_validation_loss(
    model: Module,
    val_dataloader: DataLoader,
    noise_schedule: Dict[str, torch.Tensor],
    n_T: int,
    device: torch.device,
    use_loss_mean: bool,
) -> float:
    total_loss = 0
    num_batches = 0
    criterion = MSELoss()

    with torch.no_grad():
        for x, _ in val_dataloader:
            x = x.to(device)
            t = torch.randint(0, n_T, (x.shape[0],)).to(device)
            noise = torch.randn(x.shape).to(device)
            x_t, true_noise = forward_diffusion(x, t, noise_schedule, noise=noise)

            predicted_noise = model(t=t, x=x_t)
            if hasattr(predicted_noise, "sample"):
                predicted_noise = predicted_noise.sample

            loss = criterion(predicted_noise, true_noise)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str = "cuda",
    dataset_name: str = "cifar10",
    resolution: int = 32,
    batch_size: int = 2,
) -> float:
    print(f"Computing FID for {dataset_name} with resolution {resolution}")
    with tempfile.TemporaryDirectory() as temp_dir:
        real_path = Path(temp_dir) / "real"
        gen_path = Path(temp_dir) / "gen"
        real_path.mkdir(exist_ok=True)
        gen_path.mkdir(exist_ok=True)

        for i, img in enumerate(generated_images):
            assert len(img.shape) == 3, f"Image must have 3 dimensions, got {len(img.shape)}"
            img_np = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            np.save(gen_path / f"{i}.npy", img_np)

        if dataset_name in ["cifar10"]:
            score = fid.compute_fid(
                str(gen_path),
                dataset_name=dataset_name,
                dataset_res=resolution,
                device=device,
                mode="clean",
                batch_size=batch_size,
                num_workers=0,
            )
        else:
            # Use precomputed stats.
            dataset_name_safe = make_dataset_name_safe_for_cleanfid(dataset_name)
            score = fid.compute_fid(
                str(gen_path),
                dataset_name=dataset_name_safe,
                dataset_res=resolution,
                device=device,
                mode="clean",
                dataset_split="custom",
                batch_size=batch_size,
                num_workers=0,
            )

    return score


def make_dataset_name_safe_for_cleanfid(dataset_name: str):
    return dataset_name.replace("/", "__")


def precompute_fid_stats_for_real_images(dataloader: DataLoader, config: "TrainingConfig", real_images_dir: Path):
    print(f"Precomputing FID stats for {config.num_real_samples_for_fid} real images from {config.dataset}")
    count = 0
    real_images_dir.mkdir(exist_ok=True, parents=True)
    for images, _ in dataloader:
        # save individual images as npy files
        for i, img in enumerate(images):
            np_img = img.cpu().numpy().transpose(1, 2, 0)
            # do we need to scale to 0-255?
            np_img = (np_img * 255)
            idx = count * len(images) + i
            np.save(real_images_dir / f"real_images_{idx:06d}.npy", np_img)
        count += len(images)
        if count >= config.num_real_samples_for_fid:
            break
    
    dataset_name_safe = make_dataset_name_safe_for_cleanfid(config.dataset)
    fid.make_custom_stats(dataset_name_safe, str(real_images_dir), mode="clean")

@dataclass
class TrainingConfig:
    # Dataset
    dataset: str  # dataset name
    resolution: int  # resolution of the image

    # Model architecture
    in_channels: int  # number of input channels
    resolution: int  # resolution of the image
    net: str  # network architecture
    num_denoising_steps: int  # number of timesteps

    # Training loop and optimizer
    total_steps: int  # total number of training steps
    batch_size: int  # batch size
    learning_rate: float  # initial learning rate
    weight_decay: float  # weight decay
    lr_min: float  # minimum learning rate
    warmup_steps: int  # number of warmup steps

    # Logging and evaluation
    log_every: int  # log every N steps
    sample_every: int  # sample every N steps
    save_every: int  # save model every N steps
    validate_every: int  # compute validation loss every N steps
    fid_every: int  # compute FID every N steps
    num_samples_for_fid: int = 1000  # number of samples for FID
    num_real_samples_for_fid: int = 100  # number of real samples for FID when not using CIFAR

    # Sampling
    clip_sample_range: float = 1.0  # range for clipping sample. If 0 or less, no clipping

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
    
    def __post_init__(self):
        self.update_checkpoint_dir()


@dataclass
class DiffusionModelComponents:
    denoising_model: Module
    ema_model: Optional[Module]
    optimizer: Optimizer
    lr_scheduler: Any
    noise_schedule: Dict[str, torch.Tensor]


def training_loop(
    model_components: DiffusionModelComponents,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    config: TrainingConfig,
) -> int:
    print(f"Training on {config.device}")

    device = torch.device(config.device)
    denoising_model = model_components.denoising_model.to(device)
    ema_model = model_components.ema_model
    optimizer = model_components.optimizer
    lr_scheduler = model_components.lr_scheduler

    if config.dataset not in ["cifar10"]:
        precompute_fid_stats_for_real_images(train_dataloader, config, Path(config.checkpoint_dir) / "real_images")

    if config.logger == "wandb":
        project_name = os.getenv("WANDB_PROJECT") or "nano-diffusion"
        print(f"Logging to Weights & Biases project: {project_name}")
        params = sum(p.numel() for p in model_components.denoising_model.parameters())
        run_params = asdict(config)
        run_params["model_parameters"] = params
        wandb.init(project=project_name, config=run_params)

    if config.logger == "wandb" and config.watch_model:
        print("  Watching model gradients (can be slow)")
        wandb.watch(model_components.denoising_model)

    # Move noise_schedule to device
    noise_schedule = {
        k: v.to(device) for k, v in model_components.noise_schedule.items()
    }

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    num_examples_trained = 0
    criterion = MSELoss()

    while step < config.total_steps:
        for x, _ in train_dataloader:
            if step >= config.total_steps:
                break

            num_examples_trained += x.shape[0]

            # Move batch data to device
            x = x.to(device)

            denoising_model.train()
            loss = train_step(
                denoising_model, x, noise_schedule, optimizer, config, device, criterion
            )
            denoising_model.eval()

            lr_scheduler.step()

            with torch.no_grad():
                if config.use_ema:
                    if step >= config.ema_start_step:
                        update_ema_model(ema_model, denoising_model, config.ema_beta)
                    else:
                        ema_model.load_state_dict(denoising_model.state_dict())

                if step % config.log_every == 0:
                    log_training_step(
                        step, num_examples_trained, loss, optimizer, config.logger
                    )

                if step % config.validate_every == 0 and step > 0 and val_dataloader:
                    validate_and_log(
                        compute_validation_loss,
                        model_components,
                        val_dataloader,
                        config,
                    )

                if step % config.sample_every == 0:
                    generate_and_log_samples(model_components, config, step)
                    # log_denoising_results(model_components, config, step, train_dataloader)

                if step % config.save_every == 0 and step > 0:
                    save_checkpoints(model_components, step, config)

                if step % config.fid_every == 0 and step > 0:
                    compute_and_log_fid(model_components, config, train_dataloader)

            step += 1

    if step > config.min_steps_for_final_save:
        save_final_models(model_components, config)

    return num_examples_trained


def train_step(
    denoising_model: Module,
    x_0: torch.Tensor,
    noise_schedule: Dict[str, torch.Tensor],
    optimizer: Optimizer,
    config: TrainingConfig,
    device: torch.device,
    criterion: Module,
) -> torch.Tensor:
    optimizer.zero_grad()
    x_0 = x_0.to(device)

    noise = torch.randn(x_0.shape).to(device)
    t = torch.randint(
        0, config.num_denoising_steps, (x_0.shape[0],), device=device
    ).long()
    x_t, true_noise = forward_diffusion(x_0, t, noise_schedule, noise=noise)

    predicted_noise = denoising_model(t=t, x=x_t)
    predicted_noise = (
        predicted_noise.sample
        if hasattr(predicted_noise, "sample")
        else predicted_noise
    )

    loss = criterion(predicted_noise, true_noise)
    loss = loss.mean() if config.use_loss_mean else loss

    loss.backward()

    if config.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            denoising_model.parameters(), config.max_grad_norm
        )

    optimizer.step()

    return loss


def update_ema_model(ema_model: Module, model: Module, ema_beta: float):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(ema_beta).add_(param.data, alpha=1 - ema_beta)


def save_checkpoints(
    model_components: DiffusionModelComponents, step: int, config: TrainingConfig
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


def save_final_models(
    model_components: DiffusionModelComponents, config: TrainingConfig
):
    save_model(
        model_components.denoising_model,
        Path(config.checkpoint_dir) / "final_model.pth",
        config.logger,
    )
    if config.use_ema:
        save_model(
            model_components.ema_model,
            Path(config.checkpoint_dir) / "final_ema_model.pth",
            config.logger,
        )


def save_model(model: Module, path: Path, logger: str):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")
    if logger == "wandb":
        wandb.save(str(path))


def log_training_step(
    step: int,
    num_examples_trained: int,
    loss: torch.Tensor,
    optimizer: Optimizer,
    logger: str,
):
    lr = optimizer.param_groups[0]["lr"]
    if logger == "wandb":
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
    compute_validation_loss: Callable,
    model_components: DiffusionModelComponents,
    val_dataloader: DataLoader,
    config: TrainingConfig,
):
    val_loss = compute_validation_loss(
        model_components.denoising_model,
        val_dataloader,
        model_components.noise_schedule,
        config.num_denoising_steps,
        config.device,
        config.use_loss_mean,
    )
    if config.use_ema:
        ema_val_loss = compute_validation_loss(
            model_components.ema_model,
            val_dataloader,
            model_components.noise_schedule,
            config.num_denoising_steps,
            config.device,
            config.use_loss_mean,
        )

    if config.logger == "wandb":
        log_dict = {"val_loss": val_loss}
        if config.use_ema:
            log_dict["ema_val_loss"] = ema_val_loss
        wandb.log(log_dict)
    else:
        print(f"Validation Loss: {val_loss:.4f}")
        if config.use_ema:
            print(f"EMA Validation Loss: {ema_val_loss:.4f}")


def generate_and_log_samples(
    model_components: DiffusionModelComponents, config: TrainingConfig, step: int = None
):
    device = torch.device(config.device)
    denoising_model = model_components.denoising_model
    ema_model = model_components.ema_model
    noise_schedule = model_components.noise_schedule

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

    if config.use_ema:
        ema_sampled_images = generate_samples_by_denoising(
            ema_model, x, noise_schedule, config.num_denoising_steps, device
        )
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
                        "num_batches_trained": step,
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


def compute_and_log_fid(
    model_components: DiffusionModelComponents,
    config: TrainingConfig,
    train_dataloader: DataLoader = None,
):
    print(f"Generating samples and computing FID. This can be slow.")
    device = torch.device(config.device)
    
    if config.dataset in ["cifar10"]:
        # No need to get real images, as the stats are already computed.
        real_images = None
    # else:
    #     # Get real images in batches to avoid OOM
    #     real_images = []
    #     batch_size = min(1000, config.num_real_samples_for_fid)  # Process in chunks of 1000 or less
    #     remaining = config.num_real_samples_for_fid
    #     while remaining > 0:
    #         curr_batch_size = min(batch_size, remaining)
    #         batch = get_real_images(curr_batch_size, train_dataloader)
    #         real_images.append(batch.cpu())  # Move to CPU immediately
    #         remaining -= curr_batch_size
    #     real_images = torch.cat(real_images, dim=0)
    
    batch_size = config.batch_size * 2  # Adjust this value based on your GPU memory
    num_batches = (config.num_samples_for_fid + batch_size - 1) // batch_size
    generated_images = []

    count = 0
    for i in range(num_batches):
        current_batch_size = min(batch_size, config.num_samples_for_fid - len(generated_images))
        x_t = torch.randn(current_batch_size, config.in_channels, config.resolution, config.resolution).to(device)
        batch_images = generate_samples_by_denoising(model_components.denoising_model, x_t, model_components.noise_schedule, config.num_denoising_steps, device=device, seed=i)
        generated_images.append(batch_images)
        count += current_batch_size
        print(f"Generated {count} out of {config.num_samples_for_fid} images")

    generated_images = torch.cat(generated_images, dim=0) # [:config.num_samples_for_fid]
    
    real_images = None
    fid_score = compute_fid(real_images, generated_images, device, config.dataset, config.resolution)
    print(f"FID Score: {fid_score:.4f}")

    if config.use_ema:
        ema_generated_images = []
        for i in range(num_batches):
            current_batch_size = min(batch_size, config.num_samples_for_fid - len(ema_generated_images))
            batch_images = generate_samples_by_denoising(model_components.ema_model, x_t, model_components.noise_schedule, config.num_denoising_steps, device=device, seed=i)
            ema_generated_images.append(batch_images)
        ema_generated_images = torch.cat(ema_generated_images, dim=0)[:config.num_samples_for_fid]
        ema_fid_score = compute_fid(real_images, ema_generated_images, device, config.dataset, config.resolution)
        print(f"EMA FID Score: {ema_fid_score:.4f}")

    if config.logger == "wandb":
        log_dict = {"fid": fid_score}
        if config.use_ema:
            log_dict["ema_fid"] = ema_fid_score
        wandb.log(log_dict)


def get_real_images(batch_size: int, dataloader: DataLoader) -> torch.Tensor:
    batch = next(
        iter(DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=False))
    )
    assert len(batch) == 2, f"Batch must contain 2 elements. Got {len(batch)}"
    assert isinstance(batch[0], torch.Tensor), f"First element of batch must be a tensor. Got {type(batch[0])}"
    assert len(batch[0].shape) == 4, f"First element of batch must be a 4D tensor. Got shape {batch[0].shape}"
    return batch[0]

def generate_images(
    model: Module,
    noise_schedule: Dict[str, torch.Tensor],
    n_T: int,
    device: torch.device,
    batch_size: int,
    resolution: int = 32,
    in_channels: int = 3,
) -> torch.Tensor:
    with torch.no_grad():
        x = torch.randn(batch_size, in_channels, resolution, resolution).to(device)
        samples = generate_samples_by_denoising(model, x, noise_schedule, n_T, device)
    return samples


def load_data(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    resolution = config.resolution
    transforms_list = [
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    if config.random_flip:
        transforms_list.insert(0, transforms.RandomHorizontalFlip())

    transform = transforms.Compose(transforms_list)

    if config.dataset == "cifar10":
        print("Loading CIFAR10 dataset")
        full_dataset = CIFAR10(
            root="./data/cifar10", train=True, download=True, transform=transform
        )  # list of tuples (image, label)
    elif config.dataset == "flowers":
        print("Loading Flowers dataset")
        full_dataset = FlowersDataset(transform=transform)
    elif config.dataset == "celeb":
        print("Loading CelebA dataset")
        full_dataset = CelebDataset(transform=transform)
    elif config.dataset == "pokemon":
        print("Loading Pokemon dataset")
        full_dataset = PokemonDataset(transform=transform)
    else:
        print(f"Loading dataset from Hugging Face: {config.dataset}")
        full_dataset = HuggingFaceDataset(config.dataset, transform=transform)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )
    return train_dataloader, val_dataloader


def load_model_from_wandb(model, run_path, file_name):
    print(f"Restoring model from {run_path} and {file_name}")
    model_to_resume_from = wandb.restore(file_name, run_path=run_path, replace=True)
    model.load_state_dict(torch.load(model_to_resume_from.name, weights_only=True))
    print(f"Model restored from {model_to_resume_from.name}")


def create_diffusion_model_components(
    config: TrainingConfig,
) -> DiffusionModelComponents:
    device = torch.device(config.device)
    denoising_model = create_model(
        net=config.net, in_channels=config.in_channels, resolution=config.resolution
    )
    denoising_model = denoising_model.to(device)
    # ema_model = create_ema_model(denoising_model, config.ema_beta) if config.use_ema else None
    ema_model = copy.deepcopy(denoising_model) if config.use_ema else None
    optimizer = optim.AdamW(
        denoising_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps,
        lr_min=config.lr_min,
    )
    noise_schedule = create_noise_schedule(config.num_denoising_steps, device)

    if config.init_from_wandb_run_path and config.init_from_wandb_file:
        load_model_from_wandb(
            denoising_model,
            config.init_from_wandb_run_path,
            config.init_from_wandb_file,
        )

    return DiffusionModelComponents(
        denoising_model, ema_model, optimizer, lr_scheduler, noise_schedule
    )


def create_noise_schedule(n_T: int, device: torch.device) -> Dict[str, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, n_T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1).to(device), alphas_cumprod[:-1].to(device)]
    )
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "alphas": alphas,
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="DDPM training for images")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="cifar10",
        help="Dataset to use: e.g. cifar10, flowers, celeb, pokemon, or any huggingface dataset that has an image field.",
    )
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--resolution", type=int, default=32, help="Resolution of the image. Only used for unet.")
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
        choices=[
            "dit_t0",
            "dit_t1",
            "dit_t2",
            "dit_t3",
            "dit_s2",
            "dit_b2",
            "dit_b4",
            "dit_b2",
            "dit_b4",
            "dit_l2",
            "dit_l4",
            "unet_small",
            "unet",
            "unet_big",
        ],
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
        "--warmup_steps", type=int, default=1200, help="Number of warmup steps"
    )
    parser.add_argument(
        "--total_steps", type=int, default=300000, help="Total number of training steps"
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
    args = parser.parse_args()
    return args


def log_denoising_results(
    model_components: DiffusionModelComponents,
    config: TrainingConfig,
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
            # You might want to save the EMA MSE list to a file here


def main():
    args = parse_arguments()
    config = TrainingConfig(**vars(args))

    train_dataloader, val_dataloader = load_data(config)
    model_components = create_diffusion_model_components(config)

    num_examples_trained = training_loop(
        model_components, train_dataloader, val_dataloader, config
    )

    print(f"Training completed. Total examples trained: {num_examples_trained}")


if __name__ == "__main__":
    main()
