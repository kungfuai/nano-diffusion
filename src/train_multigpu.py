"""
A minimal distributed training pipeline for diffusion.
"""

import argparse
import copy
from datetime import datetime
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import tempfile
from typing import Callable, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from torchvision.utils import save_image, make_grid
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from contextlib import contextmanager
from torch.distributed.elastic.multiprocessing.errors import record
try:
    import wandb
except ImportError:
    print("wandb not installed, skipping")

try:
    from cleanfid import fid
except ImportError:
    print("clean-fid not installed, skipping")


from nanodiffusion.datasets.celeb_dataset import CelebDataset
from nanodiffusion.datasets.flowers_dataset import FlowersDataset
from nanodiffusion.datasets.hugging_face_dataset import HuggingFaceDataset
from nanodiffusion.datasets.pokemon_dataset import PokemonDataset
from nanodiffusion.models.factory import create_model
from nanodiffusion.utils.sample import threshold_sample, denoise_and_compare
from nanodiffusion.optimizers.lr_schedule import get_cosine_schedule_with_warmup
from train_diffusion import precompute_fid_stats_for_real_images, make_dataset_name_safe_for_cleanfid

@contextmanager
def rank0_first():
    """
    This context manager ensures that the code block is executed by the rank 0 process first,
    then all other processes join.
    """
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()

def setup_distributed():
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device for this process
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    return rank, world_size, device


def cleanup_distributed():
    dist.destroy_process_group()


def forward_diffusion(x_0, t, noise_schedule, noise=None):
    _ts = t.view(-1, 1, 1, 1).cpu()
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
    thresholding=False,
    clip_sample=True,
    clip_sample_range=1.0,
):
    """
    This is the backward diffusion step, with the effect of denoising.
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

    if thresholding:
        pred_prev_sample = threshold_sample(pred_prev_sample)

    return pred_prev_sample


def generate_samples_by_denoising(
    denoising_model,
    x_T,
    noise_schedule,
    n_T,
    device,
    thresholding=False,
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
            thresholding,
            clip_sample,
            clip_sample_range,
        )

    x_t = (x_t / 2 + 0.5).clamp(0, 1)
    return x_t


def compute_validation_loss(
    model: Module,
    val_dataloader: DataLoader,
    noise_schedule: Dict[str, torch.Tensor],
    n_T: int,
    device: torch.device,
    use_loss_mean: bool, # to be removed
) -> float:
    total_loss = torch.tensor(0.0, device=device)
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

    # Gather losses from all GPUs
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    
    # Average the loss across all GPUs
    return (total_loss / (num_batches * world_size)).item()


def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str = "cuda:0",
    dataset_name: str = "cifar10",
    resolution: int = 32,
) -> float:
    print(f"Computing FID for {dataset_name} with resolution {resolution}")
    with tempfile.TemporaryDirectory() as temp_dir:
        real_path = Path(temp_dir) / "real"
        gen_path = Path(temp_dir) / "gen"
        real_path.mkdir(exist_ok=True)
        gen_path.mkdir(exist_ok=True)

        print(f"To save {len(generated_images)} generated images to {gen_path}")
        for i, img in enumerate(generated_images):
            img_np = (img.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            np.save(gen_path / f"{i}.npy", img_np)

        # list the folder and print the files
        print(f"Generated images: {list(gen_path.glob('*.npy'))}")

        if dataset_name in ["cifar10"]:
            score = fid.compute_fid(
                str(gen_path),
                dataset_name=dataset_name,
                dataset_res=resolution,
                device=device,
                mode="clean",
                batch_size=2,
            )
        else:
            for i, img in enumerate(real_images):
                img_np = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                np.save(real_path / f"{i}.npy", img_np)
            score = fid.compute_fid(
                str(gen_path),
                str(real_path),
                dataset_res=resolution,
                device=device,
                mode="clean",
                batch_size=2,
            )

    return score


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
    num_samples_to_log: int = 8  # number of samples for logging
    num_samples_for_fid: int = 1000  # number of samples for FID
    num_real_samples_for_fid: int = 10000  # number of real samples for FID

    # Regularization
    max_grad_norm: float = -1  # maximum norm for gradient clipping
    use_loss_mean: bool = False  # use loss.mean() instead of just loss
    use_ema: bool = False  # use EMA for the model
    ema_beta: float = 0.9999  # EMA decay factor
    ema_start_step: int = 0  # step to start EMA update

    # Accelerator
    device: str = "cuda:0"  # device to use for training

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
    rank: int,
    world_size: int,
    device: torch.device,
) -> int:
    
    print(f"Training loop started. Rank: {rank}, World size: {world_size}, Device: {device}.")

    # TODO: The precomputed stats is really a property of the dataset.
    # It should not be the responsibility of the training loop to precompute it.
    # We should upload the precomputed stats to huggingface, as part of the dataset
    # artifact. The clean-fid package may need to be patched to accept the precomputed
    # stats file.
    with rank0_first():
        if config.dataset not in ["cifar10"]:
            if rank == 0:
                precompute_fid_stats_for_real_images(train_dataloader, config, Path(config.checkpoint_dir) / "real_images")

    denoising_model = model_components.denoising_model.to(device)
    # Wrap model with DDP
    denoising_model = DDP(denoising_model, device_ids=[rank])
    
    ema_model = model_components.ema_model
    if ema_model is not None:
        ema_model = ema_model.to(device)
        
    optimizer = model_components.optimizer
    lr_scheduler = model_components.lr_scheduler

    if rank == 0 and config.logger == "wandb":
        project_name = os.getenv("WANDB_PROJECT") or "nano-diffusion"
        print(f"Logging to Weights & Biases project: {project_name}")
        params = sum(p.numel() for p in model_components.denoising_model.parameters())
        run_params = asdict(config)
        run_params["model_parameters"] = params
        wandb.init(project=project_name, config=run_params)

        if config.watch_model:
            print("  Watching model gradients (can be slow)")
            wandb.watch(model_components.denoising_model)

    # Move noise_schedule to device
    noise_schedule = {
        k: v.to(device) for k, v in model_components.noise_schedule.items()
    }

    if rank == 0:
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # make sure the program can write to the checkpoint directory
        # write a dummy file to the checkpoint directory
        (checkpoint_dir / "dummy.txt").touch()
        # remove the dummy file
        (checkpoint_dir / "dummy.txt").unlink()

    step = 0
    num_examples_trained = 0
    criterion = MSELoss()

    shuffle_every = 10000

    while step < config.total_steps:
        if step % shuffle_every == 0:
            epoch = step // shuffle_every
            train_dataloader.sampler.set_epoch(epoch)  # Set epoch for DistributedSampler
        
        for x, _ in train_dataloader:
            if step >= config.total_steps:
                break

            # The total number of examples on all devices.
            num_examples_trained += x.shape[0] * world_size

            # Move batch data to device
            x = x.to(device)

            denoising_model.train()
            loss = train_step(
                denoising_model, x, noise_schedule, optimizer, config, device, criterion
            )
            denoising_model.eval()

            lr_scheduler.step()

            with torch.no_grad():
                if config.use_ema and rank == 0:
                    if step >= config.ema_start_step:
                        update_ema_model(ema_model, denoising_model, config.ema_beta)
                    else:
                        state_dict = {k.replace('module.', ''): v for k, v in denoising_model.state_dict().items()}
                        ema_model.load_state_dict(state_dict)

                if step % config.log_every == 0:
                    log_training_step(
                        step, num_examples_trained, loss, optimizer, config.logger, rank
                    )

                if step % config.validate_every == 0 and step > 0 and val_dataloader:
                    validate_and_log(
                        compute_validation_loss,
                        model_components,
                        val_dataloader,
                        config,
                        rank,
                        device,
                    )

                if step % config.sample_every == 0:
                    generate_and_log_samples(model_components, config, step, rank)

                if step % config.save_every == 0 and step > 0 and rank == 0:
                    save_checkpoints(model_components, step, config)

                if step % config.fid_every == 0 and step > 0:
                    compute_and_log_fid(model_components, config, train_dataloader, rank, world_size, device)

            step += 1

    if step > config.min_steps_for_final_save and rank == 0:
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
    try:
        save_model(
            model_components.denoising_model,  # Save the inner model, not DDP wrapper
            Path(config.checkpoint_dir) / f"model_checkpoint_step_{step}.pth",
            config.logger,
        )
        if config.use_ema:
            save_model(
                model_components.ema_model,
                Path(config.checkpoint_dir) / f"ema_model_checkpoint_step_{step}.pth",
                config.logger,
            )
    except RuntimeError as e:
        rank = dist.get_rank()
        print(f"Error saving checkpoints: {e}. Checkpoint directory: {config.checkpoint_dir}. Step: {step}. Rank: {rank}.")


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
    rank: int = 0,
):
    lr = optimizer.param_groups[0]["lr"]
    if logger == "wandb" and rank == 0:
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
    rank: int,
    device: torch.device,
):
    val_loss = compute_validation_loss(
        model_components.denoising_model,
        val_dataloader,
        model_components.noise_schedule,
        config.num_denoising_steps,
        device,
        config.use_loss_mean,
    )
    
    if config.use_ema:
        ema_val_loss = compute_validation_loss(
            model_components.ema_model,
            val_dataloader,
            model_components.noise_schedule,
            config.num_denoising_steps,
            device,
            config.use_loss_mean,
        )

    # Only log from rank 0
    if rank == 0:
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
    model_components: DiffusionModelComponents, 
    config: TrainingConfig, 
    step: int,
    rank: int = 0,
):
    # Only generate samples on rank 0 to avoid duplicate work
    if rank != 0:
        return
        
    device = torch.device(config.device)
    denoising_model = model_components.denoising_model
    ema_model = model_components.ema_model
    noise_schedule = model_components.noise_schedule

    # Generate random noise
    n_samples = config.num_samples_to_log
    x = torch.randn(n_samples, config.in_channels, config.resolution, config.resolution).to(device)

    # Sample using the main model
    sampled_images = generate_samples_by_denoising(
        denoising_model,
        x, 
        noise_schedule, 
        config.num_denoising_steps, 
        device
    )
    images_processed = (
        (sampled_images * 255)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
        .round()
        .astype("uint8")
    )

    if config.logger == "wandb":
        wandb.log({
            "test_samples_step": step,
            "test_samples": [wandb.Image(img) for img in images_processed],
        })
    else:
        grid = make_grid(sampled_images, nrow=4, normalize=True, value_range=(-1, 1))
        save_image(
            grid, 
            Path(config.checkpoint_dir) / f"generated_sample_step_{step}.png"
        )

    if config.use_ema:
        ema_sampled_images = generate_samples_by_denoising(
            ema_model,  # EMA model is not wrapped in DDP
            x, 
            noise_schedule, 
            config.num_denoising_steps, 
            device
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
            wandb.log({
                "num_batches_trained": step,
                "ema_test_samples": [wandb.Image(img) for img in ema_images_processed],
            })
        else:
            grid = make_grid(
                ema_sampled_images, 
                nrow=4, 
                normalize=True, 
                value_range=(-1, 1)
            )
            save_image(
                grid,
                Path(config.checkpoint_dir) / f"ema_generated_sample_step_{step}.png",
            )


def compute_and_log_fid(
    model_components: DiffusionModelComponents,
    config: TrainingConfig,
    train_dataloader: DataLoader = None,
    rank: int = 0,
    world_size: int = 1,
    device: torch.device = None,
):  
    # TODO: this function becomes too complicated.
    # Refactor this function.
    """
    First, generate samples from the model using multiple GPUs.
    Then, gather the samples from all GPUs. This can be done in a batch-wise manner.
    Then, compute the FID score.
    """
    print(f"In compute_and_log_fid, rank: {rank}, world_size: {world_size}, device: {device}")

    ## Generate samples per GPU and save to npy files.
    samples_per_gpu = config.num_samples_for_fid // world_size
    local_batch_size = config.batch_size // world_size
    num_local_batches = (samples_per_gpu + local_batch_size - 1) // local_batch_size

    with tempfile.TemporaryDirectory() as temp_dir:
        if rank == 0:
            real_path = Path(temp_dir) / "real"
            gen_path = Path(temp_dir) / "gen"
            ema_gen_path = Path(temp_dir) / "ema_gen"
        
            real_path.mkdir(exist_ok=True, parents=True)
            gen_path.mkdir(exist_ok=True, parents=True)
            if config.use_ema:
                ema_gen_path.mkdir(exist_ok=True, parents=True)

        print(f"To generate {num_local_batches} batches of samples")

        count = 0  # this variable is used to save the generated images to npy
        ema_count = 0

        # Each GPU generates a batch of samples.
        # Then we gather all the samples from all GPUs and save them to npy files.
        for i_batch in range(num_local_batches):
            current_batch_size = min(local_batch_size, samples_per_gpu - count)
            x_t = torch.randn(
                current_batch_size, 
                config.in_channels, 
                config.resolution, 
                config.resolution,
                device=device
            )

            # Generate with main model
            import random
            seed = random.randint(0, 2**32 - 1)
            batch_images = generate_samples_by_denoising(
                model_components.denoising_model,
                x_t, 
                model_components.noise_schedule, 
                config.num_denoising_steps, 
                device=device,
                # seed=i_batch + rank * num_local_batches  # Ensure different seeds across GPUs
                seed=seed,
            )
            count += current_batch_size

            # Generate with EMA model if enabled
            if config.use_ema:
                ema_batch_images = generate_samples_by_denoising(
                    model_components.ema_model,
                    x_t,
                    model_components.noise_schedule,
                    config.num_denoising_steps,
                    device=device,
                    seed=i_batch + rank * num_local_batches
                )
                ema_count += current_batch_size
            # Gather main model images from all GPUs
            # TODO: is all_gather necessary? or just gather to rank 0?
            gathered_images_one_batch = [torch.zeros_like(batch_images) for _ in range(world_size)]
            dist.all_gather(gathered_images_one_batch, batch_images)

            # Gather EMA model images from all GPUs if enabled
            if config.use_ema:
                gathered_ema_images_one_batch = [torch.zeros_like(ema_batch_images) for _ in range(world_size)]
                dist.all_gather(gathered_ema_images_one_batch, ema_batch_images)
            
            if rank == 0:
                # Save main model images
                for i_rank, images in enumerate(gathered_images_one_batch):
                    for i_img, img in enumerate(images):
                        idx = i_batch * world_size * local_batch_size + i_rank * local_batch_size + i_img
                        # print(f"To save generated image {idx} to generated_images_{idx}.npy")
                        assert len(img.shape) == 3, f"Expected 3D tensor, got {len(img.shape)}D tensor"
                        img_np = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                        np.save(
                            gen_path / f"generated_images_{idx}.npy", 
                            img_np
                        )

                # Save EMA model images if enabled
                if config.use_ema:
                    for i_rank, images in enumerate(gathered_ema_images_one_batch):
                        for i_img, img in enumerate(images):
                            idx = i_batch * world_size * local_batch_size + i_rank * local_batch_size + i_img
                            print(f"To save EMA generated image {idx} to ema_generated_images_{idx}.npy")
                            assert len(img.shape) == 3, f"Expected 3D tensor, got {len(img.shape)}D tensor"
                            img_np = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                            np.save(
                                ema_gen_path / f"ema_generated_images_{idx}.npy",
                                img_np
                            )

        ## Compute FID score
        if rank == 0:
            if config.dataset in ["cifar10"]:
                # No need to get real images, as the stats are already computed.
                fid_score = fid.compute_fid(
                    str(gen_path),
                    dataset_name=config.dataset,
                    dataset_res=config.resolution,
                    device=device,
                    mode="clean",
                    batch_size=2,
                )
                if config.use_ema:
                    ema_fid_score = fid.compute_fid(
                        str(ema_gen_path),
                        dataset_name=config.dataset,
                        dataset_res=config.resolution,
                        device=device,
                        mode="clean",
                        batch_size=2,
                    )
            else:
                dataset_name = make_dataset_name_safe_for_cleanfid(config.dataset)
                fid_score = fid.compute_fid(
                    str(gen_path),
                    dataset_name=dataset_name,
                    dataset_res=config.resolution,
                    dataset_split="custom",
                    device=device,
                    mode="clean",
                    batch_size=2,
                    num_workers=0,
                )

                if config.use_ema:
                    ema_fid_score = fid.compute_fid(
                        str(ema_gen_path),
                        dataset_name=dataset_name,
                        dataset_res=config.resolution,
                        dataset_split="custom",
                        device=device,
                        mode="clean",
                        batch_size=2,
                        num_workers=0,
                    )
            
            print(f"FID Score: {fid_score:.4f}")
            if config.use_ema:
                print(f"EMA FID Score: {ema_fid_score:.4f}")

            if config.logger == "wandb":
                log_dict = {"fid": fid_score}
                if config.use_ema:
                    log_dict["ema_fid"] = ema_fid_score
                wandb.log(log_dict)



def generate_images(
    model: Module,
    noise_schedule: Dict[str, torch.Tensor],
    n_T: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    with torch.no_grad():
        x = torch.randn(batch_size, 3, 32, 32).to(device)
        samples = generate_samples_by_denoising(model, x, noise_schedule, n_T, device)
    return samples


def load_data(config: TrainingConfig, rank: int, world_size: int) -> Tuple[DataLoader, DataLoader]:
    transforms_list = [
        transforms.Resize((32, 32)),
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
    
    # Create distributed samplers
    #   Don't shuffle for distributed training (handled by the sampler)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True
    )
    return train_dataloader, val_dataloader


def load_model_from_wandb(model, run_path, file_name):
    print(f"Restoring model from {run_path} and {file_name}")
    model_to_resume_from = wandb.restore(file_name, run_path=run_path, replace=True)
    model.load_state_dict(torch.load(model_to_resume_from.name, weights_only=True))
    print(f"Model restored from {model_to_resume_from.name}")


def create_diffusion_model_components(
    config: TrainingConfig,
    device: torch.device,
) -> DiffusionModelComponents:
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
        "--save_every", type=int, default=100000, help="Save model every N steps"
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=3000,
        help="Compute validation loss every N steps",
    )
    parser.add_argument(
        "--fid_every", type=int, default=20000, help="Compute FID every N steps"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
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
        help="Resume from a wandb run path",
    )
    parser.add_argument(
        "--init_from_wandb_file",
        type=str,
        default=None,
        help="Resume from a wandb file",
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


@record
def main():
    args = parse_arguments()
    config = TrainingConfig(**vars(args))
    # Initialize distributed process group
    rank, world_size, device = setup_distributed()
    print(f"Rank: {rank}, World size: {world_size}, Device: {device}")

    with rank0_first():
        train_dataloader, val_dataloader = load_data(config, rank, world_size)
        with device:
            model_components = create_diffusion_model_components(config, device)

    num_examples_trained = training_loop(
        model_components, train_dataloader, val_dataloader, config, rank, world_size,
        device=device
    )

    print(f"Training completed. Total examples trained: {num_examples_trained}")


if __name__ == "__main__":
    main()
