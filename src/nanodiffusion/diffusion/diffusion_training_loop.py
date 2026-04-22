from pathlib import Path
from typing import Optional, Dict
from contextlib import nullcontext

import torch
from torch.nn import MSELoss, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..bookkeeping.diffusion_bookkeeping import DiffusionBookkeeping
from ..diffusion.diffusion_model_components import DiffusionModelComponents
from ..config.diffusion_training_config import DiffusionTrainingConfig as TrainingConfig
from ..eval.fid import precompute_fid_stats_for_real_images
from ..bookkeeping import MiniBatch
from ..diffusion.base import BaseDiffusionAlgorithm as Diffusion


def training_loop(
    model_components: DiffusionModelComponents,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    config: TrainingConfig,
) -> int:
    print(f"Training on {config.device}")

    denoising_model = model_components.denoising_model  # .to(device)
    ema_model = model_components.ema_model
    optimizer = model_components.optimizer
    lr_scheduler = model_components.lr_scheduler

    bookkeeping = DiffusionBookkeeping(config, model_components)

    if config.dataset not in ["cifar10"] and config.fid_every > 0:
        # TODO: this only applies to images
        precompute_fid_stats_for_real_images(train_dataloader, config, Path(config.cache_dir) / "real_images_for_fid")

    bookkeeping.set_up_logger()

    accelerator = None
    if config.accelerator:
        from accelerate import Accelerator

        accelerator = Accelerator(
            mixed_precision="fp16" if config.fp16 else "no",
        )
        print(f"Using accelerator with mixed precision {accelerator.mixed_precision}.")
        denoising_model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(denoising_model, train_dataloader, optimizer, lr_scheduler)

    step = 0
    num_examples_trained = 0
    criterion = MSELoss()

    if config.mask_ratio > 0:
        print(f"Patch masking enabled: mask_ratio={config.mask_ratio}, patch_mixer_depth={config.patch_mixer_depth}")
        if config.progressive_unmasking:
            print(f"Progressive unmasking: will decrease mask_ratio from {config.mask_ratio} to 0 starting at step {int(config.total_steps * config.unmask_start_ratio)}")

    while step < config.total_steps:
        for batch in train_dataloader:
            if step >= config.total_steps:
                break

            batch = MiniBatch.from_dataloader_batch(batch)
            num_examples_trained += batch.num_examples

            denoising_model.train()
            loss = train_step(
                batch, denoising_model, model_components.diffusion, optimizer, config, criterion, accelerator, step=step,
            )
            if config.log_grad_norm:
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in denoising_model.parameters() if p.grad is not None]), 2)
            denoising_model.eval()

            lr_scheduler.step()

            with torch.no_grad():
                if config.use_ema:
                    if step >= config.ema_start_step:
                        update_ema_model(ema_model, denoising_model, config.ema_beta)
                    else:
                        ema_model.load_state_dict(denoising_model.state_dict())

                bookkeeping.run_callbacks(
                    config=config,
                    step=step,
                    loss=loss,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    grad_norm=grad_norm if config.log_grad_norm else None,
                )
            step += 1

    if step > config.min_steps_for_final_save:
        save_final_models(model_components, config)
    
    if accelerator:
        accelerator.end_training()

    return num_examples_trained


def get_effective_mask_ratio(config: TrainingConfig, step: int) -> float:
    """Compute the effective mask ratio, accounting for progressive unmasking."""
    if config.mask_ratio <= 0:
        return 0.0
    if not config.progressive_unmasking:
        return config.mask_ratio
    # Linear ramp-down from mask_ratio to 0 starting at unmask_start_ratio * total_steps
    unmask_start = int(config.total_steps * config.unmask_start_ratio)
    if step < unmask_start:
        return config.mask_ratio
    progress = (step - unmask_start) / max(1, config.total_steps - unmask_start)
    return config.mask_ratio * (1.0 - progress)


def train_step(
    batch: MiniBatch,  # data
    denoising_model: Module,  # model
    diffusion: Diffusion,  # diffusion algorithm for teaching (data augmentation)
    optimizer: Optimizer,  # optimizer
    config: TrainingConfig,  # config
    criterion: Module,  # loss function
    accelerator = None,  # accelerator utility
    step: int = 0,  # current training step (for progressive unmasking)
) -> torch.Tensor:
    context = accelerator.accumulate() if accelerator else nullcontext()

    with context:
        optimizer.zero_grad()
        inputs, targets = diffusion.prepare_step_supervision(batch)
        assert str(inputs["x"].device) == str(config.device), f"Inputs are on {inputs['x'].device}, but config is on {config.device}"

        # Add mask_ratio to inputs if masking is enabled
        mask_ratio = get_effective_mask_ratio(config, step)
        if mask_ratio > 0:
            inputs["mask_ratio"] = mask_ratio

        try:
            predictions = denoising_model(**inputs)
        except Exception as e:
            for k, v in inputs.items():
                if hasattr(v, "shape"):
                    print("------ ", k, v.shape)
            raise e

        # Handle masked output: model returns (predictions, mask) when masking
        mask = None
        if isinstance(predictions, tuple):
            predictions, mask = predictions
        predictions = predictions.sample if hasattr(predictions, "sample") else predictions

        # Compute loss: masked loss if masking is active, otherwise standard MSE
        if mask is not None and hasattr(denoising_model, "patch_size"):
            from ..models.patch_masking import compute_masked_loss
            loss = compute_masked_loss(predictions, targets, mask, denoising_model.patch_size)
        else:
            loss = criterion(predictions, targets)

        if accelerator:
            accelerator.backward(loss)
        else:
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
        import wandb

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


def get_real_images(batch_size: int, dataloader: DataLoader) -> torch.Tensor:
    batch = next(
        iter(DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=False))
    )
    assert len(batch) == 2, f"Batch must contain 2 elements. Got {len(batch)}"
    assert isinstance(batch[0], torch.Tensor), f"First element of batch must be a tensor. Got {type(batch[0])}"
    assert len(batch[0].shape) == 4, f"First element of batch must be a 4D tensor. Got shape {batch[0].shape}"
    return batch[0]
