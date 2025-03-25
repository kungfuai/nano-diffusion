from pathlib import Path
from typing import Optional, Dict
from contextlib import nullcontext

import torch
from torch.nn import MSELoss, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from bookkeeping import Bookkeeping
from diffusion_model_components import DiffusionModelComponents
from config import TrainingConfig
from mini_batch import MiniBatch
from diffusion import Diffusion
from fid import precompute_fid_stats_for_real_images


def training_loop(
    model_components: DiffusionModelComponents,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    config: TrainingConfig,
) -> int:
    print(f"Training on {config.device}")

    denoising_model = model_components.denoising_model  # .to(device)
    optimizer = model_components.optimizer

    if config.dataset not in ["cifar10"] and config.fid_every > 0:
        precompute_fid_stats_for_real_images(train_dataloader, config, Path(config.cache_dir) / "real_images_for_fid", vae=model_components.vae)
        
    bookkeeping = Bookkeeping(config, model_components)
    bookkeeping.set_up_logger()

    accelerator = None
    if config.accelerator:
        from accelerate import Accelerator

        accelerator = Accelerator(
            mixed_precision="fp16" if config.fp16 else "no",
        )
        print(f"Using accelerator with mixed precision {accelerator.mixed_precision}.")
        denoising_model, train_dataloader, optimizer = accelerator.prepare(denoising_model, train_dataloader, optimizer)

    step = 0
    num_examples_trained = 0
    criterion = MSELoss()

    while step < config.total_steps:
        for batch in train_dataloader:
            if step >= config.total_steps:
                break

            batch = MiniBatch.from_dataloader_batch(batch)
            num_examples_trained += batch.num_examples

            denoising_model.train()
            loss = train_step(
                batch, denoising_model, model_components.diffusion, optimizer, config, criterion, accelerator
            )
            denoising_model.eval()

            with torch.no_grad():
                bookkeeping.run_callbacks(
                    config=config,
                    step=step,
                    loss=loss,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                )
            step += 1

    if step > config.min_steps_for_final_save:
        save_final_models(model_components, config)
    
    if accelerator:
        accelerator.end_training()

    return num_examples_trained


def train_step(
    batch: MiniBatch,  # data
    denoising_model: Module,  # model
    diffusion: Diffusion,  # diffusion algorithm for teaching (data augmentation)
    optimizer: Optimizer,  # optimizer
    config: TrainingConfig,  # config
    criterion: Module,  # loss function
    accelerator = None,  # accelerator utility
) -> torch.Tensor:
    context = accelerator.accumulate() if accelerator else nullcontext()

    with context:
        optimizer.zero_grad()
        inputs, targets = diffusion.prepare_training_examples(batch)
        assert str(inputs["x"].device) == str(config.device), f"Inputs are on {inputs['x'].device}, but config is on {config.device}"
        try:
            predictions = denoising_model(**inputs)
        except Exception as e:
            for k, v in inputs.items():
                if hasattr(v, "shape"):
                    print("------ ", k, v.shape)
            raise e
        predictions = predictions.sample if hasattr(predictions, "sample") else predictions

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
