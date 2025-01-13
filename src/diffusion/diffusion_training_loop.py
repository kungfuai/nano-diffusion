from pathlib import Path
from typing import Optional, Dict

import torch
from torch.nn import MSELoss, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.bookkeeping.diffusion_bookkeeping import DiffusionBookkeeping
from src.diffusion.diffusion_model_components import DiffusionModelComponents
from src.config.diffusion_training_config import DiffusionTrainingConfig as TrainingConfig
from src.eval.fid import precompute_fid_stats_for_real_images
from src.diffusion import forward_diffusion
from src.bookkeeping import MiniBatch
from src.utils import scale_input
from src.diffusion.noise_scheduler import sample_timesteps


def training_loop(
    model_components: DiffusionModelComponents,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    config: TrainingConfig,
) -> int:
    print(f"Training on {config.device}")

    device = torch.device(config.device)
    denoising_model = model_components.denoising_model  # .to(device)
    ema_model = model_components.ema_model
    optimizer = model_components.optimizer
    lr_scheduler = model_components.lr_scheduler
    bookkeeping = DiffusionBookkeeping(config, model_components)

    if config.dataset not in ["cifar10"]:
        precompute_fid_stats_for_real_images(train_dataloader, config, Path(config.cache_dir) / "real_images_for_fid")

    bookkeeping.set_up_logger()

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
                denoising_model, batch, model_components.noise_schedule, optimizer, config, device, criterion
            )
            # TODO: add gradient norm logging
            # grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in denoising_model.parameters() if p.grad is not None]), 2)
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
                    model_components=model_components,
                    step=step,
                    loss=loss,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader
                )
            step += 1

    if step > config.min_steps_for_final_save:
        save_final_models(model_components, config)

    return num_examples_trained


def train_step(
    denoising_model: Module,
    batch: MiniBatch,
    noise_schedule: Dict[str, torch.Tensor],
    optimizer: Optimizer,
    config: TrainingConfig,
    device: torch.device,
    criterion: Module,
) -> torch.Tensor:
    optimizer.zero_grad()
    x_0 = batch.x.to(device)
    x_0 = scale_input(x_0, config)

    true_noise = common_noise = torch.randn(x_0.shape).to(device)
    t = sample_timesteps(num_examples=x_0.shape[0], config=config)
    x_t, _ = forward_diffusion(x_0, t, noise_schedule, noise=common_noise)

    model_kwargs = {"t": t, "x": x_t}
    if config.conditional and batch.text_emb is not None:
        model_kwargs.update({
            "y": batch.text_emb.to(device),
            "p_uncond": config.cond_drop_prob
        })
    predicted_noise = denoising_model(**model_kwargs)
    predicted_noise = predicted_noise.sample if hasattr(predicted_noise, "sample") else predicted_noise

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
