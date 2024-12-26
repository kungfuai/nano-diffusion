import itertools
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict

from .diffusion import forward_diffusion
from .bookkeeping import Bookkeeping
from .config import TrainingConfig

def train(
    config: TrainingConfig,
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    noise_schedule: Dict,
    optimizer: torch.optim.Optimizer,
    steps: int=100,
    silent: bool=False,
    bookkeeping: Bookkeeping=None
) -> float:
  device = config.device
  num_denoising_steps = config.num_denoising_steps
  
  model.train()
  if not silent:
    print("Training on device:", device)
  max_train_steps = steps

  loss = None
  progress_bar = tqdm(itertools.cycle(train_dataloader), total=max_train_steps, disable=silent)
  step = 0
  criterion = MSELoss()
  for batch in progress_bar:
    x_0 = batch[0]
    x_0 = x_0.float().to(device)  # x_0 is the clean data to teach the model to generate
    optimizer.zero_grad()

    true_noise = common_noise = torch.randn(x_0.shape).to(device)
    t = torch.randint(0, num_denoising_steps, (x_0.shape[0],), device=device).long()
    x_t, _ = forward_diffusion(x_0, t, noise_schedule, noise=common_noise)

    predicted_noise = model(t=t, x=x_t)

    loss = criterion(predicted_noise, true_noise)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # try commenting it out
    optimizer.step()

    step += 1

    if not silent:
      progress_bar.set_postfix({"loss": loss.cpu().item()})

    if bookkeeping:
      bookkeeping.run_callbacks(config=config, step=step, loss=loss, optimizer=optimizer, val_dataloader=val_dataloader)

    if step >= max_train_steps:
      break

  return loss