from typing import Tuple, Optional, Callable
import torch
from torch.utils.data import DataLoader, random_split
from config import TrainingConfig
from mj_latents import MJLatentsDataset


def load_data(config: TrainingConfig, collate_fn: Optional[Callable] = None) -> Tuple[DataLoader, DataLoader]:
    full_dataset = MJLatentsDataset()

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    return train_dataloader, val_dataloader
    
