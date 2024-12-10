from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from sklearn.datasets import make_swiss_roll

from .config import TrainingConfig


class SimpleDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def load_data(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Load the data and return the train and validation dataloaders.

    Args:
        config: TrainingConfig object containing batch_size and other parameters
    Returns:
        train_dataloader: DataLoader for the training data
        val_dataloader: DataLoader for the validation data
    """
    n = int(1e+6)
    x, _ = make_swiss_roll(n_samples=n, noise=0)
    x = x[:, [0, 2]]  # Keep only x and z coordinates
    scaling = 2
    x = (x - x.mean()) / x.std() * scaling

    # Split into train/val
    x_train = x[:int(n * 0.8), :]
    x_val = x[int(n * 0.8):, :]

    # Create datasets and dataloaders
    train_dataset = SimpleDataset(x_train)
    val_dataset = SimpleDataset(x_val)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=0
    )

    return train_dataloader, val_dataloader
