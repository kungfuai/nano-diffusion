from typing import Tuple, Optional, Callable
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from .flowers_dataset import FlowersDataset
from .celeb_dataset import CelebDataset
from .pokemon_dataset import PokemonDataset
from .hugging_face_dataset import HuggingFaceDataset
from .mj_latents import MJLatentsDataset
from ..config.diffusion_training_config import DiffusionTrainingConfig as TrainingConfig


def load_data(config: TrainingConfig, collate_fn: Optional[Callable] = None) -> Tuple[DataLoader, DataLoader]:
    # TODO: consider expanding the args from config to be a more explicit list of args
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
    elif config.dataset == "mj_latents":
        print("Loading MJ latents dataset")
        full_dataset = MJLatentsDataset()
    else:
        print(f"Loading dataset from Hugging Face: {config.dataset}")
        full_dataset = HuggingFaceDataset(config.dataset, transform=None if config.data_is_latent else transform, data_is_latent=config.data_is_latent)

    train_size = int((1 - config.val_split) * len(full_dataset))
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
