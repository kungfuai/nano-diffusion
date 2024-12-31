from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from datasets import load_dataset
from .config import TrainingConfig


class HuggingFaceDataset(Dataset):
    def __init__(self, dataset_path: str, transform=None, caption_column: str = "caption_blip2-opt-2.7b"):
        self.dataset = load_dataset(dataset_path, split="train")
        self.transform = transform
        self.image_key = self.find_image_key()
        self.caption_column = caption_column

    def find_image_key(self) -> str:
        # Check if the dataset has the "image" key
        # NOTE: Can exapnd this to other common keys if needed
        if "image" in self.dataset[0].keys():
            return "image"
        raise KeyError("Dataset does not have an 'image' key")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][self.image_key]
        image = image.convert("RGB")  # Convert to RGB to ensure 3 channels
        # By default, set label to 0 to conform to current expected batch format
        label = 0
        if self.transform:
            image = self.transform(image)
        return image, {
            "label": label,
            "text": self.dataset[idx][self.caption_column],
        }


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
    full_dataset = HuggingFaceDataset(config.dataset, transform=transform, caption_column=config.caption_column)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
    return train_dataset, val_dataset
    
