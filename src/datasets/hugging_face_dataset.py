"""
Used for loading any general from Hugging Face Datasets
https://huggingface.co/datasets/keremberke/pokemon-classification
"""

from datasets import Image, load_dataset
import numpy as np
from torch.utils.data import Dataset


class HuggingFaceDataset(Dataset):
    def __init__(self, dataset_path: str, split="train", transform=None):
        self.dataset = load_dataset(dataset_path, split=split)
        self.transform = transform
        self.image_key = self.find_image_key()

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
        return image, label
