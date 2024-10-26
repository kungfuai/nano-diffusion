"""
Oxford Flowers Dataset from Hugging Face Datasets
https://huggingface.co/datasets/oxford_flowers102
"""

from datasets import load_dataset
from torch.utils.data import Dataset


class FlowersDatset(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("nelorth/oxford-flowers", split="train")
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        label = self.dataset[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label
