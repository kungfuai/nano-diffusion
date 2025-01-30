"""
CelebA faces dataset from Hugging Face Datasets
https://huggingface.co/datasets/huggan/CelebA-faces-with-attributes
"""

from datasets import load_dataset
from torch.utils.data import Dataset


class CelebDataset(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("bitmind/celeb-a-hq", split="train")
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        label = 0  # No label in CelebA dataset
        if self.transform:
            image = self.transform(image)
        return image, label
