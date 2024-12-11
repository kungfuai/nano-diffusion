"""
Pokemon dataset from Hugging Face Datasets
https://huggingface.co/datasets/keremberke/pokemon-classification
"""

from datasets import load_dataset
from torch.utils.data import Dataset


class PokemonDataset(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset(
            "keremberke/pokemon-classification", name="full", split="train"
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        label = self.dataset[idx]["labels"]
        if self.transform:
            image = self.transform(image)
        return image, label
