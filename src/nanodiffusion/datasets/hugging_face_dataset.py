"""
Used for loading any general from Hugging Face Datasets
https://huggingface.co/datasets/keremberke/pokemon-classification
"""
import warnings
import os
from datasets import load_dataset, load_from_disk
import numpy as np
import torch
from torch.utils.data import Dataset
from ..models.text_encoder import TextEncoder


class HuggingFaceDataset(Dataset):
    def __init__(self, dataset_path: str, split="train", transform=None, caption_column: str = "text", 
                 data_is_latent: bool = False, conditional: bool = False, text_encoder: str = None,
                 dataset_cache_dir_for_processed_data: str = None,
                 device: str = "cuda:0"):
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform
        self.caption_column = caption_column
        self.data_is_latent = data_is_latent
        self.conditional = conditional
        self.dataset_cache_dir_for_processed_data = dataset_cache_dir_for_processed_data or self.default_dataset_cache_dir
        
        if conditional and os.path.exists(self.dataset_cache_dir_for_processed_data):
            print(f"Loading cached dataset from {self.dataset_cache_dir_for_processed_data}")
            self.dataset = load_from_disk(self.dataset_cache_dir_for_processed_data)
        else:
            self.dataset = load_dataset(dataset_path, split=split)
            if conditional:
                assert text_encoder is not None, "text_encoder must be provided when conditional=True"
                self.text_encoder = TextEncoder(text_encoder, device=device)
                self.text_encoder.eval()
                # Compute and cache text embeddings
                self.compute_and_cache_text_embeddings(self.dataset_cache_dir_for_processed_data)
        
        self.image_key = self.find_image_key()
    
    @property
    def default_dataset_cache_dir(self):
        user = os.environ.get("USER", os.path.expanduser("~").split("/")[-1])
        return f"/home/{user}/.cache/huggingface/processed_datasets/{self.dataset_path}_with_text_embeddings_{self.split}"

    def compute_and_cache_text_embeddings(self, cache_path: str):
        """Compute and cache text embeddings for all examples in the dataset using dataset.map."""
        if not self.conditional:
            return

        # Check if "text_emb" already exists in the dataset
        if "text_emb" in self.dataset[0].keys():
            warnings.warn("text_emb is already in the dataset. It will be overwritten.")

        print("Computing and caching text embeddings...")
        with torch.no_grad():
            def add_text_embeddings(example):
                text = example[self.caption_column]
                text_emb = self.text_encoder([text])[0].cpu().numpy()
                # Return all existing fields plus the new text_emb field
                return {**example, "text_emb": text_emb}
            
            self.dataset = self.dataset.map(
                add_text_embeddings,
                batched=False,
                desc="Adding text embeddings"
            )

            # Save the dataset with text embeddings to disk
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self.dataset.save_to_disk(cache_path)
        print(f"Text embeddings computed and cached to {cache_path}.")

    def find_image_key(self) -> str:
        # Check if the dataset has the "image" key
        # NOTE: Can exapnd this to other common keys if needed
        if "image" in self.dataset[0].keys():
            return "image"
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.data_is_latent:
            image = self.dataset[idx]["image_emb"]  # TODO: this is hardcoded and not explicit to the user
        else:
            if not self.image_key:
                return self.dataset[idx]
            image = self.dataset[idx][self.image_key]
            image = image.convert("RGB")  # Convert to RGB to ensure 3 channels
            if self.transform:
                image = self.transform(image)
                
        text_emb = None
        if self.conditional and "text_emb" in self.dataset[idx].keys():
            text_emb = np.array(self.dataset[idx].get('text_emb'))
            text_emb = text_emb.reshape(text_emb.shape[0], -1)
        
        return image, text_emb
