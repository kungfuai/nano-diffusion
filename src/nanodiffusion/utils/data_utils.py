import numpy as np
import torch
import torchvision.transforms.functional as F
import os
from torchvision import transforms
from datasets import load_dataset, Dataset
from ..models.text_encoder import TextEncoder
from ..config.diffusion_training_config import DiffusionTrainingConfig as TrainingConfig
from typing import Dict


class SquarePad:
    """
    Borrowed from https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/5
    """
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')
    

def scale_input(x: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    """
    Scale the input tensor by the vae scale factor if the data is latent.

    The latent embeddings from VAE are typically too large, so we scale them down (e.g. by a factor of 0.18215)
    so that it is closer to a unit Gaussian.

    The generated latents using the denoiser will need to be scaled up by the same factor
    before being passed to the VAE decoder.
    """
    if config.data_is_latent and config.vae_scale_multiplier is not None:
        return x * config.vae_scale_multiplier
    return x


def extract_text_embeddings(config: TrainingConfig, resize_image: bool = True) -> Dict[str, Dataset]:
    """
    Extract the text embeddings from an HuggingFace image-text dataset.
    """

    dataset_name = config.dataset
    text_encoder = TextEncoder(config.text_encoder, device=config.device)
    text_encoder.eval()
    def get_text_embeddings(text: str):
        with torch.no_grad():
            return text_encoder([text])[0]

    dataset_dict = {}
    for split in ["train"]:
        dst_path = f"data/processed/{dataset_name}_w_text_emb_{split}"
        if os.path.exists(dst_path):
            continue
        captioned_emoji_train = load_dataset(dataset_name, split=split)
        # assert that the dataset has "image", "text" columns
        assert "image" in captioned_emoji_train.features, f"Dataset must have an 'image' column. Got {captioned_emoji_train.features}"
        assert "text" in captioned_emoji_train.features, f"Dataset must have a 'text' column. Got {captioned_emoji_train.features}"
        captioned_emoji_train_w_text_emb = captioned_emoji_train.map(
            lambda x: {
                "text": x["text"],
                "text_emb": get_text_embeddings(x["text"]),
                "image": x["image"].resize((config.resolution, config.resolution)) if resize_image else x["image"]
            },
            batched=False,
        )
        # save to disk
        captioned_emoji_train_w_text_emb.save_to_disk(dst_path)
        dataset_dict[split] = captioned_emoji_train_w_text_emb
    return dataset_dict



