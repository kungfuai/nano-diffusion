import tempfile
from pathlib import Path
import numpy as np
from typing import Union
import torch
from torch.utils.data import DataLoader
from src.config.diffusion_training_config import DiffusionTrainingConfig as TrainingConfig

try:
    from cleanfid import fid
except ImportError:
    print("cleanfid not installed. Please install it using `pip install cleanfid`.")


def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str = "cuda",
    dataset_name: str = "cifar10",
    resolution: int = 32,
    batch_size: int = 2,
) -> float:
    print(f"Computing FID for {dataset_name} with resolution {resolution}")
    with tempfile.TemporaryDirectory() as temp_dir:
        real_path = Path(temp_dir) / "real"
        gen_path = Path(temp_dir) / "gen"
        real_path.mkdir(exist_ok=True)
        gen_path.mkdir(exist_ok=True)

        for i, img in enumerate(generated_images):
            assert len(img.shape) == 3, f"Image must have 3 dimensions, got {len(img.shape)}"
            img_np = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            np.save(gen_path / f"{i}.npy", img_np)

        if dataset_name in ["cifar10"]:
            score = fid.compute_fid(
                str(gen_path),
                dataset_name=dataset_name,
                dataset_res=resolution,
                device=device,
                mode="clean",
                batch_size=batch_size,
                num_workers=0,
            )
        else:
            # Use precomputed stats for custom datasets.
            dataset_name_safe = make_dataset_name_safe_for_cleanfid(dataset_name)
            score = fid.compute_fid(
                str(gen_path),
                dataset_name=dataset_name_safe,
                dataset_res=resolution,
                device=device,
                mode="clean",
                dataset_split="custom",
                batch_size=batch_size,
                num_workers=0,
            )

    return score


def precompute_fid_stats_for_real_images(dataloader: DataLoader, config: TrainingConfig, real_images_dir: Path):
    print(f"Precomputing FID stats for {config.num_real_samples_for_fid} real images from {config.dataset}")
    count = 0
    real_images_dir.mkdir(exist_ok=True, parents=True)
    for images, _ in dataloader:
        # save individual images as npy files
        for i, img in enumerate(images):
            np_img = img.cpu().numpy().transpose(1, 2, 0)
            # do we need to scale to 0-255?
            np_img = (np_img * 255)
            idx = count * len(images) + i
            np.save(real_images_dir / f"real_images_{idx:06d}.npy", np_img)
        count += len(images)
        if count >= config.num_real_samples_for_fid:
            break
    
    dataset_name_safe = make_dataset_name_safe_for_cleanfid(config.dataset)
    fid.make_custom_stats(dataset_name_safe, str(real_images_dir), mode="clean")



def make_dataset_name_safe_for_cleanfid(dataset_name: str):
    return dataset_name.replace("/", "__")


