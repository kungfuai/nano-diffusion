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


def _build_stats_filename(dataset_name: str, resolution: int, model_name: str = "inception_v3") -> str:
    """Build the standardized filename for FID statistics.
    
    Args:
        dataset_name: Name of the dataset
        resolution: Image resolution
        model_name: Name of the model used for FID computation
    
    Returns:
        Formatted filename for the stats file
    """
    mode = "clean"
    split = "custom"
    model_modifier = "" if model_name == "inception_v3" else f"_{model_name}"
    name = make_dataset_name_safe_for_cleanfid(dataset_name)
    
    resolution = "na"  # hard coded to na for custom datasets
    return f"{name}_{mode}{model_modifier}_{split}_{resolution}.npz".lower()


def fid_stats_exists(dataset_name: str, resolution: int, config: TrainingConfig, real_images_dir: Path):
    import cleanfid, os, shutil

    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    stats_filename = _build_stats_filename(dataset_name, resolution)
    
    # Check in cache dir
    filepath_in_cache_dir = real_images_dir / stats_filename
    if filepath_in_cache_dir.exists():
        print(f"FID stats file {stats_filename} found in {real_images_dir}")
        print(f"Copying to {stats_folder}")
        # Create the stats folder if it doesn't exist
        os.makedirs(stats_folder, exist_ok=True)
        # copy the file to the stats folder
        shutil.copy(filepath_in_cache_dir, stats_folder)
        # Make sure to return the filepath in the stats folder
        assert os.path.exists(os.path.join(stats_folder, stats_filename)), f"FID stats file {stats_filename} not found in {stats_folder}"
    else:
        print(f"FID stats file {stats_filename} not found in {real_images_dir}")
        
    # Check in cleanfid stats folder
    outf = os.path.join(stats_folder, stats_filename)
    return outf if os.path.exists(outf) else None


def copy_stats_to_real_images_dir(real_images_dir: Path, dataset_name: str, resolution: int):
    import shutil, os
    import cleanfid
    
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    stats_filename = _build_stats_filename(dataset_name, resolution)
    outf = os.path.join(stats_folder, stats_filename)
    shutil.copy(outf, real_images_dir)


def precompute_fid_stats_for_real_images(dataloader: DataLoader, config: TrainingConfig, real_images_dir: Path):
    print(f"Precomputing FID stats for {config.num_real_samples_for_fid} real images from {config.dataset}")
    dataset_name_safe = make_dataset_name_safe_for_cleanfid(config.dataset)
    real_images_dir = real_images_dir / dataset_name_safe
    real_images_dir.mkdir(exist_ok=True, parents=True)
    existing_fid_stats_path = fid_stats_exists(config.dataset, config.resolution, config, real_images_dir)
    if existing_fid_stats_path:
        print(f"FID stats already exist for {config.dataset} at {existing_fid_stats_path}")
        return
    count = 0
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
    # also copy the stats to the real_images_dir
    copy_stats_to_real_images_dir(real_images_dir, config.dataset, config.resolution)



def make_dataset_name_safe_for_cleanfid(dataset_name: str):
    return dataset_name.replace("/", "__")


