import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.config.diffusion_training_config import DiffusionTrainingConfig as TrainingConfig
from torch.nn import Module
from tqdm.auto import tqdm

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
            img_np = img.cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_np = (img_np * 255).astype(np.uint8).transpose(1, 2, 0)
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


def _build_stats_filename(dataset_name: str, resolution: int = None, model_name: str = "inception_v3") -> str:
    """Build the standardized filename for FID statistics.
    
    Args:
        dataset_name: Name of the dataset
        resolution: Image resolution. This is not used for custom datasets.
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


def copy_stats_to_real_images_dir(real_images_dir: Path, dataset_name: str, resolution: int = None):
    import shutil, os
    import cleanfid
    
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    stats_filename = _build_stats_filename(dataset_name, resolution)
    outf = os.path.join(stats_folder, stats_filename)
    shutil.copy(outf, real_images_dir)


def precompute_fid_stats_for_real_images(dataloader: DataLoader, config: TrainingConfig, real_images_dir: Path, vae: Module = None):
    if config.data_is_latent:
        return precompute_fid_stats_for_real_image_latents(dataloader, config, real_images_dir, vae)
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
    copy_stats_to_real_images_dir(real_images_dir, config.dataset)


def precompute_fid_stats_for_real_image_latents(
    dataloader: DataLoader,
    config: TrainingConfig,
    real_images_dir: Path,
    vae: Module,
) -> None:
    print(f"Precomputing FID stats for {config.num_real_samples_for_fid} real images from {config.dataset}")
    dataset_name_safe = make_dataset_name_safe_for_cleanfid(config.dataset)
    real_images_dir = real_images_dir / dataset_name_safe
    real_images_dir.mkdir(exist_ok=True, parents=True)
    existing_fid_stats_path = fid_stats_exists(config.dataset, config.resolution, config, real_images_dir)
    if existing_fid_stats_path:
        print(f"FID stats already exist for {config.dataset} at {existing_fid_stats_path}")
        return
    count = 0
    for batch in tqdm(dataloader, desc="Collecting real images for FID stats precomputation"):
        # TODO: "image_emb" is hardcoded
        latents = batch["image_emb"]
        if isinstance(latents, list):
            # print(latents)
            # print(latents[0][0][0].shape)
            latents = torch.stack(latents).to(config.device)

        latents = latents
        # Decode latents to images using VAE
        with torch.no_grad():
            latents = latents.to(vae.device)
            images = vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
        
        # save individual images as npy files
        for i, img in enumerate(images):
            np_img = img.cpu().numpy().transpose(1, 2, 0)
            # Scale to 0-255 range
            np_img = (np_img * 255)
            idx = count * len(images) + i
            np.save(real_images_dir / f"real_images_{idx:06d}.npy", np_img)
        count += len(images)
        if count >= config.num_real_samples_for_fid:
            break
    
    dataset_name_safe = make_dataset_name_safe_for_cleanfid(config.dataset)
    fid.make_custom_stats(dataset_name_safe, str(real_images_dir), mode="clean")
    # also copy the stats to the real_images_dir
    copy_stats_to_real_images_dir(real_images_dir, config.dataset)



def make_dataset_name_safe_for_cleanfid(dataset_name: str):
    return dataset_name.replace("/", "__")


def evaluate_pretrained_model(
        wandb_run_path: str,
        wandb_file_name: str,
        is_cfm: bool = False,
        dataset_name: str = "zzsi/afhq64_16k",
        resolution: int = 64,
        net: str = "unet_small",
        num_denoising_steps: int = 1000,
        fid_batch_size: int = 2,
        n_samples: int = 50000,
        device: str = "cuda:0",
        seed: int = 0,
    ):
    from src.bookkeeping.wandb_utils import load_model_from_wandb
    from src.models.factory import create_model
    from src.datasets import load_data
    from src.config.diffusion_training_config import DiffusionTrainingConfig
    from src.train_cfm import TrainingConfig, generate_samples_with_flow_matching
    from src.bookkeeping.diffusion_bookkeeping import generate_samples_by_denoising
    from src.diffusion.diffusion_model_components import create_noise_schedule
    
    # create an empty model
    denoising_model = create_model(net=net, resolution=resolution)
    # load the pretrained weights from wandb into the model
    load_model_from_wandb(
        model=denoising_model, run_path=wandb_run_path, file_name=wandb_file_name
    )
    denoising_model = denoising_model.to(device)
    denoising_model.eval()

    if is_cfm:
        config = TrainingConfig(
            net=net,
            resolution=resolution,
            dataset=dataset_name,
            device=device,
        )
    else:
        config = DiffusionTrainingConfig(
            net=net,
            resolution=resolution,
            dataset=dataset_name,
            device=device,
        )
        noise_schedule = create_noise_schedule(n_T=num_denoising_steps, device=device)

    train_dataloader, val_dataloader = load_data(config)

    # precompute fid stats on real images
    if dataset_name not in ["cifar10"]:
        precompute_fid_stats_for_real_images(
            dataloader=train_dataloader,
            config=config,
            real_images_dir=Path(config.cache_dir) / "real_images_for_fid"
        )
    # generate n_samples synthetic images
    count = 0
    sampled_images = []
    generation_batch_size = 100
    num_batches = (n_samples + generation_batch_size - 1) // generation_batch_size
    for i in range(num_batches):
        current_batch_size = min(generation_batch_size, n_samples - len(sampled_images))
        if is_cfm:  
            sampled_images.append(generate_samples_with_flow_matching(
                denoising_model, device, current_batch_size, resolution=resolution,
                in_channels=config.in_channels, seed=seed+i, num_denoising_steps=num_denoising_steps))
        else:
            x_t = torch.randn(current_batch_size, config.in_channels, resolution, resolution).to(device)
            samples = generate_samples_by_denoising(
                denoising_model, x_t, noise_schedule=noise_schedule, n_T=num_denoising_steps, device=device, seed=seed+i, quiet=True)
            samples = (samples / 2 + 0.5).clamp(0, 1)
            sampled_images.append(samples)
        count += current_batch_size
        print(f"Generated {count} out of {n_samples} images. num_denoising_steps: {num_denoising_steps}")
    sampled_images = torch.cat(sampled_images, dim=0)

    # compute fid score
    fid_score = compute_fid(
        real_images=None,  # Not needed as we're using pre-computed stats
        generated_images=sampled_images,
        device=device,
        dataset_name=dataset_name,
        resolution=resolution,
        batch_size=fid_batch_size,
    )
    print(f"FID Score: {fid_score:.4f}")
    return fid_score


if __name__ == "__main__":
    # Example usage:
    # GPU_DEVICES=0 bin/run.sh python -m src.eval.fid --wandb_run_path zzsi_kungfu/nano-diffusion/qixqrrc0 --wandb_file_name logs/train/2025-01-05_20-48-33/final_model.pth --dataset_name cifar10 --resolution 32 --net unet --n_samples 500 --fid_batch_size 2 --num_denoising_steps 1000  
    from argparse import ArgumentParser
    # TODO: given the wandb run path, get the hyperparameters from the wandb run, e.g. resolution, net, etc.
    parser = ArgumentParser()
    parser.add_argument("--wandb_run_path", type=str, default="zzsi_kungfu/nano-diffusion/gsygnamk")
    parser.add_argument("--wandb_file_name", type=str, default="logs/train/2025-01-05_03-13-30/final_model.pth")
    parser.add_argument("--dataset_name", type=str, default="zzsi/afhq64_16k")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--net", type=str, default="unet_small")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--fid_batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--is_cfm", action="store_true")
    parser.add_argument("--num_denoising_steps", type=int, default=100)
    args = parser.parse_args()

    evaluate_pretrained_model(
        wandb_run_path=args.wandb_run_path,
        wandb_file_name=args.wandb_file_name,
        dataset_name=args.dataset_name,
        resolution=args.resolution,
        net=args.net,
        n_samples=args.n_samples,
        fid_batch_size=args.fid_batch_size,
        device=args.device,
        seed=args.seed,
        num_denoising_steps=args.num_denoising_steps,
        is_cfm=args.is_cfm,
    )
