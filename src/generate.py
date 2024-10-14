"""
Load a wandb run and generate samples from it.
"""
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import wandb
import torch
from torchvision.utils import save_image, make_grid
from src.train import generate_samples_by_denoising, create_noise_schedule, create_model


def main():
    parser = ArgumentParser()
    parser.add_argument("--net", type=str, default="unet_small")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--wandb_run_path", type=str, default="zzsi_kungfu/nano-diffusion/ld0i9jv2")
    parser.add_argument("--wandb_file_name", type=str, default="logs/train/final_model.pth")
    args = parser.parse_args()

    if args.model_path is not None:
        loaded_model_state = torch.load(args.model_path, weights_only=True)
    elif args.wandb_run_path is not None:
        print(f"Restoring model from {args.wandb_run_path} and {args.wandb_file_name}")
        model_to_resume_from = wandb.restore(args.wandb_file_name, run_path=args.wandb_run_path)
        loaded_model_state = torch.load(model_to_resume_from.name, weights_only=True)
    else:
        raise ValueError("Either model_path or wandb_run_path must be provided")
    model = create_model(net=args.net)
    model.load_state_dict(loaded_model_state)

    device = "cuda:0"
    model.to(device)
    resolution = args.resolution
    torch.manual_seed(0)
    x_T = torch.randn(8, 3, resolution, resolution).to(device)
    noise_schedule = create_noise_schedule(n_T=1000, device=device)
    samples = generate_samples_by_denoising(
        denoising_model=model, x_T=x_T,
        noise_schedule=noise_schedule, n_T=1000, device=device)
    print(samples.min(), samples.max())
    images_processed = (samples * 255).cpu()
    # save to a file
    images_processed = make_grid(images_processed, nrow=2)
    images_processed = images_processed.permute(1, 2, 0).numpy().astype("uint8")
    Path("logs/test").mkdir(parents=True, exist_ok=True)
    Image.fromarray(images_processed).save("logs/test/samples.png")
    print(f"Samples saved to logs/test/samples.png")


if __name__ == "__main__":
    main()
