"""
Module for training a VQVAE model. Code from https://github.com/nreHieW/minVAR/tree/main
"""

import argparse
from datetime import datetime
import json
import os

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.vqvae import VQVAE
from src.datasets.hugging_face_dataset import HuggingFaceDataset

vqvae_model_params = {
    "DIM": 128,
    "VOCAB_SIZE": 4096,
    "PATCH_SIZES": [1, 2, 3, 4, 6, 8],
    "CHANNELS": 3,
}
vqvae_training_params = {
    "batch_size": 128,
    "lr": 1e-4,
    "epochs": 100,
}


def make_log_dir(log_dir: str) -> str:
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"{log_dir}/{start_time}"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_data(batch_size=1024, dataset="mnist"):
    if dataset == "cifar":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_ds = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_ds = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_ds = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_ds = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        # Use HuggingFace datasets
        train_ds = HuggingFaceDataset(
            dataset_path=dataset, split="train", transform=transform
        )
        test_ds = HuggingFaceDataset(
            dataset_path=dataset, split="val", transform=transform
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=True
    )

    print(len(train_loader), len(test_loader))

    return train_loader, test_loader


def plot_images(pred, original=None):
    n = pred.size(0)
    pred = pred * 0.5 + 0.5
    pred = pred.clamp(0, 1)
    img = pred.cpu().detach()

    if original is not None:
        original = original * 0.5 + 0.5
        original = original.clamp(0, 1)
        original = original.cpu().detach()
        img = torch.cat([original, img], dim=0)

    img_grid = make_grid(img, nrow=n)
    img_grid = img_grid.permute(1, 2, 0).numpy()
    img_grid = (img_grid * 255).astype("uint8")
    plt.imshow(img_grid)
    plt.axis("off")


def print_latent_stats(latents, name="Latents"):
    mean_val = latents.mean().item()
    std_val = latents.std().item()
    min_val = latents.min().item()
    max_val = latents.max().item()
    print(
        f"{name}: mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}"
    )


def train_one_epoch(
    vq_model: VQVAE,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
):
    epoch_loss = 0
    epoch_recon_loss = 0
    pbar = tqdm(train_loader)
    for i, (x, c) in enumerate(pbar):
        x, c = x.cuda(), c.cuda()
        optimizer.zero_grad()

        # # Optionally check and print the latent stats from the encoder on the first batch of each epoch.
        # if i == 0:
        #     with torch.no_grad():
        #         latent = vq_model.encoder(x)
        #         print_latent_stats(latent, name=f"Epoch {epoch} Encoder Output")

        xhat, r_maps, idxs, scales, q_loss = vq_model(x)
        recon_loss = F.mse_loss(xhat, x)
        loss = recon_loss + q_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        pbar.set_postfix(loss=loss.item())

    epoch_loss /= len(train_loader)
    epoch_recon_loss /= len(train_loader)
    return epoch_loss, epoch_recon_loss


def test_one_epoch(vq_model: VQVAE, test_loader: DataLoader, epoch: int, log_dir: str):
    with torch.no_grad():
        total_loss = 0
        total_recon_loss = 0
        for i, (x, c) in enumerate(test_loader):
            x, c = x.cuda(), c.cuda()
            xhat, r_maps, idxs, scales, q_loss = vq_model(x)
            recon_loss = F.mse_loss(xhat, x)
            loss = recon_loss + q_loss
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()

        total_loss /= len(test_loader)
        total_recon_loss /= len(test_loader)

        print(
            f"Epoch: {epoch}, Test Loss: {total_loss}, Test Recon Loss: {total_recon_loss}"
        )

        x = x[:10, :].cuda()
        x_hat = vq_model(x)[0]

        plot_images(pred=x_hat, original=x)
        plt.savefig(f"{log_dir}/vqvae_{epoch}.png")
        plt.close()


def train_vqvae(
    dataset: str, model_params: dict, training_params: dict, log_dir: str
) -> VQVAE:
    print("=" * 10 + "Training VQVAE" + "=" * 10)

    # Initialize the VQVAE model
    vq_model = VQVAE(
        dim=model_params["DIM"],
        vocab_size=model_params["VOCAB_SIZE"],
        patch_sizes=model_params["PATCH_SIZES"],
        num_channels=model_params["CHANNELS"],
    )

    # Move the model to the GPU
    vq_model = vq_model.to("cuda")

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(vq_model.parameters(), lr=training_params["lr"])

    # Get the data loaders
    train_loader, test_loader = get_data(
        batch_size=training_params["batch_size"], dataset=dataset
    )

    # Train the model
    for epoch in range(training_params["epochs"]):
        epoch_loss, epoch_recon_loss = train_one_epoch(
            vq_model, train_loader, optimizer
        )
        print(f"Epoch: {epoch}, Loss: {epoch_loss}, Recon Loss: {epoch_recon_loss}")

        if epoch % 5 == 0:
            test_one_epoch(vq_model, test_loader, epoch, log_dir)

    # Save the model
    dataset_name = dataset.split("/")[-1]  # zzsi/afhq64_16k -> afhq64_16k
    torch.save(vq_model.state_dict(), f"{log_dir}/{dataset_name}_vqvae.pth")

    # Save the training params
    with open(f"{log_dir}/vqvae_training_params.json", "w") as f:
        json.dump(training_params, f)

    # Save the model params
    with open(f"{log_dir}/vqvae_model_params.json", "w") as f:
        json.dump(model_params, f)

    return vq_model


if __name__ == "__main__":
    dataset = "zzsi/afhq64_16k"
    log_dir = make_log_dir("logs/train_vqvae")
    vq_model = train_vqvae(dataset, vqvae_model_params, vqvae_training_params, log_dir)
