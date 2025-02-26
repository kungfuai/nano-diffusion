"""
Module for training a VAR model. Code from https://github.com/nreHieW/minVAR/tree/main
"""

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.var import VAR
from src.models.vqvae import VQVAE
from src.train_vqvae import get_data, make_log_dir, plot_images, vqvae_model_params

var_model_params = {
    "DIM": 512,
    "N_HEADS": 16,
    "N_LAYERS": 16,
    "PATCH_SIZES": [1, 2, 3, 4, 6, 8],
    "N_CLASSES": 10,
}
var_training_params = {
    "batch_size": 128,
    "lr": 1e-4,
    "epochs": 100,
}


def load_vqvae(vqvae_model_path: str) -> VQVAE:
    vqvae = VQVAE(
        vqvae_model_params["DIM"],
        vqvae_model_params["VOCAB_SIZE"],
        vqvae_model_params["PATCH_SIZES"],
        num_channels=vqvae_model_params["CHANNELS"],
    )
    vqvae.load_state_dict(torch.load(vqvae_model_path))
    vqvae = vqvae.to("cuda")
    vqvae.eval()

    for param in vqvae.parameters():
        param.requires_grad = False

    return vqvae


def train_one_epoch(
    var_model: VAR,
    vqvae: VQVAE,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    epoch_loss = 0
    for i, (x, c) in enumerate(train_loader):
        x, c = x.cuda(), c.cuda()
        optimizer.zero_grad()

        _, _, idxs_R_BL, scales_BlC, _ = vqvae(x)
        idx_BL = torch.cat(idxs_R_BL, dim=1)
        scales_BlC = scales_BlC.cuda()
        logits_BLV = var_model(scales_BlC, cond=c)
        loss = F.cross_entropy(
            logits_BLV.view(-1, logits_BLV.size(-1)), idx_BL.view(-1)
        )

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    return epoch_loss


def train_var(
    vqvae: VQVAE,
    dataset: str,
    model_params: dict,
    training_params: dict,
    log_dir: str,
) -> VAR:
    print("=" * 10 + "Training VAR" + "=" * 10)
    var_model = VAR(
        vqvae=vqvae,
        dim=model_params["DIM"],
        n_heads=model_params["N_HEADS"],
        n_layers=model_params["N_LAYERS"],
        patch_sizes=model_params["PATCH_SIZES"],
        n_classes=model_params["N_CLASSES"],
    )
    optimizer = torch.optim.AdamW(var_model.parameters(), lr=training_params["lr"])

    print(f"VQVAE Parameters: {sum(p.numel() for p in vqvae.parameters())/1e6:.2f}M")
    print(f"VAR Parameters: {sum(p.numel() for p in var_model.parameters())/1e6:.2f}M")

    train_loader, test_loader = get_data(
        batch_size=training_params["batch_size"], dataset=dataset
    )
    var_model = var_model.to("cuda")
    for epoch in tqdm(range(training_params["epochs"])):
        epoch_loss = train_one_epoch(var_model, vqvae, train_loader, optimizer)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")

        if epoch % 5 == 0:
            with torch.no_grad():

                cond = torch.arange(10).cuda()
                out_B3HW = var_model.generate(cond, 0)
                plot_images(pred=out_B3HW)

                plt.savefig(f"{log_dir}/var_{epoch}.png")
                plt.close()

    # Save the model
    dataset_name = dataset.split("/")[-1]  # zzsi/afhq64_16k -> afhq64_16k
    torch.save(var_model.state_dict(), f"{log_dir}/{dataset_name}_var.pth")

    # Save the model params
    with open(f"{log_dir}/var_model_params.json", "w") as f:
        json.dump(model_params, f)

    # Save the training params
    with open(f"{log_dir}/var_training_params.json", "w") as f:
        json.dump(training_params, f)

    return var_model


if __name__ == "__main__":
    # Requires a trained VQVAE model as input
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_model_path", type=str, required=True)
    args = parser.parse_args()

    dataset = "zzsi/afhq64_16k"
    log_dir = make_log_dir("logs/train_var")

    vqvae = load_vqvae(args.vqvae_model_path)

    var_model = train_var(
        vqvae=vqvae,
        dataset=dataset,
        model_params=var_model_params,
        training_params=var_training_params,
        log_dir=log_dir,
    )
