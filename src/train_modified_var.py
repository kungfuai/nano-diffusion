import json
import os
import os.path as osp

import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets, transforms
from tqdm import tqdm


from src.datasets.hugging_face_dataset import HuggingFaceDataset

# VQVAE from oritinal paper https://github.com/FoundationVision/VAR
from src.models.full_vqvae import VQVAE

# VAR from https://github.com/nreHieW/minVAR/tree/main
from src.models.modified_var import (
    VAR,
)
from src.train_vqvae import make_log_dir, plot_images


var_model_params = {
    "dim": 160,
    "n_heads": 16,
    "n_layers": 16,
    "patch_sizes": [1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
    "n_classes": 10,
}
var_training_params = {
    "batch_size": 128,
    "lr": 1e-4,
    "epochs": 100,
}


def build_vae(
    # Shared args
    device,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
    # VQVAE args
    V=4096,
    Cvae=32,
    ch=160,
    share_quant_resi=4,
) -> VQVAE:
    vae_local = VQVAE(
        vocab_size=V,
        z_channels=Cvae,
        ch=ch,
        test_mode=True,
        share_quant_resi=share_quant_resi,
        v_patch_nums=patch_nums,
    ).to(device)
    return vae_local


def download_checkpoint():
    MODEL_DEPTH = 16
    assert MODEL_DEPTH in {16, 20, 24, 30}

    # Download checkpoint
    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt = "vae_ch160v4096z32.pth"
    if not osp.exists(vae_ckpt):
        os.system(f"wget {hf_home}/{vae_ckpt}")

    # Build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if "vae" not in globals() or "var" not in globals():
        vae = build_vae(
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,  # hard-coded VQVAE hyperparameters
            device=device,
            patch_nums=patch_nums,
        )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"prepare finished.")


def get_data(batch_size=1024, dataset="mnist"):
    if dataset == "cifar":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (256, 256), interpolation=transforms.InterpolationMode.BICUBIC
                ),
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
                transforms.Resize(
                    (256, 256), interpolation=transforms.InterpolationMode.BICUBIC
                ),
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
                transforms.Resize(
                    (256, 256), interpolation=transforms.InterpolationMode.BICUBIC
                ),
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


def train_one_epoch(
    var_model: VAR,
    vqvae: VQVAE,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    epoch_loss = 0
    pbar = tqdm(train_loader)
    for i, (x, c) in enumerate(pbar):
        x, c = x.cuda(), c.cuda()
        optimizer.zero_grad()

        # !!! This is potentially where things are going wrong !!!
        # If you compare this with what's in train_var.py you can see the changes I've made
        # These changes attempt to bridge the gap between the original VAR and the minVAR implementation
        gt_idx_Bl = vqvae.img_to_idxBl(x)
        idx_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l = vqvae.quantize.idxBl_to_var_input(gt_idx_Bl)
        # print(f"x_BLCv_wo_first_l stats: {x_BLCv_wo_first_l.min().item()}, {x_BLCv_wo_first_l.mean().item()}, {x_BLCv_wo_first_l.max().item()}")
        logits_BLV = var_model(x_BLCv_wo_first_l, cond=c)
        # print(f"logits_BLV stats: {logits_BLV.min().item()}, {logits_BLV.mean().item()}, {logits_BLV.max().item()}")
        loss = F.cross_entropy(
            logits_BLV.view(-1, logits_BLV.size(-1)), idx_BL.view(-1)
        )
        # !!!!!!

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    epoch_loss /= len(train_loader)

    return epoch_loss


def train_modified_var(
    vqvae: VQVAE,
    dataset: str,
    var_model_params: dict,
    var_training_params: dict,
    log_dir: str,
) -> VAR:
    print("=" * 10 + "Training VAR" + "=" * 10)
    vqvae.eval()

    for param in vqvae.parameters():
        param.requires_grad = False

    var_model = VAR(
        vqvae=vqvae,
        dim=var_model_params["dim"],
        n_heads=var_model_params["n_heads"],
        n_layers=var_model_params["n_layers"],
        patch_sizes=var_model_params["patch_sizes"],
        n_classes=var_model_params["n_classes"],
    )
    optimizer = torch.optim.AdamW(var_model.parameters(), lr=1e-4)

    train_loader, test_loader = get_data(batch_size=16, dataset=dataset)

    print(f"VQVAE Parameters: {sum(p.numel() for p in vqvae.parameters())/1e6:.2f}M")
    print(f"VAR Parameters: {sum(p.numel() for p in var_model.parameters())/1e6:.2f}M")

    var_model = var_model.to("cuda")

    for epoch in range(var_training_params["epochs"]):
        epoch_loss = train_one_epoch(var_model, vqvae, train_loader, optimizer)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")

        if epoch % 5 == 0:
            with torch.no_grad():

                cond = torch.arange(10).cuda()
                out_B3HW = var_model.generate(cond, 0)
                plot_images(pred=out_B3HW)

                plt.savefig(f"{log_dir}/var_{epoch}.png")
                plt.close()

    dataset_name = dataset.split("/")[-1]  # zzsi/afhq64_16k -> afhq64_16k
    torch.save(var_model.state_dict(), f"{log_dir}/{dataset_name}_var.pth")

    with open(f"{log_dir}/var_model_params.json", "w") as f:
        json.dump(var_model_params, f)

    with open(f"{log_dir}/var_training_params.json", "w") as f:
        json.dump(var_training_params, f)

    return var_model


if __name__ == "__main__":
    dataset = "zzsi/afhq64_16k"

    log_dir = make_log_dir("logs/train_modified_var")

    vqvae = build_vae(
        device="cuda",
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    )

    train_modified_var(vqvae, dataset, var_model_params, var_training_params, log_dir)
