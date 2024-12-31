from .unets import UNet
from .config import TrainingConfig


def create_unet_model(config: TrainingConfig, device: str):
    denoising_model = UNet(
        image_size=config.resolution,
        text_embed_dim=config.text_embed_dim,
    ).to(device)

    print(f"model params: {sum(p.numel() for p in denoising_model.parameters()) / 1e6:.2f} M")
    return denoising_model