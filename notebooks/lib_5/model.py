from tld_net import Denoiser
from config import TrainingConfig


def create_denoiser_model(config: TrainingConfig, device: str):
    denoising_model = Denoiser(
        image_size=config.resolution,
        embed_dim=128,
        noise_embed_dims=256,
        patch_size=2,
        n_layers=3,
        mlp_multiplier=2,
        n_channels=4,
        dropout=0,
        cond_embed_dim=config.cond_embed_dim,
    ).to(device)

    print(f"model params: {sum(p.numel() for p in denoising_model.parameters()) / 1e6:.2f} M")
    return denoising_model