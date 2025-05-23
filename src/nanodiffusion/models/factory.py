from nanodiffusion.models.unets import UNet, UNetBig, UNetSmall
from nanodiffusion.models.dit import DiT
from nanodiffusion.models.tld import Denoiser as TLD

def choices():
    return [
        "tld_t2", "tld_s2", "tld_b2", "dit_t0", "dit_t1", "dit_t2", "dit_t3",
        "dit_s2", "dit_b2", "dit_b4", "dit_l2", "dit_l4",
        "unet_small", "unet", "unet_big", "unet_diffusers"
    ]

def create_model(net: str = "unet", resolution: int = 32, in_channels: int = 3, cond_embed_dim: int = None):
    print(f"Creating model {net} with resolution {resolution} and in_channels {in_channels} and cond_embed_dim {cond_embed_dim}")
    if net == "tld_t2":
        return TLD(
            image_size=resolution,
            embed_dim=128,
            noise_embed_dims=256,
            patch_size=2,
            n_layers=3,
            mlp_multiplier=2,
            n_channels=in_channels,
            dropout=0,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "tld_s2":
        return TLD(
            image_size=resolution,
            embed_dim=128,
            noise_embed_dims=256,
            patch_size=2,
            n_layers=12,
            mlp_multiplier=4,
            n_channels=in_channels,
            dropout=0,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "tld_b2":
        return TLD(
            image_size=resolution,
            embed_dim=768,
            noise_embed_dims=256,
            patch_size=2,
            n_layers=12,
            mlp_multiplier=4,
            n_channels=in_channels,
            dropout=0,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "dit_t0":
        return DiT(
            input_size=resolution,
            patch_size=2,
            in_channels=in_channels,
            learn_sigma=False,
            hidden_size=32,
            mlp_ratio=2,
            depth=3,
            num_heads=1,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "dit_t1":
        return DiT(
            input_size=resolution,
            patch_size=2,
            in_channels=in_channels,
            learn_sigma=False,
            hidden_size=32 * 6,
            mlp_ratio=2,
            depth=3,
            num_heads=6,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "dit_t2":
        return DiT(
            input_size=resolution,
            patch_size=2,
            in_channels=in_channels,
            learn_sigma=False,
            hidden_size=32,
            mlp_ratio=2,
            depth=12,
            num_heads=1,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "dit_t3":
        model = DiT(
                input_size=resolution,
                patch_size=2,
                in_channels=in_channels,
                learn_sigma=False,
                hidden_size=32 * 6,
                mlp_ratio=2,
                depth=12,
                num_heads=6,
                cond_embed_dim=cond_embed_dim,
            )
    elif net == "dit_s2":
        model = DiT(
            input_size=resolution,
            depth=12,
            in_channels=in_channels,
            hidden_size=384,
            patch_size=2,
            num_heads=6,
            learn_sigma=False,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "dit_b2":
        model = DiT(
            input_size=resolution,
            depth=12,
            in_channels=in_channels,
            hidden_size=768,
            patch_size=2,
            num_heads=12,
            learn_sigma=False,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "dit_b4":
        model = DiT(
            input_size=resolution,
            depth=12,
            in_channels=in_channels,
            hidden_size=768,
            patch_size=4,
            num_heads=12,
            learn_sigma=False,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "dit_l2":
        model = DiT(
            input_size=resolution,
            depth=24,
            in_channels=in_channels,
            hidden_size=1024,
            patch_size=2,
            num_heads=16,
            learn_sigma=False,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "dit_l4":
        model = DiT(
            input_size=resolution,
            depth=24,
            in_channels=in_channels,
            hidden_size=1024,
            patch_size=4,
            num_heads=16,
            learn_sigma=False,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "unet_small":
        model = UNetSmall(
            image_size=resolution,
            in_channels=in_channels,
            out_channels=in_channels,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "unet":
        model = UNet(
            image_size=resolution,
            in_channels=in_channels,
            out_channels=in_channels,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "unet_big":
        model = UNetBig(
            image_size=resolution,
            in_channels=in_channels,
            out_channels=in_channels,
            cond_embed_dim=cond_embed_dim,
        )
    elif net == "unet_diffusers":
        if cond_embed_dim is not None:
            raise ValueError("diffusers UNet does not support conditional models")
        
        from diffusers import UNet2DModel

        model = UNet2DModel(
            sample_size=resolution,
            in_channels=in_channels,
            out_channels=in_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    else:
        raise ValueError(f"Unsupported network architecture: {net}")
    
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    return model