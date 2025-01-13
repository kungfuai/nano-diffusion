from src.models.unets import UNet, UNetBig, UNetSmall

from src.models.dit import DiT

# from src.models.dit_cond import DiT


def create_model(net: str = "unet", resolution: int = 32, in_channels: int = 3):
    if net == "dit_t0":
        return DiT(
            input_size=32,
            patch_size=2,
            in_channels=in_channels,
            learn_sigma=False,
            hidden_size=32,
            mlp_ratio=2,
            depth=3,
            num_heads=1,
            class_dropout_prob=0.1,
        )
    elif net == "dit_t1":
        return DiT(
            input_size=32,
            patch_size=2,
            in_channels=in_channels,
            learn_sigma=False,
            hidden_size=32 * 6,
            mlp_ratio=2,
            depth=3,
            num_heads=6,
            class_dropout_prob=0.1,
        )
    elif net == "dit_t2":
        return DiT(
            input_size=32,
            patch_size=2,
            in_channels=in_channels,
            learn_sigma=False,
            hidden_size=32,
            mlp_ratio=2,
            depth=12,
            num_heads=1,
            class_dropout_prob=0.1,
        )
    elif net == "dit_t3":
        model = DiT(
            input_size=32,
            patch_size=2,
            in_channels=in_channels,
            learn_sigma=False,
            hidden_size=32 * 6,
            mlp_ratio=2,
            depth=12,
            num_heads=6,
            class_dropout_prob=0.1,
        )
    elif net == "dit_s2":
        model = DiT(
            depth=12,
            in_channels=in_channels,
            hidden_size=384,
            patch_size=2,
            num_heads=6,
            learn_sigma=False,
        )
    elif net == "dit_b2":
        model = DiT(
            depth=12,
            in_channels=in_channels,
            hidden_size=384,
            patch_size=2,
            num_heads=6,
            learn_sigma=False,
        )
    elif net == "dit_b4":
        model = DiT(
            depth=12,
            in_channels=in_channels,
            hidden_size=384,
            patch_size=4,
            num_heads=6,
            learn_sigma=False,
        )
    elif net == "dit_l2":
        model = DiT(
            depth=12,
            in_channels=in_channels,
            hidden_size=768,
            patch_size=2,
            num_heads=12,
            learn_sigma=False,
        )
    elif net == "dit_l4":
        model = DiT(
            depth=12,
            in_channels=in_channels,
            hidden_size=768,
            patch_size=4,
            num_heads=12,
            learn_sigma=False,
        )
    elif net == "unet_small":
        model = UNetSmall(
            image_size=resolution,
        )
    elif net == "unet":
        model = UNet(
            image_size=resolution,
        )
    elif net == "unet_big":
        model = UNetBig(
            image_size=resolution,
        )
    elif net == "unet_diffusers":
        from diffusers import UNet2DModel

        model = UNet2DModel(
            sample_size=resolution,
            in_channels=3,
            out_channels=3,
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
