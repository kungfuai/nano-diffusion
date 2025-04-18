"""
Linear Variational Diffusion Model

Adapted from https://github.com/apapiu/transformer_latent_diffusion.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy
from dataclasses import dataclass, asdict
import os
from typing import Optional, Callable, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
import wandb
import clip
from accelerate import Accelerator
from diffusers import AutoencoderKL
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.utils.data import random_split
from torchvision import transforms

from src.models.factory import create_model, choices
from src.datasets import load_data
from src.datasets.mj_latents import MJLatentsDataset, collate_fn
from src.datasets.hugging_face_dataset import HuggingFaceDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_pil = transforms.ToPILImage()


@dataclass
class TrainingConfig:
    # Dataset
    dataset: str = "zzsi/afhq64_16k_latents_sdxl_blip2" # dataset name
    resolution: int = 8 # resolution of the image
    
    # Model architecture
    in_channels: int = 3 # number of input channels
    net: str = "unet_small" # network architecture
    num_denoising_steps: int = 100 # number of timesteps
    
    # Flow plan algorithm
    plan: str = "ot"  # flow matching plan

    # Training loop and optimizer
    total_steps: int = 100000  # total number of training steps
    batch_size: int = 128 # batch size
    learning_rate: float = 1e-4 # initial learning rate
    weight_decay: float = 0.0 # weight decay
    lr_min: float = 1e-6 # minimum learning rate
    warmup_steps: int = 1000 # number of warmup steps

    # Logging and evaluation
    log_every: int = 100 # log every N steps
    sample_every: int = 1000 # sample every N steps
    save_every: int = 50000 # save model every N steps
    validate_every: int = 10000 # compute validation loss every N steps
    fid_every: int = 10000 # compute FID every N steps
    num_samples_for_fid: int = 1000 # number of samples for FID
    num_samples_for_logging: int = 8 # number of samples for logging
    num_real_samples_for_fid: int = 10000 # number of real image samples for FID precomputation

    # Regularization
    max_grad_norm: float = -1 # maximum norm for gradient clipping
    use_loss_mean: bool = False # use loss.mean() instead of just loss
    use_ema: bool = False # use EMA for the model
    ema_beta: float = 0.9999 # EMA decay factor

    # Accelerator
    device: str = "cuda" # device to use for training

    # Logging
    logger: str = "wandb" # logging method
    cache_dir: str = f"{os.path.expanduser('~')}/.cache" # cache directory in the home directory, same across runs
    checkpoint_dir: str = "logs/train" # checkpoint directory, different for each run
    min_steps_for_final_save: int = 100 # minimum steps for final save
    watch_model: bool = False # watch the model with wandb
    init_from_wandb_run_path: str = None # initialize model from a wandb run path
    init_from_wandb_file: str = None # initialize model from a wandb file

    # Data augmentation
    random_flip: bool = False # randomly flip images horizontally

    # VAE and conditioning
    vae_model_name: str = "madebyollin/sdxl-vae-fp16-fix"  # VAE model name
    vae_scale_factor: float = 0.18215  # scale factor for the VAE encoding outputs (so that the std is close to 1)
    cond_embed_dim: int = 768  # dimension of the conditioning embedding (before the projection layer)
    cond_drop_prob: float = 0.2  # probability of dropping conditioning during training
    guidance_scale: float = 4.5  # guidance scale for classifier-free guidance


    def update_checkpoint_dir(self):
        # Update the checkpoint directory use a timestamp
        self.checkpoint_dir = f"{self.checkpoint_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    def __post_init__(self):
        self.update_checkpoint_dir()


@dataclass
class DenoiserConfig:
    image_size: int = 32
    # image_size: int = 8
    noise_embed_dims: int = 256
    patch_size: int = 2
    embed_dim: int = 768 # 128
    dropout: float = 0
    n_layers: int = 12
    text_emb_size: int = 768
    n_channels: int = 4
    mlp_multiplier: int = 4 

@dataclass
class DenoiserLoad:
    dtype: torch.dtype = torch.float32
    file_url: str | None = None
    local_filename: str | None = None

@dataclass
class VaeConfig:
    vae_scale_factor: float = 8
    vae_name: str = "madebyollin/sdxl-vae-fp16-fix"
    vae_dtype: torch.dtype = torch.float32

@dataclass
class ClipConfig:
    clip_model_name: str = "ViT-L/14"
    clip_dtype: torch.dtype = torch.float16

@dataclass
class DataConfig:
    """where is the latent data stored"""
    latent_path: str  
    text_emb_path: str
    val_path: str

@dataclass
class TrainConfig:
    batch_size: int = 256 
    lr: float = 3e-4
    n_epoch: int = 100
    alpha: float = 0.999
    from_scratch: bool = True
    ##betas determine the distribution of noise seen during training
    beta_a: float = 1  
    beta_b: float = 2.5
    save_and_eval_every_iters: int = 5000
    run_id: str = "mj_101M"
    model_name: str = "mj_101M.pt"
    compile: bool = True
    use_ema: bool = True
    save_model: bool = True
    use_wandb: bool = True
    checkpoint_dir: str = "logs/checkpoints"

    def update_checkpoint_dir(self):
        """ Update the checkpoint directory use a timestamp
        """
        from datetime import datetime
        
        self.checkpoint_dir = f"{self.checkpoint_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
    def __post_init__(self):
        self.update_checkpoint_dir()


@dataclass
class LTDConfig:
    """main config for inference"""
    denoiser_cfg: DenoiserConfig = field(default_factory=DenoiserConfig)
    denoiser_load: DenoiserLoad = field(default_factory=DenoiserLoad)
    vae_cfg: VaeConfig = field(default_factory=VaeConfig)
    clip_cfg: ClipConfig = field(default_factory=ClipConfig)


@dataclass
class ModelConfig:
    """main config for getting data, training and inference"""
    data_config: DataConfig
    denoiser_config: DenoiserConfig = field(default_factory=DenoiserConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    vae_cfg: VaeConfig = field(default_factory=VaeConfig)
    clip_cfg: ClipConfig = field(default_factory=ClipConfig)
    net: str = "tld_b2"

@dataclass
class ModelComponents:
    denoising_model: nn.Module
    ema_model: nn.Module | None
    vae: AutoencoderKL
    optimizer: torch.optim.Optimizer
    diffuser: "DiffusionGenerator"
    ema_diffuser: "DiffusionGenerator"


def forward_diffusion(x_0: Tensor, t: Tensor, beta_a: float, beta_b: float, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Apply forward diffusion process to add noise to images."""
    if noise is None:
        noise = torch.randn_like(x_0)
    
    noise_level = torch.tensor(
        np.random.beta(beta_a, beta_b, len(x_0)), 
        device=x_0.device
    )
    signal_level = 1 - noise_level
    x_noisy = noise_level.view(-1, 1, 1, 1) * noise + signal_level.view(-1, 1, 1, 1) * x_0
    
    return x_noisy.float(), noise


class SinusoidalEmbedding(nn.Module):
    def __init__(self, emb_min_freq=1.0, emb_max_freq=1000.0, embedding_dims=32):
        super(SinusoidalEmbedding, self).__init__()

        frequencies = torch.exp(
            torch.linspace(np.log(emb_min_freq), np.log(emb_max_freq), embedding_dims // 2)
        )

        self.register_buffer("angular_speeds", 2.0 * torch.pi * frequencies)

    def forward(self, x):
        embeddings = torch.cat(
            [torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1
        )
        return embeddings


class MHAttention(nn.Module):
    def __init__(self, is_causal=False, dropout_level=0.0, n_heads=4):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):
        assert q.size(-1) == k.size(-1)
        assert k.size(-2) == v.size(-2)

        q, k, v = [rearrange(x, "bs n (h d) -> bs h n d", h=self.n_heads) for x in [q, k, v]]

        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.is_causal,
            dropout_p=self.dropout_level if self.training else 0,
        )

        out = rearrange(out, "bs h n d -> bs n (h d)", h=self.n_heads)

        return out


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0.0, n_heads=4):
        super().__init__()
        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x):
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        return self.mha(q, k, v)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.kv_linear = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x, y):
        q = self.q_linear(x)
        k, v = self.kv_linear(y).chunk(2, dim=2)
        return self.mha(q, k, v)


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPSepConv(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        """see: https://github.com/ofsoundof/LocalViT"""
        super().__init__()
        self.mlp = nn.Sequential(
            # this Conv with kernel size 1 is equivalent to the Linear layer in a "regular" transformer MLP
            nn.Conv2d(embed_dim, mlp_multiplier * embed_dim, kernel_size=1, padding="same"),
            nn.Conv2d(
                mlp_multiplier * embed_dim,
                mlp_multiplier * embed_dim,
                kernel_size=3,
                padding="same",
                groups=mlp_multiplier * embed_dim,
            ),  # <- depthwise conv
            nn.GELU(),
            nn.Conv2d(mlp_multiplier * embed_dim, embed_dim, kernel_size=1, padding="same"),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        w = h = int(np.sqrt(x.size(1)))  # only square images for now
        x = rearrange(x, "bs (h w) d -> bs d h w", h=h, w=w)
        x = self.mlp(x)
        x = rearrange(x, "bs d h w -> bs (h w) d")
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        is_causal: bool,
        mlp_multiplier: int,
        dropout_level: float,
        mlp_class: type[MLP] | type[MLPSepConv],
    ):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim, is_causal, dropout_level, n_heads=embed_dim // 64)
        self.cross_attention = CrossAttention(
            embed_dim, is_causal=False, dropout_level=0, n_heads=embed_dim // 64
        )
        self.mlp = mlp_class(embed_dim, mlp_multiplier, dropout_level)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.self_attention(self.norm1(x)) + x
        x = self.cross_attention(self.norm2(x), y) + x
        x = self.mlp(self.norm3(x)) + x
        return x

class DenoiserTransBlock(nn.Module):
    def __init__(
        self,
        patch_size: int,
        img_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        mlp_multiplier: int = 4,
        n_channels: int = 4,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        seq_len = int((self.img_size / self.patch_size) * (self.img_size / self.patch_size))
        patch_dim = self.n_channels * self.patch_size * self.patch_size

        self.patchify_and_embed = nn.Sequential(
            nn.Conv2d(
                self.n_channels,
                patch_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("bs d h w -> bs (h w) d"),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.rearrange2 = Rearrange(
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=int(self.img_size / self.patch_size),
            p1=self.patch_size,
            p2=self.patch_size,
        )

        self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer("precomputed_pos_enc", torch.arange(0, seq_len).long())

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=self.embed_dim,
                    mlp_multiplier=self.mlp_multiplier,
                    # note that this is a non-causal block since we are
                    # denoising the entire image no need for masking
                    is_causal=False,
                    dropout_level=self.dropout,
                    mlp_class=MLPSepConv,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.out_proj = nn.Sequential(nn.Linear(self.embed_dim, patch_dim), self.rearrange2)

    def forward(self, x, cond):
        x = self.patchify_and_embed(x)
        pos_enc = self.precomputed_pos_enc[: x.size(1)].expand(x.size(0), -1)
        x = x + self.pos_embed(pos_enc)

        for block in self.decoder_blocks:
            x = block(x, cond)

        return self.out_proj(x)


class Denoiser(nn.Module):
    def __init__(
        self,
        image_size: int,
        noise_embed_dims: int,
        patch_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        text_emb_size: int = 768,
        mlp_multiplier: int = 4,
        n_channels: int = 4
    ):
        super().__init__()

        self.image_size = image_size
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim
        self.n_channels = n_channels

        self.fourier_feats = nn.Sequential(
            SinusoidalEmbedding(embedding_dims=noise_embed_dims),
            nn.Linear(noise_embed_dims, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.denoiser_trans_block = DenoiserTransBlock(patch_size, image_size, embed_dim, dropout, n_layers, mlp_multiplier, n_channels)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.label_proj = nn.Linear(text_emb_size, self.embed_dim)

    def forward(self, x, noise_level, label):
        noise_level = self.fourier_feats(noise_level).unsqueeze(1)

        label = self.label_proj(label).unsqueeze(1)

        noise_label_emb = torch.cat([noise_level, label], dim=1)  # bs, 2, d
        noise_label_emb = self.norm(noise_label_emb)

        x = self.denoiser_trans_block(x, noise_label_emb)

        return x
    
    from dataclasses import dataclass, asdict



@dataclass
class DiffusionGenerator:
    model: Denoiser
    vae: AutoencoderKL
    device: torch.device
    model_dtype: torch.dtype = torch.float32

    @torch.no_grad()
    def generate(
        self,
        labels: Tensor,  # embeddings to condition on
        n_iter: int = 30,
        num_imgs: int = 16,
        class_guidance: float = 3,
        seed: int = 10,
        scale_factor: int = 8,  # latent scaling before decoding - should be ~ std of latent space
        img_size: int = 32,  # height, width of latent
        sharp_f: float = 0.1,
        bright_f: float = 0.1,
        exponent: float = 1,
        seeds: Tensor | None = None,
        noise_levels=None,
        use_ddpm_plus: bool = True,
    ):
        """Generate images via reverse diffusion.
        if use_ddpm_plus=True uses Algorithm 2 DPM-Solver++(2M) here: https://arxiv.org/pdf/2211.01095.pdf
        else use ddim with alpha = 1-sigma
        """
        if noise_levels is None:
            noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
        noise_levels[0] = 0.99

        if use_ddpm_plus:
            lambdas = [np.log((1 - sigma) / sigma) for sigma in noise_levels]  # log snr
            hs = [lambdas[i] - lambdas[i - 1] for i in range(1, len(lambdas))]
            rs = [hs[i - 1] / hs[i] for i in range(1, len(hs))]

        x_t = self.initialize_image(seeds, num_imgs, img_size, seed)

        labels = torch.cat([labels, torch.zeros_like(labels)])
        self.model.eval()

        x0_pred_prev = None

        for i in tqdm(range(len(noise_levels) - 1)):
            curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]

            x0_pred = self.pred_image(x_t, labels, curr_noise, class_guidance)

            if x0_pred_prev is None:
                x_t = ((curr_noise - next_noise) * x0_pred + next_noise * x_t) / curr_noise
            else:
                if use_ddpm_plus:
                    # x0_pred is a combination of the two previous x0_pred:
                    D = (1 + 1 / (2 * rs[i - 1])) * x0_pred - (1 / (2 * rs[i - 1])) * x0_pred_prev
                else:
                    # ddim:
                    D = x0_pred

                x_t = ((curr_noise - next_noise) * D + next_noise * x_t) / curr_noise

            x0_pred_prev = x0_pred

        x0_pred = self.pred_image(x_t, labels, next_noise, class_guidance)

        # shifting latents works a bit like an image editor:
        x0_pred[:, 3, :, :] += sharp_f
        x0_pred[:, 0, :, :] += bright_f

        x0_pred_img = self.vae.decode((x0_pred * scale_factor).to(self.model_dtype))[0].cpu()
        return x0_pred_img, x0_pred

    def pred_image(self, noisy_image, labels, noise_level, class_guidance):
        num_imgs = noisy_image.size(0)
        noises = torch.full((2 * num_imgs, ), noise_level)
        # x0_pred = self.model(
        #     torch.cat([noisy_image, noisy_image]),
        #     noises.to(self.device, self.model_dtype),
        #     labels.to(self.device, self.model_dtype),
        # )
        x0_pred= self.model(
            t=noises.to(self.device, self.model_dtype),
            x=torch.cat([noisy_image, noisy_image]),
            y=labels.to(self.device, self.model_dtype),
            p_uncond=0,
        )
        x0_pred = self.apply_classifier_free_guidance(x0_pred, num_imgs, class_guidance)
        return x0_pred

    def initialize_image(self, seeds, num_imgs, img_size, seed):
        """Initialize the seed tensor."""
        if seeds is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            return torch.randn(
                num_imgs,
                4,  # TODO: pass this from config
                img_size,
                img_size,
                dtype=self.model_dtype,
                device=self.device,
                generator=generator,
            )
        else:
            return seeds.to(self.device, self.model_dtype)

    def apply_classifier_free_guidance(self, x0_pred, num_imgs, class_guidance):
        """Apply classifier-free guidance to the predictions."""
        x0_pred_label, x0_pred_no_label = x0_pred[:num_imgs], x0_pred[num_imgs:]
        return class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label


def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


@torch.no_grad()
def encode_text(label, model):
    text_tokens = clip.tokenize(label, truncate=True).to(device)
    text_encoding = model.encode_text(text_tokens)
    return text_encoding.cpu()


class DiffusionTransformer:
    def __init__(self, cfg: LTDConfig):
        denoiser = Denoiser(**asdict(cfg.denoiser_cfg))
        denoiser = denoiser.to(cfg.denoiser_load.dtype)

        if cfg.denoiser_load.file_url is not None:
            if cfg.denoiser_load.local_filename is not None:
                print(f"Downloading model from {cfg.denoiser_load.file_url}")
                download_file(cfg.denoiser_load.file_url, cfg.denoiser_load.local_filename)
                state_dict = torch.load(cfg.denoiser_load.local_filename, map_location=torch.device("cpu"))
                denoiser.load_state_dict(state_dict)

        denoiser = denoiser.to(device)

        vae = AutoencoderKL.from_pretrained(cfg.vae_cfg.vae_name, 
        torch_dtype=cfg.vae_cfg.vae_dtype).to(device)

        self.clip_model, preprocess = clip.load(cfg.clip_cfg.clip_model_name)
        self.clip_model = self.clip_model.to(device)

        self.diffuser = DiffusionGenerator(denoiser, vae, device, cfg.denoiser_load.dtype)

    def generate_image_from_text(
        self, prompt: str, class_guidance=6, seed=11, num_imgs=1, img_size=32, n_iter=15
    ):
        nrow = int(np.sqrt(num_imgs))

        cur_prompts = [prompt] * num_imgs
        labels = encode_text(cur_prompts, self.clip_model)
        out, out_latent = self.diffuser.generate(
            labels=labels,
            num_imgs=num_imgs,
            img_size=self.diffuser.model.image_size,
            class_guidance=class_guidance,
            seed=seed,
            n_iter=n_iter,
            exponent=1,
            scale_factor=8,
            sharp_f=0,
            bright_f=0,
        )

        out = to_pil((vutils.make_grid((out + 1) / 2, nrow=nrow, padding=4)).float().clip(0, 1))
        return out

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_per_layer(model: nn.Module):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")


def update_ema(ema_model: nn.Module, model: nn.Module, alpha: float = 0.999):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)


def train_step(
    accelerator: Accelerator,
    model: nn.Module,
    x_0: Tensor,
    label: Tensor,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    device: torch.device,
    criterion: nn.Module,
) -> Tensor:
    """Execute single training step."""    
    
    noise = torch.randn_like(x_0)
    noise_level = torch.tensor(
        np.random.beta(config.train_config.beta_a, config.train_config.beta_b, len(x_0)), device=accelerator.device
    ).float()
    t = noise_level
    # t = torch.randint(0, config.denoiser_config.n_layers, (x_0.shape[0],)).to(device)
    x_t, _ = forward_diffusion(x_0, t, config.train_config.beta_a, config.train_config.beta_b, noise=noise)

    # Random label masking
    # prob = 0.15
    # mask = torch.rand(label.size(0), device=device) < prob
    # label[mask] = 0

    # predicted_noise = model(t=t, x=x_t, label=label)
    # print(f"t: {t.shape}, x_t: {x_t.shape}, label: {label.shape}")
    with accelerator.accumulate():
        optimizer.zero_grad()
        # predicted_data = model(x_t, t.view(-1, 1), label=label)
        predicted_data = model(x=x_t, t=t, y=label, p_uncond=0.15)
        if hasattr(predicted_data, "sample"):
            predicted_data = predicted_data.sample

        loss = criterion(predicted_data, x_0)
        accelerator.backward(loss)
        optimizer.step()

    return loss


def create_model_components(config: ModelConfig, device: torch.device) -> ModelComponents:
    """Initialize all model components."""
    # denoiser = Denoiser(**asdict(config.denoiser_config))
    denoiser = create_model(net=config.net, resolution=32, in_channels=4, cond_embed_dim=768)
    denoiser = denoiser.to(device)
    
    ema_model = copy.deepcopy(denoiser) if config.train_config.use_ema else None
    
    vae = AutoencoderKL.from_pretrained(
        config.vae_cfg.vae_name,
        torch_dtype=config.vae_cfg.vae_dtype
    ).to(device)
    
    optimizer = torch.optim.Adam(
        denoiser.parameters(),
        lr=config.train_config.lr
    )

    diffuser = DiffusionGenerator(denoiser, vae, device, torch.float32)
    ema_diffuser = DiffusionGenerator(ema_model, vae, device, torch.float32)
    return ModelComponents(denoiser, ema_model, vae, optimizer, diffuser, ema_diffuser)


def save_model(model: nn.Module, path: str | Path, use_wandb: bool = False):
    """Save model checkpoint."""
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")
    # if use_wandb:
    #     wandb.save(str(path))

def save_and_evaluate(
    model_components: ModelComponents,
    config: ModelConfig,
    step: int,
    val_dataloader: DataLoader,
    device: torch.device,
):
    """Save checkpoints and generate evaluation samples."""
    # Generate and save samples
    diffuser = model_components.diffuser
    ema_diffuser = model_components.ema_diffuser
    out, _ = diffuser.generate(
        labels=emb_val, # torch.repeat_interleave(labels, 2, dim=0),
        num_imgs=len(emb_val),
        class_guidance=4.5,
        seed=10,
        n_iter=40,
        exponent=1,
        sharp_f=0.1,
        img_size=config.denoiser_config.image_size,
    )
    out = ((out + 1) / 2).clamp(0, 1)
    images_processed = (out * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")
    # out = to_pil(vutils.make_grid((out + 1) / 2, nrow=8, padding=4).float().clip(0, 1))
    # Path(config.train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # out.save(Path(config.train_config.checkpoint_dir) / "img.jpg")
    
    if config.train_config.use_wandb:
        wandb.log({
            "step": step,
            # "test_samples_step": step,
            "test_samples": [wandb.Image(img) for img in images_processed],
        })

    ema_out, _ = ema_diffuser.generate(
        labels=emb_val, # torch.repeat_interleave(labels, 2, dim=0),
        num_imgs=len(emb_val),
        class_guidance=4.5,
        seed=10,
        n_iter=40,
        exponent=1,
        sharp_f=0.1,
        img_size=config.denoiser_config.image_size,
    )
    ema_out = ((ema_out + 1) / 2).clamp(0, 1)
    images_processed = (ema_out * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")
    if config.train_config.use_wandb:
        wandb.log({
            "step": step,
            "ema_samples": [wandb.Image(img) for img in images_processed],
        })

    # Save model checkpoints
    if config.train_config.save_model:
        save_model(
            model_components.denoising_model,
            Path(config.train_config.checkpoint_dir) / f"model_{step}.pth",
            config.train_config.use_wandb
        )

def training_loop(
    model_components: ModelComponents,
    train_dataloader: DataLoader,
    config: ModelConfig,
    device: torch.device,
    accelerator: Accelerator,
) -> int:
    """Main training loop."""
    
    model = model_components.denoising_model
    ema_model = model_components.ema_model
    optimizer = model_components.optimizer
    
    if config.train_config.compile:
        model = torch.compile(model)
    
    model, train_dataloader, optimizer = accelerator.prepare(
        model, train_dataloader, optimizer
    )
    
    if config.train_config.use_wandb:
        accelerator.init_trackers(
            project_name="nano-diffusion",
            config=asdict(config)
        )
    
    criterion = nn.MSELoss()
    global_step = 0
    num_examples_trained = 0
    
    for epoch in range(config.train_config.n_epoch):
        accelerator.print(f"epoch: {epoch + 1}")
        
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            x = batch['image_emb']
            y = batch['text_emb']
            num_examples_trained += x.shape[0]
            x = x.float() / config.vae_cfg.vae_scale_factor
            
            loss = train_step(
                accelerator,
                model, x, y.float(),
                optimizer, config, device, criterion
            )
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            if global_step % config.train_config.save_and_eval_every_iters == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_and_evaluate(
                        model_components, config, global_step,
                        None, device
                    )
            
            if config.train_config.use_ema and accelerator.is_main_process:
                update_ema(ema_model, model, config.train_config.alpha)
            
            if global_step % 100 == 0:
                accelerator.log({"train_loss": loss.item()}, step=global_step)
            
            global_step += 1
    
    accelerator.end_training()
    return num_examples_trained


def load_data(config: TrainingConfig, collate_fn: Optional[Callable] = None) -> Tuple[DataLoader, DataLoader]:
    # TODO: consider expanding the args from config to be a more explicit list of args
    resolution = config.resolution
    transforms_list = [
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    if config.random_flip:
        transforms_list.insert(0, transforms.RandomHorizontalFlip())

    transform = transforms.Compose(transforms_list)

    if config.dataset == "mj_latents":
        print("Loading MJ latents dataset")
        full_dataset = MJLatentsDataset()
    else:
        print(f"Loading dataset from Hugging Face: {config.dataset}")
        full_dataset = HuggingFaceDataset(config.dataset, transform=transform)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--logger", type=str, default=None)
    parser.add_argument("--net", type=str, default="tld_b2", choices=choices())
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    data_config = DataConfig(
        # latent_path="latent_folder/afhq64_16k_image_emb.npy",
        # text_emb_path="latent_folder/afhq64_16k_text_emb.npy",
        # val_path="latent_folder/afhq64_16k_val_text_emb.npy",
        #######
        latent_path="data/raw/mj_latents.npy",
        text_emb_path="data/raw/mj_text_emb.npy",
        val_path=None,
    )

    config = ModelConfig(
        net=args.net,
        data_config=data_config,
        train_config=TrainConfig(n_epoch=500, save_model=False, compile=False, use_wandb=args.logger == "wandb"),
    )

    denoiser_config = config.denoiser_config
    train_config = config.train_config
    dataconfig = config.data_config

    log_with="wandb" if train_config.use_wandb else None
    accelerator = Accelerator(
        mixed_precision="fp16" if args.fp16 else "no",
        log_with=log_with
    )

    accelerator.print("Loading Data:")
    new_config = TrainingConfig(dataset="mj_latents", batch_size=256, resolution=32)
    train_dataloader, val_dataloader = load_data(new_config, collate_fn=collate_fn)
    emb_val = next(iter(val_dataloader))['text_emb'][:8]
    # latent_train_data = torch.tensor(np.load(dataconfig.latent_path), dtype=torch.float32)
    # train_label_embeddings = torch.tensor(np.load(dataconfig.text_emb_path), dtype=torch.float32)
    # val_offset = int(len(train_label_embeddings) * 0.8)
    # emb_val = train_label_embeddings[val_offset:val_offset+8]
    # dataset = TensorDataset(latent_train_data, train_label_embeddings)
    # train_loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)

    model_components = create_model_components(config, device)

    num_examples_trained = training_loop(
        model_components, train_dataloader, config, device, accelerator
    )
    print(f"Training completed. Total examples trained: {num_examples_trained}")
