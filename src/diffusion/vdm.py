"""
DDPM is a diffusion training and sampling algorithm.
"""
from typing import Dict, Callable, List
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from .base import BaseDiffusionAlgorithm
from ..bookkeeping.mini_batch import MiniBatch
from ..config.diffusion_training_config import DiffusionTrainingConfig
from ..diffusion.noise_scheduler import generate_samples_by_denoising


class VDMForwardDiffusion:
    def __init__(self, beta_a: float, beta_b: float):
        self.beta_a = beta_a
        self.beta_b = beta_b

    def compute(self, x_0, t=None, noise=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion process to add noise to images.

        The non-standard diffusion parameterization used in Variational Diffusion Models (VDM)—sampling noise levels and
        signal levels directly from a Beta distribution instead of using the standard cumulative product of alpha_t.
        The forward diffusion process is both simpler and more flexible than DDPM.

        Args:
            x_0: The input images to add noise to.
            t: The noise levels to add to the images.
            noise: The noise to add to the images. If not provided, random noise is added.

        Returns:
            A tuple containing the noise levels (time steps), the noisy images and the noise added to the images.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        if t is None:
            t = torch.tensor(
                np.random.beta(self.beta_a, self.beta_b, len(x_0)), 
                device=x_0.device
            )
        noise_level = t
        signal_level = 1 - noise_level
        x_noisy = noise_level.view(-1, 1, 1, 1) * noise + signal_level.view(-1, 1, 1, 1) * x_0
        
        return noise_level, x_noisy.float(), noise
    
    def __call__(self, x_0, t=None, noise=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.compute(x_0, t, noise)


class VDMTrainingExampleGenerator:
    def __init__(self, target_type: str, num_denoising_steps: int, device: str | torch.device):
        self.target_type = target_type
        self.num_denoising_steps = num_denoising_steps
        self.device = device

    def generate(self, x_0, forward_diffusion, y, p_uncond) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        noise = torch.randn_like(x_0)
        t, x_t, noise = forward_diffusion.compute(x_0, noise=noise)
        if self.target_type == "noise":
            targets = noise
        elif self.target_type == "x_0":
            targets = x_0
        else:
            raise ValueError("Unsupported target_type.")
        inputs = {"x": x_t, "t": t}
        if y is not None:
            inputs["y"] = y
            inputs["p_uncond"] = p_uncond
        return inputs, targets


class VDMSampler:
    """
    DDPM sampling algorithm for generating samples (e.g. images) using a denoising model.
    """
    def __init__(self,
            denoising_model: nn.Module,
            num_denoising_steps: int = 1000,
            decoder: nn.Module | Callable = None,
            vae_scale_multiplier: float = 1.0,
            use_ddpm_plus: bool = True,
            model_dtype: torch.dtype = torch.float32,
            seed=0,
        ):
        self.denoising_model = denoising_model
        self.num_denoising_steps = num_denoising_steps
        self.decoder = decoder
        self.vae_scale_multiplier = vae_scale_multiplier
        self.model_dtype = model_dtype
        self.seed = seed
        self.use_ddpm_plus = use_ddpm_plus

    def sample(self,
            x_T,
            y = None,
            guidance_scale: float = 0,
            exponent: float = 1,
            sharp_f: float = 0,
            bright_f: float = 0,
            noise_levels: List[float] = None,
            seed: int = None,
            quiet: bool = None,
        ):
        """
        Generate samples from the denoising model.

        Generate images via reverse diffusion.
        if use_ddpm_plus=True uses Algorithm 2 DPM-Solver++(2M) here: https://arxiv.org/pdf/2211.01095.pdf
        else use ddim with alpha = 1-sigma

        Args:
            x_T: The initial noise tensor. This provides initialization and the shape of the generated output.
            y: Embeddings of the prompt or conditioning information. It can be text embeddings.
              Only applicable to conditional generation.
            guidance_scale: The scale for the guidance mechanism.
              Only applicalbe to conditional generation.
            seed: The seed for the random number generator.
            quiet: Whether to suppress the progress bar.
        
        Returns:
            The generated images.
        """
        assert x_T is not None, "x_T must be provided"
        n_iter = self.num_denoising_steps
        if noise_levels is None:
            noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
        noise_levels[0] = 0.99

        if self.use_ddpm_plus:
            lambdas = [np.log((1 - sigma) / sigma) for sigma in noise_levels]  # log snr
            hs = [lambdas[i] - lambdas[i - 1] for i in range(1, len(lambdas))]
            rs = [hs[i - 1] / hs[i] for i in range(1, len(hs))]

        x_t = x_T

        if y is not None:
            if hasattr(self.denoising_model, "get_null_cond_embed"):
                null_embeddings = self.denoising_model.get_null_cond_embed(batch_size=len(y))
                null_embeddings = null_embeddings.to(y.device, y.dtype)
            else:
                null_embeddings = torch.zeros_like(y)
            y = torch.cat([y, null_embeddings])
        x0_pred_prev = None

        if quiet:
            pbar = tqdm(range(len(noise_levels) - 1))
        else:
            pbar = range(len(noise_levels) - 1)
        for i in pbar:
            curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]
            x0_pred = self.pred_image(noisy_image=x_t, y=y, noise_level=curr_noise, guidance_scale=guidance_scale)

            if x0_pred_prev is None:
                x_t = ((curr_noise - next_noise) * x0_pred + next_noise * x_t) / curr_noise
            else:
                if self.use_ddpm_plus:
                    # x0_pred is a combination of the two previous x0_pred:
                    D = (1 + 1 / (2 * rs[i - 1])) * x0_pred - (1 / (2 * rs[i - 1])) * x0_pred_prev
                else:
                    # ddim:
                    D = x0_pred

                x_t = ((curr_noise - next_noise) * D + next_noise * x_t) / curr_noise

            x0_pred_prev = x0_pred

        x0_pred = self.pred_image(x_t, y, next_noise, guidance_scale)

        # shifting latents works a bit like an image editor:
        if x_T.shape[1] == 4:
            x0_pred[:, 3, :, :] += sharp_f
            x0_pred[:, 0, :, :] += bright_f
        sampled_x = x0_pred
        # x0_pred_img = self.vae.decode((x0_pred / self.vae_scale_multiplier).to(self.model_dtype))[0].cpu()

        if self.decoder:
            sampled_x = sampled_x / self.vae_scale_multiplier  # Scale back to the original scale
            sampled_decoded = self.decoder(sampled_x)
            sampled_images = sampled_decoded.sample if hasattr(sampled_decoded, "sample") else sampled_decoded
            sampled_images = (sampled_images / 2 + 0.5).clamp(0, 1)
            # sampled_images = sampled_images - sampled_images.min()
            # sampled_images = sampled_images / sampled_images.max()
        else:
            sampled_images = x0_pred
            sampled_images = (sampled_images / 2 + 0.5).clamp(0, 1)
        return sampled_images

    def pred_image(self, noisy_image, y, noise_level, guidance_scale):
        num_imgs = noisy_image.size(0)
        noises = torch.full((2 * num_imgs, ), noise_level)
        inputs = {"t": noises.to(noisy_image.device, self.model_dtype), "x": torch.cat([noisy_image, noisy_image])}
        if y is not None:
            inputs["y"] = y.to(noisy_image.device, self.model_dtype)
            inputs["p_uncond"] = 0
        with torch.no_grad():
            x0_pred = self.denoising_model(**inputs)
            x0_pred = self.apply_classifier_free_guidance(x0_pred, num_imgs, guidance_scale)
        return x0_pred

    def apply_classifier_free_guidance(self, x0_pred, num_imgs, guidance_scale):
        """Apply classifier-free guidance to the predictions."""
        x0_pred_label, x0_pred_no_label = x0_pred[:num_imgs], x0_pred[num_imgs:]
        return guidance_scale * x0_pred_label + (1 - guidance_scale) * x0_pred_no_label



class VDM(BaseDiffusionAlgorithm):  
    def __init__(self,
            config: DiffusionTrainingConfig,
            denoising_model: nn.Module = None,
            beta_a: float = 1.0,
            beta_b: float = 2.5,
            vae: nn.Module = None,
        ):
        self.config = config
        self.denoising_model = denoising_model
        self.forward_diffusion = VDMForwardDiffusion(beta_a, beta_b)
        self.training_example_generator = VDMTrainingExampleGenerator(
            target_type="x_0",
            num_denoising_steps=config.num_denoising_steps,
            device=config.device,
        )
        self.sampler = VDMSampler(
            denoising_model=self.denoising_model,
            num_denoising_steps=config.num_denoising_steps,
            decoder=vae.decode if vae is not None else None,
            vae_scale_multiplier=config.vae_scale_multiplier,
        )

    def prepare_training_examples(self, batch: MiniBatch):
        """
        Prepare a training example for the denoising model.

        This is used in the traiining step or a validation step.
        """
        device = self.config.device
        config = self.config

        x_0 = batch.x.to(device)
        if config.data_is_latent and config.vae_scale_multiplier is not None:
            x_0 = x_0 * config.vae_scale_multiplier
        
        y = None
        p_uncond = None
        if config.conditional and batch.text_emb is not None:
            y = batch.text_emb.to(device)
            p_uncond = config.cond_drop_prob

        return self.training_example_generator.generate(x_0, self.forward_diffusion, y, p_uncond)

    def sample(self, x_T, y=None, guidance_scale: float = 0,
            exponent: float = 1, sharp_f: float = 0, bright_f: float = 0,
            noise_levels: List[float] = None, seed: int = None, quiet: bool = None,):
        """
        Generate samples from the denoising model.

        Args:
            x_T: The initial noise tensor. This provides initialization and the shape of the generated output.
            y: Embeddings of the prompt or conditioning information. It can be text embeddings.
              Only applicable to conditional generation.
            guidance_scale: The scale for the guidance mechanism.
              Only applicalbe to conditional generation.
            exponent: The exponent for the noise levels.
            sharp_f: The amount of sharpening to apply to the generated images.
            bright_f: The amount of brightening to apply to the generated images.
            noise_levels: The noise levels to use for the denoising process.
            seed: The seed for the random number generator.
            quiet: Whether to suppress the progress bar.
        """
        return self.sampler.sample(
            x_T=x_T,
            y=y,
            guidance_scale=guidance_scale,
            exponent=exponent,
            sharp_f=sharp_f,
            bright_f=bright_f,
            noise_levels=noise_levels,
            seed=seed,
            quiet=quiet,
        )
