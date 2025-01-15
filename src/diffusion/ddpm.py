"""
DDPM is a diffusion training and sampling algorithm.
"""
from typing import Dict, Callable
import torch
from torch import nn
from .base import BaseDiffusionAlgorithm
from ..bookkeeping.mini_batch import MiniBatch
from ..config.diffusion_training_config import DiffusionTrainingConfig
from .noise_scheduler import forward_diffusion, denoising_step, generate_samples_by_denoising


class ForwardDiffusion:
    def __init__(self, noise_schedule: Dict[str, torch.Tensor]):
        self.noise_schedule = noise_schedule

    def compute(self, x_0, t, noise=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion process to add noise to images.

        Args:
            x_0: The input images to add noise to.
            t: The noise levels to add to the images.
            noise: The noise to add to the images. If not provided, random noise is added.

        Returns:
            A tuple containing the noise levels (time steps), the noisy images and the noise added to the images.
        """
        return (t,) + forward_diffusion(x_0, t, self.noise_schedule, noise)
    
    def __call__(self, x_0, t, noise=None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.compute(x_0, t, noise)


class TrainingExampleGenerator:
    def __init__(self, target_type: str, num_denoising_steps: int, device: str | torch.device):
        self.target_type = target_type
        self.num_denoising_steps = num_denoising_steps
        self.device = device

    def generate(self, x_0, forward_diffusion, y, p_uncond) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        num_examples = x_0.shape[0]
        noise = torch.randn_like(x_0)
        t = torch.randint(0, self.num_denoising_steps, (num_examples,), device=self.device).long()
        _, x_t, noise = forward_diffusion.compute(x_0, t, noise)
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


class DDPMSampler:
    """
    DDPM sampling algorithm for generating samples (e.g. images) using a denoising model.
    """
    def __init__(self,
            denoising_model: nn.Module,
            num_denoising_steps: int = 1000,
            guidance_scale: float = None,
            noise_schedule: Dict[str, torch.Tensor] = None,
            clip_sample=True,
            clip_sample_range=1.0,
            decoder: nn.Module | Callable = None,
            vae_scale_multiplier: float = 1.0,
            seed=0,
        ):
        self.denoising_model = denoising_model
        self.num_denoising_steps = num_denoising_steps
        self.noise_schedule = noise_schedule
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.guidance_scale = guidance_scale
        self.decoder = decoder
        self.vae_scale_multiplier = vae_scale_multiplier
        self.seed = seed

    def sample(self, x_T, y = None, guidance_scale: float = None, seed: int = None, quiet: bool = None):
        sampled_x = generate_samples_by_denoising(
            denoising_model=self.denoising_model,
            x_T=x_T,
            y=y,
            noise_schedule=self.noise_schedule,
            n_T=self.num_denoising_steps,
            guidance_scale=guidance_scale or self.guidance_scale,
            clip_sample=self.clip_sample,
            clip_sample_range=self.clip_sample_range,
            seed=seed or self.seed,
            quiet=quiet,
        )
        if self.decoder:
            sampled_x = sampled_x / self.vae_scale_multiplier  # Scale back to the original scale
            sampled_decoded = self.decoder(sampled_x)
            sampled_images = sampled_decoded.sample if hasattr(sampled_decoded, "sample") else sampled_decoded
            sampled_images = (sampled_images / 2 + 0.5).clamp(0, 1)
            # sampled_images = sampled_images - sampled_images.min()
            # sampled_images = sampled_images / sampled_images.max()
            # print(f"sampled_images: min={sampled_images.min()}, max={sampled_images.max()}, std={sampled_images.std()}")
        else:
            # Assume the data is image.
            # TODO: add logic to postprocess other data types (e.g. audio).
            sampled_images = sampled_x
            sampled_images = (sampled_images / 2 + 0.5).clamp(0, 1)
        return sampled_images


class DDPM(BaseDiffusionAlgorithm):
    def __init__(self,
            config: DiffusionTrainingConfig,
            denoising_model: nn.Module = None,
            noise_schedule: Dict[str, torch.Tensor] = None,
            vae: nn.Module = None,
        ):
        self.config = config
        self.noise_schedule = noise_schedule
        self.denoising_model = denoising_model
        self.forward_diffusion = ForwardDiffusion(noise_schedule)
        self.training_example_generator = TrainingExampleGenerator(
            target_type="noise",
            num_denoising_steps=config.num_denoising_steps,
            device=config.device,
        )
        self.sampler = DDPMSampler(
            denoising_model=self.denoising_model,
            num_denoising_steps=config.num_denoising_steps,
            noise_schedule=self.noise_schedule,
            guidance_scale=config.guidance_scale,
            clip_sample=(config.clip_sample_range or 0) > 0,
            clip_sample_range=config.clip_sample_range,
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

    def sample(self, x_T, y=None, guidance_scale: float = None, seed: int = None, quiet: bool = None):
        """
        Generate samples from the denoising model.

        Args:
            x_T: The initial noise tensor. This provides initialization and the shape of the generated output.
            y: Embeddings of the prompt or conditioning information. It can be text embeddings.
              Only applicable to conditional generation.
            guidance_scale: The scale for the guidance mechanism.
              Only applicalbe to conditional generation.
            seed: The seed for the random number generator.
            quiet: Whether to suppress the progress bar.
        """
        return self.sampler.sample(x_T, y, guidance_scale, seed, quiet)
