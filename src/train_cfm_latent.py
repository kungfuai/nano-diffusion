"""
A minimal training pipeline for flow matching.

Supports latents and conditioning.
"""

import argparse
import copy
from datetime import datetime
from typing import Callable, Union
import os
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchdyn.core import NeuralODE

try:
    import wandb
except ImportError:
    print("wandb not installed, skipping")

try:
    from cleanfid import fid
except ImportError:
    print("clean-fid not installed, skipping")


from src.plan.ot import OTPlanSampler
from src.models.factory import create_model, choices
from src.optimizers.lr_schedule import get_cosine_schedule_with_warmup
from src.models.factory import create_model
from src.datasets import load_data
from src.bookkeeping.cfm_bookkeeping import log_training_step, \
    compute_fid, save_model, \
    save_checkpoints
from src.cfm.cfm_training_loop import update_ema_model, save_final_models, save_model, get_real_images, precompute_fid_stats_for_real_images
from src.bookkeeping.wandb_utils import load_model_from_wandb


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatcher:
    """Base class for conditional flow matching methods. This class implements the independent
    conditional flow matching methods from [1] and serves as a parent class for all other flow
    matching methods.

    It implements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla log p_t(x|x0, x1)$
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : Union[float, int]
        """
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        """Compute the lambda function, see Eq.(23) [3].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        lambda : score weighting function

        References
        ----------
        [4] Simulation-free Schrodinger bridges via score and flow matching, Preprint, Tong et al.
        """
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)



class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """Child class for optimal transport conditional flow matching method. This class implements
    the OT-CFM methods from [1] and inherits the ConditionalFlowMatcher parent class.

    It overrides the sample_location_and_conditional_flow.
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : Union[float, int]
        ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
        """
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, shape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


def generate_samples_with_flow_matching(denoising_model, device, text_embeddings, guidance_scale, vae, num_samples: int = 8, resolution: int = 32, in_channels: int = 3, parallel: bool = False, seed: int = 0, num_denoising_steps: int = 100):
    """Generate samples.

    Parameters
    ----------
    denoising_model:
        represents the neural network that we want to generate samples from
    vae:
        VAE model to decode the latents back to images
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    """
    model = denoising_model
    
    if parallel:
        import copy
        model = copy.deepcopy(denoising_model)
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model = model.to(device)

    with torch.no_grad():
        torch.manual_seed(seed)
        # create a closure for the conditioned model
        def f(t, x, *args, **kwargs):
            # TODO: Your vector field does not have `nn.Parameters` to optimize.
            # classifier-free guidance
            if isinstance(t, int):
                t = torch.full((x.shape[0],), t, device=x.device)
            elif isinstance(t, torch.Tensor):
                # if t is a scalar
                if len(t.shape) == 0:
                    t = torch.full((x.shape[0],), t, device=x.device)
                elif t.shape[0] == 1:
                    t = t.repeat(x.shape[0])
            else:
                raise ValueError("t must be an integer or a tensor with the same number of elements as x")
            
            x_twice = torch.cat([x] * 2)
            t_twice = torch.cat([t] * 2)
            if text_embeddings is not None:
                uncond_embeddings = denoising_model.get_null_cond_embed(batch_size=x.shape[0])
                embeddings_cat = torch.cat([uncond_embeddings.to(device), text_embeddings])
            else:
                embeddings_cat = None
                # print("No text embeddings for generation")
            
            with torch.no_grad():
                model_output = model(t=t_twice, x=x_twice, y=embeddings_cat)
            if hasattr(model_output, "sample"):
                model_output = model_output.sample
            
            # Split predictions and perform guidance
            v_t_uncond, v_t_cond = model_output.chunk(2)
            # print(f"v_t_uncond min: {v_t_uncond.min().item()}, max: {v_t_uncond.max().item()}, mean: {v_t_uncond.mean().item()}, std: {v_t_uncond.std().item()}")
            v_t = (1 - guidance_scale) * v_t_uncond + guidance_scale * v_t_cond
            # print(f"delta in v: {(v_t - v_t_uncond).mean().item()}")
            return v_t
        
        node = NeuralODE(f, solver="euler", sensitivity="adjoint")
        with torch.no_grad():
            # Generate latents
            traj = node.trajectory(
                torch.randn(num_samples, in_channels, resolution, resolution, device=device),
                t_span=torch.linspace(0, 1, num_denoising_steps, device=device),
            )
            latents = traj[-1, :].view([-1, in_channels, resolution, resolution]) # .clip(-1, 1)
            
            # Decode latents to data (e.g. images) using VAE
            print(f"text embedding mean: {text_embeddings.mean().item()}, std: {text_embeddings.std().item()}")
            print(f"latents min: {latents.min().item()}, max: {latents.max().item()}, mean: {latents.mean().item()}, std: {latents.std().item()}")
            data = vae.decode(latents / vae.config.scaling_factor).sample
            print(f"data min: {data.min().item()}, max: {data.max().item()}, mean: {data.mean().item()}, std: {data.std().item()}")
            # data = (data / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
    
    return data

def compute_validation_loss(model: Module, val_dataloader: DataLoader, device: torch.device, FM: ConditionalFlowMatcher, config: "TrainingConfig") -> float:
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            x_1 = batch["image_emb"].float().to(device)
            x_1 = x_1 * config.vae_scale_factor
            text_embeddings = batch.get("text_emb")
            if text_embeddings is not None:
                text_embeddings = text_embeddings.float().to(device)
            
            x_0 = torch.randn_like(x_1).to(device)
            if config.plan == "simple":
                t, x_t, u_t = FM.sample_location_and_conditional_flow(x_0, x_1)
            elif config.plan == "ot":
                t, x_t, u_t, _, text_embeddings = FM.guided_sample_location_and_conditional_flow(x_0, x_1, y0=None, y1=text_embeddings)

            v_t = model(t=t, x=x_t, y=text_embeddings)
            v_t = v_t.sample if hasattr(v_t, "sample") else v_t
            
            loss = torch.mean((v_t - u_t) ** 2)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

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
class FlowMatchingModelComponents:
    denoising_model: Module
    ema_model: Optional[Module]
    optimizer: Optimizer
    lr_scheduler: Any
    FM: ConditionalFlowMatcher  # TODO: generalize this
    vae: Optional[Module]


def training_loop(
    model_components: FlowMatchingModelComponents,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    config: TrainingConfig
) -> int:
    print(f"Training on {config.device}")

    device = torch.device(config.device)
    denoising_model = model_components.denoising_model.to(device)
    ema_model = model_components.ema_model
    optimizer = model_components.optimizer
    lr_scheduler = model_components.lr_scheduler

    if config.dataset not in ["cifar10"]:
        precompute_fid_stats_for_real_images(train_dataloader, config, Path(config.cache_dir) / "real_images_for_fid")

    if config.logger == "wandb":
        project_name = os.getenv("WANDB_PROJECT") or "nano-diffusion"
        print(f"Logging to Weights & Biases project: {project_name}")
        params = sum(p.numel() for p in model_components.denoising_model.parameters())
        run_params = asdict(config)
        run_params["model_parameters"] = params
        wandb.init(project=project_name, config=run_params)
    

    if config.logger == "wandb" and config.watch_model:
        print("  Watching model gradients (can be slow)")
        wandb.watch(model_components.denoising_model)
    
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    num_examples_trained = 0

    while step < config.total_steps:
        for batch in train_dataloader:
            if step >= config.total_steps:
                break
            
            x = batch["image_emb"]
            text_embeddings = batch.get("text_emb")
            num_examples_trained += x.shape[0]
            
            # Move batch data to device
            x = x.float().to(device)
            x = x * config.vae_scale_factor
            if text_embeddings is not None:
                text_embeddings = text_embeddings.float().to(device)
            
            denoising_model.train()
            loss = train_step(denoising_model, x, text_embeddings, model_components.FM, optimizer, config, device)
            denoising_model.eval()
            
            lr_scheduler.step()
            
            with torch.no_grad():
                if config.use_ema:
                    update_ema_model(ema_model, denoising_model, config.ema_beta)
            
                if step % config.log_every == 0:
                    log_training_step(step, num_examples_trained, loss, optimizer, config.logger)
                
                if step % config.validate_every == 0 and step > 0 and val_dataloader:
                    validate_and_log(compute_validation_loss, model_components, val_dataloader, config)
                
                if step % config.sample_every == 0:
                    # # TODO: use text embeddings from a validation batch
                    # if val_text_embeddings is not None:
                    #     assert text_embeddings.shape[0] >= config.num_samples_for_logging, f"Text embeddings shape: {text_embeddings.shape}, num_samples_for_logging: {config.num_samples_for_logging}"
                    #     text_embeddings = text_embeddings[:config.num_samples_for_logging]
                    generate_and_log_samples(model_components, config, val_dataloader, seed=0, step=step)
                
                if step % config.save_every == 0 and step > 0:
                    save_checkpoints(model_components, step, config)
                
                # TODO: FID is not working
                # if step % config.fid_every == 0 and step > 0:
                #     compute_and_log_fid(model_components, config, train_dataloader)
            
            step += 1
    
    if step > config.min_steps_for_final_save:
        save_final_models(model_components, config)

    return num_examples_trained

def train_step(denoising_model: Module, x_1: torch.Tensor, text_embeddings: torch.Tensor, FM: ConditionalFlowMatcher,
               optimizer: Optimizer, config: TrainingConfig, device: torch.device) -> torch.Tensor:
    optimizer.zero_grad()
    x_1 = x_1.to(device)

    x_0 = torch.randn_like(x_1).to(device)
    if config.plan == "simple":
        t, x_t, u_t = FM.sample_location_and_conditional_flow(x_0, x_1)
    elif config.plan == "ot":
        t, x_t, u_t, _, text_embeddings = FM.guided_sample_location_and_conditional_flow(x_0, x_1, y0=None, y1=text_embeddings)

    v_t = denoising_model(t=t, x=x_t, y=text_embeddings, p_uncond=config.cond_drop_prob)
    v_t = v_t.sample if hasattr(v_t, "sample") else v_t
    
    loss = torch.mean((v_t - u_t) ** 2)
    
    loss.backward()
    
    if config.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(denoising_model.parameters(), config.max_grad_norm)
    
    optimizer.step()
    
    return loss

def save_checkpoints(model_components: FlowMatchingModelComponents, step: int, config: TrainingConfig):
    save_model(model_components.denoising_model, Path(config.checkpoint_dir) / f"model_checkpoint_step_{step}.pth", config.logger)
    if config.use_ema:
        save_model(model_components.ema_model, Path(config.checkpoint_dir) / f"ema_model_checkpoint_step_{step}.pth", config.logger)

def validate_and_log(compute_validation_loss: Callable, model_components: FlowMatchingModelComponents, val_dataloader: DataLoader, config: TrainingConfig):
    val_loss = compute_validation_loss(model_components.denoising_model, val_dataloader, config.device, FM=model_components.FM, config=config)
    if config.use_ema:
        ema_val_loss = compute_validation_loss(model_components.ema_model, val_dataloader, config.device, FM=model_components.FM, config=config)
    
    if config.logger == "wandb":
        log_dict = {"val_loss": val_loss}
        if config.use_ema:
            log_dict["ema_val_loss"] = ema_val_loss
        wandb.log(log_dict)
    else:
        print(f"Validation Loss: {val_loss:.4f}")
        if config.use_ema:
            print(f"EMA Validation Loss: {ema_val_loss:.4f}")

def generate_and_log_samples(model_components: FlowMatchingModelComponents, config: TrainingConfig, val_dataloader: DataLoader, step: int = None, seed: int = 0):
    print(f"Generating and logging {config.num_samples_for_logging} samples on {config.device}")
    device = torch.device(config.device)
    val_batch = next(iter(val_dataloader))
    if "text_embeddings" in val_batch or "text_emb" in val_batch:
        text_embeddings = val_batch["text_embeddings"] if "text_embeddings" in val_batch else val_batch["text_emb"]
        text_embeddings = text_embeddings[:config.num_samples_for_logging].float().to(device)
        text_embeddings = text_embeddings.reshape(text_embeddings.shape[0], -1)
    else:
        text_embeddings = None
    
    denoising_model = model_components.denoising_model
    ema_model = model_components.ema_model

    # Generate random noise
    n_samples = config.num_samples_for_logging
    # Sample using the main model
    sampled_images = generate_samples_with_flow_matching(
        denoising_model=denoising_model, device=device, num_samples=n_samples,
        text_embeddings=text_embeddings,
        resolution=config.resolution, in_channels=config.in_channels,
        seed=seed, num_denoising_steps=config.num_denoising_steps,
        vae=model_components.vae,
        guidance_scale=config.guidance_scale
    )
    # normalize the images by min and max
    min_val = sampled_images.min()
    max_val = sampled_images.max()
    sampled_images = (sampled_images - min_val) / (max_val - min_val)
    images_processed = (sampled_images * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")

    if config.logger == "wandb":
        wandb.log({
            "num_batches_trained": step,
            "test_samples": [wandb.Image(img) for img in images_processed],
        })
    else:
        grid = make_grid(sampled_images, nrow=4, normalize=True, value_range=(-1, 1))
        save_image(grid, Path(config.checkpoint_dir) / f"generated_samples_step_{step}.png")

    if config.use_ema:
        ema_sampled_images = generate_samples_with_flow_matching(
            denoising_model=ema_model, device=device, num_samples=n_samples,
            text_embeddings=text_embeddings,
            resolution=config.resolution, in_channels=config.in_channels, seed=seed, num_denoising_steps=config.num_denoising_steps,
            vae=model_components.vae,
            guidance_scale=config.guidance_scale
        )
        # normalize the images by min and max
        min_val = ema_sampled_images.min()
        max_val = ema_sampled_images.max()
        ema_sampled_images = (ema_sampled_images - min_val) / (max_val - min_val)
        ema_images_processed = (ema_sampled_images * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")
        
        if config.logger == "wandb":
            wandb.log({
                "num_batches_trained": step,
                "ema_test_samples": [wandb.Image(img) for img in ema_images_processed],
            })
        else:
            # make grid
            grid = make_grid(ema_sampled_images, nrow=4, normalize=True, value_range=(-1, 1))
            save_image(grid, Path(config.checkpoint_dir) / f"ema_generated_samples_step_{step}.png")


def compute_and_log_fid(model_components: FlowMatchingModelComponents, config: TrainingConfig, train_dataloader: DataLoader = None):
    device = torch.device(config.device)
    
    if config.dataset in ["cifar10"]:
        # No need to get real images, as the stats are already computed.
        real_images = None
    
    batch_size = config.batch_size
    num_batches = (config.num_samples_for_fid + batch_size - 1) // batch_size
    generated_images = []

    def batch_generator():
        while True:
            for batch in train_dataloader:
                if batch["image_emb"].shape[0] < config.batch_size:
                    continue
                yield batch

    batch_gen = iter(batch_generator())

    count = 0
    for i in range(num_batches):
        current_batch_size = min(batch_size, config.num_samples_for_fid - len(generated_images))
        batch = next(batch_gen)
        text_embeddings = batch.get("text_emb")
        if text_embeddings is not None:
            print(f"Text embeddings shape 0: {text_embeddings.shape}")
            text_embeddings = text_embeddings[:current_batch_size].reshape(current_batch_size, -1).float().to(device)
            print(f"Text embeddings shape 1: {text_embeddings.shape}")
        batch_images = generate_samples_with_flow_matching(
            denoising_model=model_components.denoising_model, device=device, num_samples=current_batch_size,
            text_embeddings=text_embeddings,
            resolution=config.resolution, in_channels=config.in_channels,
            seed=i, num_denoising_steps=config.num_denoising_steps,
            vae=model_components.vae,
            guidance_scale=config.guidance_scale
        )
        generated_images.append(batch_images)
        count += current_batch_size
        print(f"Generated {count} out of {config.num_samples_for_fid} images")

    generated_images = torch.cat(generated_images, dim=0) # [:config.num_samples_for_fid]
    
    real_images = None
    fid_score = compute_fid(real_images, generated_images, device, config.dataset, config.resolution)
    print(f"FID Score: {fid_score:.4f}")

    if config.use_ema:
        ema_generated_images = []
        batch_size = config.batch_size
        num_batches = (config.num_samples_for_fid + batch_size - 1) // batch_size
        count = 0
        batch_gen = iter(batch_generator())
        for i in range(num_batches):
            current_batch_size = min(batch_size, config.num_samples_for_fid - len(ema_generated_images))
            batch = next(batch_gen)
            text_embeddings = batch.get("text_emb")
            batch_images = generate_samples_with_flow_matching(
                denoising_model=model_components.ema_model, device=device, num_samples=current_batch_size,
                text_embeddings=text_embeddings,
                resolution=config.resolution, in_channels=config.in_channels,
                seed=i, num_denoising_steps=config.num_denoising_steps,
                vae=model_components.vae,
                guidance_scale=config.guidance_scale
            )
            ema_generated_images.append(batch_images)
            count += current_batch_size
            print(f"EMA Generated {count} out of {config.num_samples_for_fid} images")
        
        ema_generated_images = torch.cat(ema_generated_images, dim=0) # [:config.num_samples_for_fid]
        ema_fid_score = compute_fid(real_images, ema_generated_images, device, config.dataset, resolution=config.resolution)
        print(f"EMA FID Score: {ema_fid_score:.4f}")
    
    if config.logger == "wandb":
        log_dict = {"fid": fid_score}
        if config.use_ema:
            log_dict["ema_fid"] = ema_fid_score
        wandb.log(log_dict)



def create_flow_matching_model_components(config: TrainingConfig) -> FlowMatchingModelComponents:
    device = torch.device(config.device)
    
    # Load VAE if specified
    vae = None
    if config.vae_model_name:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(config.vae_model_name, torch_dtype=torch.float32).to(device)
        vae.eval()
    
    denoising_model = create_model(net=config.net, in_channels=config.in_channels, resolution=config.resolution, cond_embed_dim=config.cond_embed_dim)
    denoising_model = denoising_model.to(device)
    ema_model = copy.deepcopy(denoising_model) if config.use_ema else None
    optimizer = optim.AdamW(denoising_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.total_steps, lr_min=config.lr_min)
    
    if config.plan == "ot":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0)
    elif config.plan == "simple":
        FM = ConditionalFlowMatcher(sigma=0)
    else:
        raise ValueError(f"Unknown plan: {config.plan}")
    
    if config.init_from_wandb_run_path and config.init_from_wandb_file:
        load_model_from_wandb(denoising_model, config.init_from_wandb_run_path, config.init_from_wandb_file)

    return FlowMatchingModelComponents(denoising_model, ema_model, optimizer, lr_scheduler, FM, vae)


def parse_arguments():
    parser = argparse.ArgumentParser(description="CFM training for images")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="zzsi/afhq64_16k_latents_sdxl_blip2",
        help="A latents dataset",
    )
    parser.add_argument("--in_channels", type=int, default=4, help="Number of input channels")
    parser.add_argument("--resolution", type=int, default=8, help="Resolution of the image. Only used for unet.")
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none", help="Logging method")
    parser.add_argument("--net", type=str, choices=choices(), default="unet_small", help="Network architecture")
    parser.add_argument("--plan", type=str, choices=["ot", "simple"], default="ot", help="The Flow Plan method")
    parser.add_argument("--num_denoising_steps", type=int, default=100, help="Number of timesteps in the flow matching process")
    parser.add_argument("--num_samples_for_logging", type=int, default=8, help="Number of samples for logging")
    parser.add_argument("--num_samples_for_fid", type=int, default=1000, help="Number of samples for FID")
    parser.add_argument("--num_real_samples_for_fid", type=int, default=10000, help="Number of real image samples for FID precomputation")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--lr_min", type=float, default=2e-6, help="Minimum learning rate")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--sample_every", type=int, default=1500, help="Sample every N steps")
    parser.add_argument("--save_every", type=int, default=60000, help="Save model every N steps")
    parser.add_argument("--validate_every", type=int, default=1500, help="Compute validation loss every N steps")
    parser.add_argument("--fid_every", type=int, default=6000, help="Compute FID every N steps")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument("--max_grad_norm", type=float, default=1, help="Maximum norm for gradient clipping. Use -1 to disable.")
    parser.add_argument("--use_loss_mean", action="store_true", help="Use loss.mean() instead of just loss")
    parser.add_argument("--watch_model", action="store_true", help="Use wandb to watch the model")
    parser.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average (EMA) for the model")
    parser.add_argument("--ema_beta", type=float, default=0.999, help="EMA decay factor")
    parser.add_argument("--random_flip", action="store_true", help="Randomly flip images horizontally")
    parser.add_argument("--checkpoint_dir", type=str, default="logs/train", help="Checkpoint directory")
    parser.add_argument("--init_from_wandb_run_path", type=str, default=None, help="Initialize model from a wandb run path")
    parser.add_argument("--init_from_wandb_file", type=str, default=None, help="Initialize model from a wandb file")
    args = parser.parse_args()
    return args



def collate_fn(batch):
    assert "image_emb" in batch[0], f"Data must be a dict that contains 'image_emb'. Got {type(batch[0])}"
    data = {
        'image_emb': torch.stack([torch.from_numpy(np.array(item['image_emb'])) for item in batch]).float(),
    }
    if "text_emb" in batch[0]:
        text_emb = torch.stack([torch.from_numpy(np.array(item["text_emb"])) for item in batch])
        data["text_emb"] = text_emb.reshape(text_emb.shape[0], -1).float()
    return data


def main():
    args = parse_arguments()
    config = TrainingConfig(**vars(args))
    
    train_dataloader, val_dataloader = load_data(config, collate_fn=collate_fn)
    model_components = create_flow_matching_model_components(config)
    
    num_examples_trained = training_loop(model_components, train_dataloader, val_dataloader, config)
    
    print(f"Training completed. Total examples trained: {num_examples_trained}")


if __name__ == "__main__":
    """
    Diagnostic run:
        
    GPU_DEVICES=0 bash bin/run.sh python -m src.train_cfm_latent --net unet_small --validate_every 30 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61
    """
    main()