"""
A minimal training pipeline for flow matching.

The CFM implementation is based on https://github.com/atong01/conditional-flow-matching

## Main components

1. Flow Process:
   Defines a continuous-time path from the data distribution to a simple prior distribution (e.g., Gaussian).

2. Vector Field Estimation:
   A neural network learns to estimate the time-dependent vector field that guides the flow process.

3. Loss Function:
   Minimizes the difference (typically L2 loss) between the estimated and ground truth vector fields.
   
4. Sampling:
   Generates new samples by integrating the learned vector field backwards in time from the prior distribution.

5. ODE Solver:
   Numerical method (e.g., Euler, Runge-Kutta) used to solve the ordinary differential equation defined by the vector field.

6. Time Conditioning:
   The vector field estimation is conditioned on the current time step to capture the dynamics of the flow process.
"""

import argparse
import copy
from typing import Callable, Union
import os
from pathlib import Path
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image, make_grid
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
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
from src.models.factory import create_model
from src.models.ema import create_ema_model
from src.optimizers.lr_schedule import get_cosine_schedule_with_warmup
from src.train import create_model, create_ema_model, log_training_step, \
    compute_fid, save_model, load_data, update_ema_model, save_final_models, get_real_images, \
    save_checkpoints, load_model_from_wandb



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


def generate_samples_with_flow_matching(denoising_model, device, num_samples: int = 8, parallel: bool = False, seed: int = 0):
    """Generate samples.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model_ = denoising_model
    
    if parallel:
        import copy
        model_ = copy.deepcopy(denoising_model)
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.to(device)

    with torch.no_grad():
        torch.manual_seed(seed)
        node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
        with torch.no_grad():
            traj = node_.trajectory(
                torch.randn(num_samples, 3, 32, 32, device=device),
                t_span=torch.linspace(0, 1, 100, device=device),
            )
            traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
            traj = traj / 2 + 0.5
    
    return traj  # range is expected to be [0, 1]

def compute_validation_loss(model: Module, val_dataloader: DataLoader, device: torch.device, FM: ConditionalFlowMatcher) -> float:
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for x_1, _ in val_dataloader:
            x_1 = x_1.to(device)
            
            x_0 = torch.randn_like(x_1).to(device)
            t, x_t, u_t = FM.sample_location_and_conditional_flow(x_0, x_1)

            v_t = model(t=t, x=x_t)
            v_t = v_t.sample if hasattr(v_t, "sample") else v_t
            
            loss = torch.mean((v_t - u_t) ** 2)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

@dataclass
class TrainingConfig:
    # Model architecture
    in_channels: int # number of input channels
    resolution: int # resolution of the image
    net: str # network architecture
    num_denoising_steps: int # number of timesteps
    
    # Flow plan algorithm
    plan: str # flow matching plan

    # Training loop and optimizer
    total_steps: int  # total number of training steps
    batch_size: int # batch size
    learning_rate: float # initial learning rate
    weight_decay: float # weight decay
    lr_min: float # minimum learning rate
    warmup_steps: int # number of warmup steps

    # Logging and evaluation
    log_every: int # log every N steps
    sample_every: int # sample every N steps
    save_every: int # save model every N steps
    validate_every: int # compute validation loss every N steps
    fid_every: int # compute FID every N steps
    num_samples_for_fid: int = 1000 # number of samples for FID
    num_samples_for_logging: int = 16 # number of samples for logging

    # Regularization
    max_grad_norm: float = -1 # maximum norm for gradient clipping
    use_loss_mean: bool = False # use loss.mean() instead of just loss
    use_ema: bool = False # use EMA for the model
    ema_beta: float = 0.9999 # EMA decay factor

    # Accelerator
    device: str = "cuda" # device to use for training

    # Logging
    logger: str = "wandb" # logging method
    checkpoint_dir: str = "logs/train" # checkpoint directory
    min_steps_for_final_save: int = 100 # minimum steps for final save
    watch_model: bool = False # watch the model with wandb
    init_from_wandb_run_path: str = None # initialize model from a wandb run path
    init_from_wandb_file: str = None # initialize model from a wandb file

    # Data augmentation
    random_flip: bool = False # randomly flip images horizontally


@dataclass
class FlowMatchingModelComponents:
    denoising_model: Module
    ema_model: Optional[Module]
    optimizer: Optimizer
    lr_scheduler: Any
    FM: ConditionalFlowMatcher  # TODO: generalize this


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
        for x, _ in train_dataloader:
            if step >= config.total_steps:
                break
            
            num_examples_trained += x.shape[0]
            
            # Move batch data to device
            x = x.to(device)
            
            denoising_model.train()
            loss = train_step(denoising_model, x, model_components.FM, optimizer, config, device)
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
                    generate_and_log_samples(model_components, config, seed=0, step=step)
                    # log_denoising_results(model_components, config, step, train_dataloader)
                
                if step % config.save_every == 0 and step > 0:
                    save_checkpoints(model_components, step, config)
                
                if step % config.fid_every == 0 and step > 0:
                    compute_and_log_fid(model_components, config, train_dataloader)
            
            step += 1
    
    if step > config.min_steps_for_final_save:
        save_final_models(model_components, config)

    return num_examples_trained
def train_step(denoising_model: Module, x_1: torch.Tensor, FM: ConditionalFlowMatcher, 
               optimizer: Optimizer, config: TrainingConfig, device: torch.device) -> torch.Tensor:
    optimizer.zero_grad()
    x_1 = x_1.to(device)

    x_0 = torch.randn_like(x_1).to(device)
    t, x_t, u_t = FM.sample_location_and_conditional_flow(x_0, x_1)

    v_t = denoising_model(t=t, x=x_t)
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
    val_loss = compute_validation_loss(model_components.denoising_model, val_dataloader, config.device, FM=model_components.FM)
    if config.use_ema:
        ema_val_loss = compute_validation_loss(model_components.ema_model, val_dataloader, config.device, FM=model_components.FM)
    
    if config.logger == "wandb":
        log_dict = {"val_loss": val_loss}
        if config.use_ema:
            log_dict["ema_val_loss"] = ema_val_loss
        wandb.log(log_dict)
    else:
        print(f"Validation Loss: {val_loss:.4f}")
        if config.use_ema:
            print(f"EMA Validation Loss: {ema_val_loss:.4f}")

def generate_and_log_samples(model_components: FlowMatchingModelComponents, config: TrainingConfig, step: int = None, seed: int = 0):
    print(f"Generating and logging samples on {config.device}")
    device = torch.device(config.device)
    denoising_model = model_components.denoising_model
    ema_model = model_components.ema_model

    # Generate random noise
    # TODO: make this a config parameter
    n_samples = config.num_samples_for_logging
    # Sample using the main model
    sampled_images = generate_samples_with_flow_matching(denoising_model, device, n_samples, seed=seed)
    images_processed = (sampled_images * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")

    if config.logger == "wandb":
        for i in range(images_processed.shape[0]):
            wandb.log({
                "num_batches_trained": step,
                "test_samples": [wandb.Image(img) for img in images_processed],
            })
    else:
        grid = make_grid(sampled_images, nrow=4, normalize=True, value_range=(-1, 1))
        save_image(grid, Path(config.checkpoint_dir) / f"generated_samples_step_{step}.png")

    if config.use_ema:
        ema_sampled_images = generate_samples_with_flow_matching(ema_model, device, n_samples)
        ema_images_processed = (ema_sampled_images * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")
        
        if config.logger == "wandb":
            for i in range(ema_images_processed.shape[0]):
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
    
    real_images = get_real_images(config.num_samples_for_fid, train_dataloader)
    generated_images = generate_samples_with_flow_matching(model_components.denoising_model, device, config.num_samples_for_fid)
    
    fid_score = compute_fid(real_images, generated_images, device)
    print(f"FID Score: {fid_score:.4f}")

    if config.use_ema:
        ema_generated_images = generate_samples_with_flow_matching(model_components.ema_model, device, config.num_samples_for_fid)
        ema_fid_score = compute_fid(real_images, ema_generated_images, device)
        print(f"EMA FID Score: {ema_fid_score:.4f}")
    
    if config.logger == "wandb":
        log_dict = {"fid": fid_score}
        if config.use_ema:
            log_dict["ema_fid"] = ema_fid_score
        wandb.log(log_dict)


def load_data(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    if config.random_flip:
        transforms_list.insert(0, transforms.RandomHorizontalFlip())
    
    transform = transforms.Compose(transforms_list)

    full_dataset = CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    return train_dataloader, val_dataloader

def create_flow_matching_model_components(config: TrainingConfig) -> FlowMatchingModelComponents:
    device = torch.device(config.device)
    denoising_model = create_model(net=config.net, in_channels=config.in_channels, resolution=config.resolution)
    denoising_model = denoising_model.to(device)
    # ema_model = create_ema_model(denoising_model, config.ema_beta) if config.use_ema else None
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

    return FlowMatchingModelComponents(denoising_model, ema_model, optimizer, lr_scheduler, FM)


def create_noise_schedule(n_T: int, device: torch.device) -> Dict[str, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, n_T).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)
    alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1].to(device)])
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="CFM training for images")
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none", help="Logging method")
    parser.add_argument("--net", type=str, choices=[
        "dit_t0", "dit_t1", "dit_t2", "dit_t3",
        "dit_s2", "dit_b2", "dit_b4", 
        "dit_b2", "dit_b4", "dit_l2", "dit_l4",
        "unet_small", "unet", "unet_large",
    ], default="unet_small", help="Network architecture")
    parser.add_argument("--plan", type=str, choices=["ot", "simple"], default="ot", help="The Flow Plan method")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--resolution", type=int, default=32, help="Resolution of the image. Only used for unet.")
    parser.add_argument("--num_denoising_steps", type=int, default=1000, help="Number of timesteps in the diffusion process")
    parser.add_argument("--num_samples_for_logging", type=int, default=16, help="Number of samples for logging")
    parser.add_argument("--num_samples_for_fid", type=int, default=1000, help="Number of samples for FID")
    parser.add_argument("--total_steps", type=int, default=300000, help="Total number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
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
    parser.add_argument("--max_grad_norm", type=float, default=-1, help="Maximum norm for gradient clipping")
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


def denoise_and_compare(
        model: torch.nn.Module, 
        images: torch.Tensor, 
        forward_diffusion: Callable, 
        denoising_step: Callable, 
        noise_schedule: Dict, 
        n_T: int,  # timesteps for diffusion
        device: str,
    ):
    # TODO: adapt it to use flow matching.
    torch.manual_seed(10)
    model.eval()
    with torch.no_grad():
        # Add noise to the images
        t = torch.randint(0, n_T, (images.shape[0],), device=device)
        x_t, _ = forward_diffusion(images, t, noise_schedule)
        
        # Denoise the images
        pred_noise = model(x_t, t)
        if hasattr(pred_noise, "sample"):
            pred_noise = pred_noise.sample
        pred_previous_images = denoising_step(model, x_t, t, noise_schedule)
        # Compute the predicted original images using the correct formula
        alpha_t = noise_schedule["alphas"][t][:, None, None, None]
        alpha_t_cumprod = noise_schedule["alphas_cumprod"][t][:, None, None, None]
        pred_original_images = (
            x_t - ((1 - alpha_t) / (1 - alpha_t_cumprod).sqrt()) * pred_noise) / (alpha_t / (1 - alpha_t_cumprod).sqrt())
    model.train()
    return x_t, pred_original_images

def log_denoising_results(model_components: FlowMatchingModelComponents, config: TrainingConfig, step: int, train_dataloader: DataLoader):
    device = torch.device(config.device)
    denoising_model = model_components.denoising_model
    ema_model = model_components.ema_model
    noise_schedule = model_components.noise_schedule

    # Get a batch of real images
    real_images, _ = next(iter(train_dataloader))
    real_images = real_images[:8].to(device)  # Use 8 images for visualization

    # Denoise and compare using real images
    denoised, _ = denoise_and_compare(denoising_model, real_images, noise_schedule, config.num_denoising_steps, device)

    # Create grids
    real_grid = make_grid(real_images, nrow=4, normalize=True, value_range=(-1, 1))
    denoised_grid = make_grid(denoised, nrow=4, normalize=True, value_range=(-1, 1))

    if config.logger == "wandb":
        wandb.log({
            "real_images": wandb.Image(real_grid),
            "denoised_images": wandb.Image(denoised_grid),
        })
    else:
        save_image(real_grid, Path(config.checkpoint_dir) / f"real_images_step_{step}.png")
        save_image(denoised_grid, Path(config.checkpoint_dir) / f"denoised_images_step_{step}.png")
        # You might want to save the MSE list to a file here

    if config.use_ema:
        ema_denoised, _ = denoise_and_compare(ema_model, real_images, noise_schedule, config.num_denoising_steps, device)
        ema_denoised_grid = make_grid(ema_denoised, nrow=4, normalize=True, value_range=(-1, 1))
        
        if config.logger == "wandb":
            wandb.log({
                "ema_denoised_images": wandb.Image(ema_denoised_grid),
            })
        else:
            save_image(ema_denoised_grid, Path(config.checkpoint_dir) / f"ema_denoised_images_step_{step}.png")


def main():
    args = parse_arguments()
    config = TrainingConfig(**vars(args))
    
    train_dataloader, val_dataloader = load_data(config)
    model_components = create_flow_matching_model_components(config)
    
    num_examples_trained = training_loop(model_components, train_dataloader, val_dataloader, config)
    
    print(f"Training completed. Total examples trained: {num_examples_trained}")


if __name__ == "__main__":
    main()