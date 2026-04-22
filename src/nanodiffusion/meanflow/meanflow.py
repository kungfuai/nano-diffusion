"""
MeanFlow: One-step generation via average velocity field learning.

Implements the MeanFlow framework from "Mean Flows for One-step Generative Modeling"
(Geng et al., NeurIPS 2025). Instead of learning instantaneous velocity v_t(x_t) as in
standard flow matching, MeanFlow learns the average velocity u(z_t, r, t) over an
interval [r, t], enabling one-step generation: x_0 = z_1 - u(z_1, r=0, t=1).

Key components:
1. MeanFlowModelWrapper: wraps existing models (DiT, UNet, TLD) to condition on
   both t and delta=t-r via a second time embedder.
2. MeanFlow Identity: u = v - (t-r) * du/dt, used as a self-supervised training signal.
3. JVP computation: efficiently computes the total time derivative du/dt.
4. Adaptive L2 loss: down-weights easy examples for stable training.

References:
- https://arxiv.org/abs/2505.13447
- https://arxiv.org/abs/2511.23342
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn


class MeanFlowModelWrapper(nn.Module):
    """Wraps an existing denoising model to support MeanFlow's (z, t, r) interface.

    Adds a parallel time embedder for delta=t-r. The delta embedding is summed with
    the base model's time embedding to produce a combined conditioning vector.
    Supports DiT, UNet, and TLD architectures.
    """

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self._model_type = self._detect_type()
        self._build_delta_embedder()

    def _detect_type(self):
        cls_name = type(self.model).__name__
        if cls_name == "DiT":
            return "dit"
        elif cls_name == "UNetModel":
            return "unet"
        elif cls_name == "Denoiser":
            return "tld"
        raise ValueError(f"Unsupported model type: {cls_name}")

    def _build_delta_embedder(self):
        if self._model_type == "dit":
            hidden_size = self.model.t_embedder.mlp[-1].out_features
            freq_size = self.model.t_embedder.frequency_embedding_size
            self.delta_embedder = _TimestepEmbedder(hidden_size, freq_size)
        elif self._model_type == "unet":
            mc = self.model.model_channels
            ted = self.model.time_embed[-1].out_features
            self.delta_embedder = nn.Sequential(
                nn.Linear(mc, ted), nn.SiLU(), nn.Linear(ted, ted)
            )
            self._unet_mc = mc
        elif self._model_type == "tld":
            ed = self.model.embed_dim
            ned = self.model.noise_embed_dims
            self.delta_feats = nn.Sequential(
                _SinusoidalEmbedding(embedding_dims=ned),
                nn.Linear(ned, ed), nn.GELU(), nn.Linear(ed, ed),
            )

    def forward(self, z, t, r, y=None):
        delta = t - r
        if self._model_type == "dit":
            return self._forward_dit(z, t, delta, y)
        elif self._model_type == "unet":
            return self._forward_unet(z, t, delta, y)
        elif self._model_type == "tld":
            return self._forward_tld(z, t, delta, y)

    def _forward_dit(self, z, t, delta, y):
        m = self.model
        x = m.x_embedder(z) + m.pos_embed
        c = m.t_embedder(t) + self.delta_embedder(delta)
        if y is not None:
            c = c + m.cond_proj(y)
        for block in m.blocks:
            x = block(x, c)
        x = m.final_layer(x, c)
        x = m.unpatchify(x)
        return x

    def _forward_unet(self, z, t, delta, y):
        m = self.model
        emb = m.time_embed(_timestep_embedding(t, m.model_channels))
        emb = emb + self.delta_embedder(_timestep_embedding(delta, self._unet_mc))
        if y is not None:
            emb = emb + m.cond_proj(y)
        h = z.type(m.dtype)
        hs = []
        for module in m.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = m.middle_block(h, emb)
        for module in m.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(z.dtype)
        return m.out(h)

    def _forward_tld(self, z, t, delta, y):
        m = self.model
        if t.ndim == 1:
            t = t[:, None].float()
            delta = delta[:, None].float()
        t_emb = m.fourier_feats(t).unsqueeze(1)
        delta_emb = self.delta_feats(delta).unsqueeze(1)
        if y is not None:
            y_emb = m.label_proj(y).unsqueeze(1)
        else:
            y_emb = torch.zeros_like(t_emb)
        cond = torch.cat([t_emb + delta_emb, y_emb], dim=1)
        cond = m.norm(cond)
        return m.denoiser_trans_block(z, cond)


# --- Time sampling ---

def sample_t_r(batch_size, device, flow_ratio=0.5, dist="lognorm", mu=-0.4, sigma=1.0):
    """Sample (t, r) pairs for MeanFlow training.

    Args:
        flow_ratio: fraction of samples where r=t (instantaneous velocity).
        dist: "lognorm" for logit-normal or "uniform".
        mu, sigma: logit-normal distribution parameters.

    Returns:
        t, r: tensors of shape (batch_size,) with 0 < r <= t < 1.
    """
    if dist == "lognorm":
        normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
        samples = 1.0 / (1.0 + np.exp(-normal_samples))  # sigmoid
    else:
        samples = np.random.rand(batch_size, 2).astype(np.float32)

    t_np = np.maximum(samples[:, 0], samples[:, 1])
    r_np = np.minimum(samples[:, 0], samples[:, 1])

    # For flow_ratio fraction, set r = t (instantaneous velocity learning)
    num_flow = int(flow_ratio * batch_size)
    indices = np.random.permutation(batch_size)[:num_flow]
    r_np[indices] = t_np[indices]

    return torch.tensor(t_np, device=device), torch.tensor(r_np, device=device)


# --- Loss computation ---

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """Adaptive L2 loss: sg(w) * ||delta||^2 where w = 1/(||delta||^2 + c)^p."""
    delta_sq = torch.mean(error ** 2, dim=list(range(1, error.ndim)))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    return (w.detach() * delta_sq).mean()


def compute_meanflow_loss(model, x, device, flow_ratio=0.5, time_dist="lognorm",
                          time_mu=-0.4, time_sigma=1.0, gamma=0.5):
    """Compute MeanFlow training loss using the MeanFlow identity and JVP.

    The MeanFlow identity: u(z_t, r, t) = v(z_t, t) - (t-r) * d/dt u(z_t, r, t)
    Target: u_tgt = v - (t-r) * dudt, where dudt is computed via JVP.

    Args:
        model: MeanFlowModelWrapper that takes (z, t, r).
        x: clean data batch, shape (B, C, H, W), assumed normalized to [-1, 1].
        device: torch device.
        flow_ratio: fraction of samples with r=t.
        time_dist: time sampling distribution.
        time_mu, time_sigma: logit-normal parameters.
        gamma: adaptive loss power (0.5 = default from paper).

    Returns:
        loss: scalar training loss.
        mse: scalar MSE for logging (detached).
    """
    batch_size = x.shape[0]
    t, r = sample_t_r(batch_size, device, flow_ratio, time_dist, time_mu, time_sigma)

    t_ = t.view(-1, 1, 1, 1)
    r_ = r.view(-1, 1, 1, 1)

    e = torch.randn_like(x)
    z = (1 - t_) * x + t_ * e  # noisy sample on the flow path
    v = e - x  # ground truth instantaneous velocity

    # Compute u and du/dt via JVP (forward-mode AD).
    # Tangent vectors: dz/dt = v, dt/dt = 1, dr/dt = 0
    # This computes the total time derivative of u along the flow trajectory.
    # We disable flash/efficient SDPA backends because they don't support
    # the higher-order gradients required by JVP.
    fn = partial(_model_fn, model=model)
    tangents = (v, torch.ones_like(t), torch.zeros_like(r))

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        u, dudt = torch.autograd.functional.jvp(
            fn, (z, t, r), tangents, create_graph=True,
        )

    # Rearranging the MeanFlow identity gives:
    #   u + (t-r) * du/dt = v
    # so the composite quantity on the left approximates the instantaneous
    # velocity v, even though u itself is an interval-conditioned average
    # velocity rather than an instantaneous field. We detach the target branch
    # during optimization, but that stop-gradient does not change the forward
    # interpretation of the identity.
    #
    # MeanFlow target: u_tgt = v - (t-r) * dudt
    u_tgt = v - (t_ - r_) * dudt

    error = u - u_tgt.detach()
    loss = adaptive_l2_loss(error, gamma=gamma)
    mse = (error.detach() ** 2).mean()

    return loss, mse


def _model_fn(z, t, r, model):
    """Thin wrapper for JVP: calls model(z, t, r)."""
    return model(z, t, r)


# --- Sampling ---

@torch.no_grad()
def generate_samples_meanflow(model, device, num_samples=8, resolution=32,
                              in_channels=3, sample_steps=1, seed=0):
    """Generate samples using MeanFlow's average velocity field.

    For one-step generation (sample_steps=1): x = z - u(z, r=0, t=1).
    For multi-step: iteratively apply z_r = z_t - (t-r) * u(z_t, r, t).

    Args:
        model: MeanFlowModelWrapper.
        sample_steps: number of sampling steps (1 for one-step generation).
    """
    model.eval()
    torch.manual_seed(seed)

    z = torch.randn(num_samples, in_channels, resolution, resolution, device=device)
    t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

    for i in range(sample_steps):
        t = torch.full((num_samples,), t_vals[i], device=device)
        r = torch.full((num_samples,), t_vals[i + 1], device=device)
        t_ = t.view(-1, 1, 1, 1)
        r_ = r.view(-1, 1, 1, 1)
        u = model(z, t, r)
        z = z - (t_ - r_) * u

    z = z.clip(-1, 1)
    z = z / 2 + 0.5  # denormalize to [0, 1]
    return z


# --- Helper modules ---

def _timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embedding (matches UNet implementation)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for TLD compatibility."""
    def __init__(self, emb_min_freq=1.0, emb_max_freq=1000.0, embedding_dims=32):
        super().__init__()
        frequencies = torch.exp(
            torch.linspace(np.log(emb_min_freq), np.log(emb_max_freq), embedding_dims // 2)
        )
        self.register_buffer("angular_speeds", 2.0 * torch.pi * frequencies)

    def forward(self, x):
        return torch.cat(
            [torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1
        )


class _TimestepEmbedder(nn.Module):
    """Timestep embedder for DiT compatibility."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        args = t[:, None].float() * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            t_freq = torch.cat([t_freq, torch.zeros_like(t_freq[:, :1])], dim=-1)
        return self.mlp(t_freq)
