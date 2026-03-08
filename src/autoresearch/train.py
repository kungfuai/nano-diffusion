"""
train.py — This is the file you modify.

Goal: achieve the lowest val_loss on CIFAR-10 image generation
within the 5-minute training budget.

Everything in this file is fair game:
  - Model architecture (size, depth, patch size, attention, ...)
  - Training algorithm (flow matching, DDPM, rectified flow, ...)
  - Optimizer (AdamW, Muon, Lion, ...)
  - Learning rate schedule
  - Batch size
  - Data augmentation
  - Regularization

Do NOT modify prepare.py — it is read-only and contains the fixed evaluation.

After running, this script prints:
  val_loss: X.XXXX   ← primary benchmark metric (lower is better)
  fid: XX.XX         ← secondary metric (lower is better)
  elapsed: XXXs

Baseline (current): val_loss ~0.197, fid ~65, elapsed ~300s
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from prepare import (
    get_loaders, evaluate, compute_fid,
    TIME_BUDGET_SECONDS, DEVICE, RESOLUTION, IN_CHANNELS, SEED,
)

# ─── Hyperparameters (change freely) ──────────────────────────────────────────
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 500
LR_MIN = 2e-5
USE_EMA = True
EMA_BETA = 0.999    # effective window ~1000 steps, good for ~3K step runs
RANDOM_FLIP = True
# Flow matching path: "linear" | "gvp"
PATH_TYPE = "linear"
# ODE steps for sample generation (used for FID)
SAMPLE_STEPS = 50


# ─── Model architecture ────────────────────────────────────────────────────────

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_dim = freq_dim

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
        return self.mlp(emb)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=True)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        xn = modulate(self.norm1(x), s1, sc1)
        attn_out, _ = self.attn(xn, xn, xn)
        x = x + g1.unsqueeze(1) * attn_out
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x


class DiT(nn.Module):
    """Diffusion Transformer for CIFAR-10 (32×32).

    Default: ~4M params. Good starting point for 5-min training budget.
    """
    def __init__(
        self,
        img_size=32, patch_size=2, in_ch=3,
        hidden=192, depth=6, heads=6, mlp_ratio=4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.hidden = hidden
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_ch, hidden, patch_size, stride=patch_size)
        self.register_buffer("pos_embed", self._sincos2d(img_size // patch_size, hidden))

        self.t_embed = TimestepEmbedder(hidden)
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(hidden, elementwise_affine=False)
        self.adaLN_out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=True))
        self.head = nn.Linear(hidden, patch_size * patch_size * in_ch)
        self._init_weights()

    @staticmethod
    def _sincos2d(gs, dim):
        assert dim % 4 == 0
        omega = torch.arange(dim // 4, dtype=torch.float64) / (dim // 4)
        omega = 1.0 / (10000 ** omega)
        g = torch.arange(gs, dtype=torch.float64)
        y, x = torch.meshgrid(g, g, indexing="ij")
        px = torch.outer(x.flatten(), omega)
        py = torch.outer(y.flatten(), omega)
        embed = torch.cat([px.sin(), px.cos(), py.sin(), py.cos()], dim=1)
        return embed.float().unsqueeze(0)

    def _init_weights(self):
        nn.init.zeros_(self.adaLN_out[-1].weight)
        nn.init.zeros_(self.adaLN_out[-1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(-1, h, w, p, p, self.in_ch)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(-1, self.in_ch, h * p, w * p)

    def forward(self, t, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.t_embed(t)
        for block in self.blocks:
            x = block(x, c)
        shift, scale = self.adaLN_out(c).chunk(2, dim=-1)
        x = modulate(self.norm_out(x), shift, scale)
        return self.unpatchify(self.head(x))


# ─── Flow matching ─────────────────────────────────────────────────────────────

def linear_path(t, x0, x1):
    t_ = t.view(-1, 1, 1, 1)
    return t_ * x1 + (1 - t_) * x0, x1 - x0


def gvp_path(t, x0, x1):
    a = torch.sin(math.pi * t / 2).view(-1, 1, 1, 1)
    s = torch.cos(math.pi * t / 2).view(-1, 1, 1, 1)
    da = (math.pi / 2) * torch.cos(math.pi * t / 2).view(-1, 1, 1, 1)
    ds = -(math.pi / 2) * torch.sin(math.pi * t / 2).view(-1, 1, 1, 1)
    return a * x1 + s * x0, da * x1 + ds * x0


PATH_FNS = {"linear": linear_path, "gvp": gvp_path}


def flow_loss(model, x1, device):
    """Flow matching loss on a batch. Called by evaluate() in prepare.py."""
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=device)
    xt, ut = PATH_FNS.get(PATH_TYPE, linear_path)(t, x0, x1)
    vt = model(t=t, x=xt)
    return ((vt - ut) ** 2).mean()


# ─── ODE sampling for FID ──────────────────────────────────────────────────────

@torch.no_grad()
def sample(model, n, device):
    """Euler ODE. Returns (n, 3, 32, 32) in [0, 1]."""
    x = torch.randn(n, IN_CHANNELS, RESOLUTION, RESOLUTION, device=device)
    dt = 1.0 / SAMPLE_STEPS
    for i in range(SAMPLE_STEPS):
        t = torch.full((n,), i * dt, device=device)
        x = x + model(t=t, x=x) * dt
    return (x.clamp(-1, 1) + 1) / 2


# ─── LR schedule ──────────────────────────────────────────────────────────────

def cosine_lr(step, total_steps, lr_max, lr_min, warmup):
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    p = (step - warmup) / max(total_steps - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))


def update_ema(ema, model, beta):
    with torch.no_grad():
        for ep, mp in zip(ema.parameters(), model.parameters()):
            ep.mul_(beta).add_(mp, alpha=1 - beta)


# ─── Main training loop ────────────────────────────────────────────────────────

def train():
    torch.manual_seed(SEED)
    device = torch.device(DEVICE)

    train_loader, val_loader = get_loaders(BATCH_SIZE, random_flip=RANDOM_FLIP)

    model = DiT(
        img_size=RESOLUTION, patch_size=2, in_ch=IN_CHANNELS,
        hidden=192, depth=6, heads=6, mlp_ratio=4.0,
    ).to(device)
    ema = copy.deepcopy(model) if USE_EMA else None

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )

    t_start = time.time()
    step = 0
    total_steps_est = 3000   # rough estimate; used for LR schedule only
    losses = []

    while True:
        for x1, _ in train_loader:
            elapsed = time.time() - t_start
            if elapsed >= TIME_BUDGET_SECONDS:
                break

            x1 = x1.to(device)
            lr = cosine_lr(step, total_steps_est, LEARNING_RATE, LR_MIN, WARMUP_STEPS)
            for g in optimizer.param_groups:
                g["lr"] = lr

            model.train()
            optimizer.zero_grad()
            loss = flow_loss(model, x1, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if USE_EMA:
                update_ema(ema, model, EMA_BETA)

            losses.append(loss.item())
            step += 1

            if step % 500 == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])
                print(f"  step {step:5d} | loss {avg:.4f} | lr {lr:.2e} | elapsed {elapsed:.0f}s")
        else:
            if time.time() - t_start < TIME_BUDGET_SECONDS:
                continue
        break

    elapsed = time.time() - t_start
    print(f"\nTraining done: {step} steps in {elapsed:.0f}s")

    # ─── Evaluation ───────────────────────────────────────────────────────────
    # Primary: evaluate raw model (and EMA if available)
    val_loss = evaluate(model, val_loader, device, flow_loss)
    print(f"\nval_loss: {val_loss:.4f}")

    if USE_EMA:
        ema_val_loss = evaluate(ema, val_loader, device, flow_loss)
        print(f"ema_val_loss: {ema_val_loss:.4f}")
        # Report best of the two
        best_val = min(val_loss, ema_val_loss)
        print(f"best_val_loss: {best_val:.4f}")
        eval_model = ema if ema_val_loss <= val_loss else model
    else:
        eval_model = model

    # FID (optional, ~60s)
    if os.environ.get("COMPUTE_FID", "1") == "1":
        result = compute_fid(eval_model, sample, device)
        if result is not None:
            fid, isc = result
            print(f"fid: {fid:.2f}")
            print(f"inception_score: {isc:.2f}")
    else:
        print("fid: (skipped)")

    print(f"elapsed: {elapsed:.0f}s")
    return val_loss


if __name__ == "__main__":
    train()
