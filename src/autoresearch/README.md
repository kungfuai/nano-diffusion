# nano-diffusion autoresearch

Automated AI research for diffusion model training — inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

An AI agent modifies `train.py`, runs a 5-minute training experiment on CIFAR-10, evaluates results,
and iterates. The goal is to find the best diffusion model configuration for image generation within
a fixed compute budget.

## Quick Start

```bash
cd src/autoresearch

# Prepare data (downloads CIFAR-10 to ~/.cache/torchvision)
python prepare.py

# Run baseline training (5 minutes)
python train.py

# Run without FID for faster iteration
COMPUTE_FID=0 python train.py
```

## Structure

| File | Editable | Purpose |
|------|---------|---------|
| `train.py` | **YES** | Model, optimizer, training loop — the agent modifies this |
| `prepare.py` | NO | Fixed: data loading, evaluation protocol, constants |
| `program.md` | NO | Agent instructions and research objectives |
| `README.md` | NO | This file |

## The Metric

`val_loss` — MSE between predicted and true velocity on CIFAR-10 validation, averaged over 10 batches.
Lower is better. Analogous to `val_bpb` in language model autoresearch.

Secondary metric: FID (Fréchet Inception Distance). Lower is better.

## Running Autoresearch

Point your AI coding agent at this directory with `program.md` as context:

```
You are doing autonomous ML research. Read program.md, then:
1. Look at the current train.py and its baseline results
2. Propose a change
3. Run: COMPUTE_FID=0 python train.py
4. If val_loss improved: keep it. If not: revert.
5. Repeat until the time budget is exhausted.
```

## Baseline

| Metric | Value |
|--------|-------|
| val_loss | ~0.197 |
| FID | ~65 |
| Steps in 5 min | ~2,900 |
| Model | DiT: hidden=192, depth=6, heads=6, ~4.2M params |
| Algorithm | Linear flow matching, AdamW, cosine LR |

## Task: Image Generation

The model learns to generate CIFAR-10 images (32×32 RGB) using **flow matching**:
- Start from Gaussian noise
- Learn a velocity field that transforms noise → images
- Sample by integrating the velocity ODE

This is more modern than DDPM and can generate in 10-50 steps instead of 1000.
