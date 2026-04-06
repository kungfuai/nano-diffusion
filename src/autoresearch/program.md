# nano-diffusion autoresearch

You are an AI research agent running automated experiments on a diffusion model training task.

## The Task

**Goal**: achieve the lowest `val_loss` on CIFAR-10 image generation within a 5-minute training budget.

This is continuous image generation using **flow matching** (a modern alternative to DDPM). The model
learns to predict a velocity field that transforms Gaussian noise into CIFAR-10 images.

The metric `val_loss` is the MSE between predicted and true velocity, computed on 10 validation
batches. Lower is better. This is an analogue of `val_bpb` in language models.

## The Rules

1. **You may only modify `train.py`**. This is the only file you edit.
2. **Do NOT modify `prepare.py`** — it is read-only and contains the fixed evaluation protocol.
3. The training script runs for exactly 5 minutes (`TIME_BUDGET_SECONDS = 300`).
4. All else being equal, **simpler is better**. A small improvement with ugly complexity is not worth it.
   Removing something and getting equal or better results is a great outcome.
5. If you find a change that hurts results, revert it and try something else.

## What You Can Change

Everything in `train.py` is fair game:

**Architecture:**
- Model size: hidden dim, depth, number of heads, MLP ratio
- Patch size (2 or 4 — smaller patch = more tokens = richer representation but slower)
- Attention variant: standard MHA → flash attention, grouped-query, linear attention
- Add positional encoding improvements (RoPE, ALiBi)
- Add normalization changes (RMSNorm instead of LayerNorm)
- Pre-norm vs post-norm, QK-norm
- Add gating, mixture-of-experts, etc.

**Training algorithm:**
- Flow matching path: `linear` (current), `gvp` (trigonometric), `vp` (variance-preserving)
- Prediction type: velocity (current), noise/epsilon, x0 prediction
- Loss weighting: uniform time sampling → logit-normal, log-normal, min-SNR weighting
- Timestep sampling distribution

**Optimizer:**
- AdamW hyperparameters: beta1, beta2, epsilon, weight decay
- Learning rate schedule: cosine (current), warmup-stable-decay, constant, linear
- Gradient clipping value
- Try Muon optimizer, Lion, 8-bit Adam, etc.
- Try EMA beta values

**Data:**
- Batch size (larger = more stable gradients but fewer steps)
- Data augmentation (RandomHorizontalFlip is current)
- Add RandAugment, CutMix, Mixup, etc.

**Other:**
- Mixed precision (torch.autocast for fp16/bf16 — can 2-3x throughput)
- Compile with torch.compile
- Gradient accumulation

## Baseline

The baseline `train.py` uses:
- DiT architecture: hidden=192, depth=6, heads=6, ~4.2M params
- Linear flow matching (velocity prediction)
- AdamW, cosine LR, batch=256, 5 min budget
- **Baseline val_loss: ~0.197**
- **Baseline FID: ~65** (estimated; run with `COMPUTE_FID=1`)

## Workflow

For each experiment:
1. Read the current `train.py`
2. Propose ONE change (or a small set of related changes)
3. Run: `python train.py` (from the `autoresearch/` directory)
4. Record: val_loss, FID (if computed), any notes
5. If val_loss improved: keep the change. If not: revert.
6. Repeat.

Track results in a table as you go:

| Experiment | Change | val_loss | FID | Notes |
|-----------|--------|---------|-----|-------|
| 0 (baseline) | — | 0.197 | ~65 | DiT/6, hidden=192, linear, AdamW |
| 1 | ... | ... | ... | ... |

## Hardware

- 2× RTX PRO 6000 Blackwell (96GB each). Use `DEVICE = "cuda:0"` or `"cuda:1"`.
- The machine has PyTorch 2.10.0+cu128 installed.
- At batch_size=256 with hidden=192, depth=6: ~2,500 steps in 5 min.
- To disable FID (faster experiments): `COMPUTE_FID=0 python train.py`

## Research Hints

- **Logit-normal time sampling** improved val_loss by ~8% in rectified flow experiments on this codebase.
- **torch.compile** + bf16 autocast can give 2-3x speedup, enabling 2-3x more steps in the time budget.
- **Larger batch** (512-1024) with flash attention can improve stability at same step count.
- **Patch size 4** halves sequence length, allowing deeper models within the same compute.
- **Weight decay** on all params (including LayerNorm) has historically helped in similar tasks.
- **GVP path** (trigonometric) had slightly worse FID than linear in preliminary experiments.

Good luck!
