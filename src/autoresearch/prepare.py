"""
prepare.py — READ-ONLY. Do NOT modify this file.

Fixed utilities for the nano-diffusion autoresearch benchmark:
  - Data loading (CIFAR-10)
  - Evaluation (val_loss + FID)
  - Fixed benchmark constants

The agent only modifies train.py.
"""

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

# ─── Fixed benchmark constants ────────────────────────────────────────────────
TIME_BUDGET_SECONDS = 300   # 5-minute training budget per experiment
DEVICE = "cuda:0"           # single-GPU
RESOLUTION = 32             # CIFAR-10 image size
IN_CHANNELS = 3             # RGB
EVAL_BATCH_SIZE = 512       # batch size for validation loss
NUM_VAL_BATCHES = 10        # how many batches to use for fast val_loss estimate
NUM_FID_SAMPLES = 2048      # samples for FID (takes ~60s on Blackwell)
SEED = 42


def get_data_transforms():
    """Fixed normalization: maps [0,1] -> [-1, 1]."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_loaders(batch_size: int, random_flip: bool = True, cache_dir: str = None):
    """Load CIFAR-10 train and val splits.

    Returns
    -------
    train_loader, val_loader
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/torchvision")

    transforms = T.Compose([
        T.RandomHorizontalFlip() if random_flip else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_transforms = get_data_transforms()

    full = torchvision.datasets.CIFAR10(root=cache_dir, train=True, download=True, transform=transforms)
    val_ds = torchvision.datasets.CIFAR10(root=cache_dir, train=False, download=True, transform=val_transforms)

    train_loader = DataLoader(full, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, val_loader, device, flow_fn):
    """Compute validation loss.  This is the primary benchmark metric.

    Parameters
    ----------
    model : nn.Module
        The denoising model (must accept t, x as inputs).
    val_loader : DataLoader
        Validation data loader from get_loaders().
    device : torch.device or str
    flow_fn : callable
        A function (model, x1, device) -> loss  — the same loss used in training.
        Must be deterministic given the same x1 (use a fixed seed inside).

    Returns
    -------
    val_loss : float  ← the benchmark metric. Lower is better.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    for i, (x1, _) in enumerate(val_loader):
        if i >= NUM_VAL_BATCHES:
            break
        x1 = x1.to(device)
        loss = flow_fn(model, x1, device)
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def compute_fid(model, sample_fn, device):
    """Compute FID against CIFAR-10 test set using torch-fidelity.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    sample_fn : callable
        A function (model, n, device) -> Tensor(n, 3, 32, 32) in [0, 1].
    device : str or torch.device

    Returns
    -------
    fid : float  or None if torch-fidelity not available
    """
    try:
        import torch_fidelity
    except ImportError:
        print("torch-fidelity not installed; skipping FID. Install with: pip install torch-fidelity")
        return None

    import tempfile
    from PIL import Image
    import numpy as np

    cache_dir = os.path.expanduser("~/.cache/torchvision")

    # Generate samples
    print(f"Generating {NUM_FID_SAMPLES} samples for FID...")
    t0 = time.time()
    model.eval()
    samples = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, NUM_FID_SAMPLES, batch_size):
            n = min(batch_size, NUM_FID_SAMPLES - len(samples) * batch_size)
            if len(samples) * batch_size >= NUM_FID_SAMPLES:
                break
            batch = sample_fn(model, batch_size, device)
            samples.append(batch.cpu())
    samples = torch.cat(samples, dim=0)[:NUM_FID_SAMPLES]
    print(f"  Generated {len(samples)} samples in {time.time() - t0:.1f}s")

    # Save to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, img in enumerate(samples):
            arr = (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(tmpdir, f"{i:05d}.png"))

        metrics = torch_fidelity.calculate_metrics(
            input1=tmpdir,
            input2="cifar10-train",
            cuda=True,
            fid=True,
            isc=True,
            verbose=False,
        )

    fid = metrics.get("frechet_inception_distance", None)
    isc = metrics.get("inception_score_mean", None)
    return fid, isc


if __name__ == "__main__":
    print("Preparing CIFAR-10 dataset...")
    train_loader, val_loader = get_loaders(batch_size=128)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    x, y = next(iter(val_loader))
    print(f"Sample batch: x={x.shape}, range=[{x.min():.2f}, {x.max():.2f}]")
    print("Done. CIFAR-10 ready.")
