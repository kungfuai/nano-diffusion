"""
Patch masking utilities for efficient DiT training.

Based on Micro-Diffusion (Stretching Each Dollar, CVPR 2025):
https://github.com/SonyResearch/micro_diffusion

Masking 50-75% of patches during training reduces compute by 3-4x while
maintaining generation quality, especially when combined with a lightweight
patch-mixer that processes all patches before masking (deferred masking).
"""

import torch
from typing import Dict


def get_mask(batch: int, length: int, mask_ratio: float, device: torch.device) -> Dict[str, torch.Tensor]:
    """Generate a random binary mask for patch tokens.

    Parameters
    ----------
    batch : int
        Batch size.
    length : int
        Number of patches (sequence length).
    mask_ratio : float
        Fraction of patches to mask (remove). E.g., 0.75 means keep 25%.
    device : torch.device
        Device for tensors.

    Returns
    -------
    dict with keys:
        mask : (N, T) binary tensor, 0 = keep, 1 = remove
        ids_keep : (N, len_keep) indices of kept patches
        ids_restore : (N, T) indices to unshuffle back to original order
    """
    len_keep = int(length * (1 - mask_ratio))
    noise = torch.rand(batch, length, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]

    mask = torch.ones([batch, length], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return {
        "mask": mask,
        "ids_keep": ids_keep,
        "ids_restore": ids_restore,
    }


def mask_out_tokens(x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
    """Remove masked tokens from the sequence.

    Parameters
    ----------
    x : (N, L, D) tensor of patch tokens
    ids_keep : (N, len_keep) indices of tokens to keep

    Returns
    -------
    x_masked : (N, len_keep, D) tensor with only kept tokens
    """
    N, L, D = x.shape
    return torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))


def unmask_tokens(x: torch.Tensor, ids_restore: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    """Restore full sequence by inserting mask tokens at removed positions.

    Parameters
    ----------
    x : (N, len_keep, D) tensor of processed kept tokens
    ids_restore : (N, T) indices to restore original spatial order
    mask_token : (1, 1, D) learned mask token to insert at removed positions

    Returns
    -------
    x_full : (N, T, D) tensor with full sequence restored
    """
    num_masked = ids_restore.shape[1] - x.shape[1]
    mask_tokens = mask_token.expand(x.shape[0], num_masked, -1)
    x_ = torch.cat([x, mask_tokens], dim=1)
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
    return x_


def compute_masked_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Compute MSE loss only on unmasked patches.

    Parameters
    ----------
    predictions : (N, C, H, W) predicted output
    targets : (N, C, H, W) ground truth
    mask : (N, num_patches) binary mask, 0 = keep, 1 = remove
    patch_size : int
        Spatial size of each patch.

    Returns
    -------
    loss : scalar tensor, mean MSE over unmasked patches only
    """
    import torch.nn.functional as F

    # Per-pixel MSE
    loss = (predictions - targets) ** 2  # (N, C, H, W)
    # Pool to patch level: average over channels then spatial pooling
    loss = F.avg_pool2d(loss.mean(dim=1, keepdim=True), patch_size).flatten(1)  # (N, num_patches)
    # Only count unmasked patches
    unmask = 1 - mask  # 0=keep -> 1, 1=remove -> 0
    loss = (loss * unmask).sum(dim=1) / unmask.sum(dim=1).clamp(min=1)  # (N,)
    return loss.mean()
