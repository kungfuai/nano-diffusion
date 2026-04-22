"""
Masked Diffusion Language Model (MDLM) for discrete text generation.

Implements absorbing-state diffusion where tokens are progressively masked.
The model learns to predict original tokens from masked sequences.

Key references:
- MDLM (NeurIPS 2024): https://github.com/kuleshov-group/mdlm
- dLLM: https://github.com/ZHZisZZ/dllm
- LLaDA: https://github.com/ML-GSAI/LLaDA
"""

from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseDiffusionAlgorithm


class LinearNoiseSchedule:
    """
    Linear noise schedule: mask_prob(t) = t.

    This is equivalent to the log-linear schedule in MDLM:
      sigma(t) = -log(1 - t)
      alpha(t) = 1 - t
      mask_prob(t) = t
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Signal retention probability: alpha(t) = 1 - t."""
        return 1 - t

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Probability of masking each token at time t."""
        return t

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Loss weight w(t) = 1 / t for the SUBS parameterization.

        From the continuous-time ELBO: w(t) = -alpha'(t) / (1 - alpha(t)) = 1/t.
        We clamp to avoid division by zero near t=0.
        """
        return 1.0 / t.clamp(min=self.eps)


class MDLMForwardProcess:
    """Forward masking process: independently mask each token with probability t."""

    def __init__(self, mask_token_id: int):
        self.mask_token_id = mask_token_id

    def mask_tokens(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply forward masking process.

        Args:
            x_0: Clean token IDs [B, L]
            t: Masking probability for each example [B] or [B, 1]

        Returns:
            x_t: Masked token IDs [B, L]
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [B, 1]

        # Independently mask each token with probability t
        mask = torch.rand_like(x_0, dtype=torch.float) < t
        x_t = torch.where(mask, self.mask_token_id, x_0)
        return x_t


class MDLMSampler:
    """
    Sampling algorithms for MDLM.

    Supports two strategies:
    - 'ddpm_cache': Stochastic posterior sampling (from MDLM paper)
    - 'topk': Confidence-based top-k unmasking (from LLaDA)
    """

    def __init__(
        self,
        model: nn.Module,
        mask_token_id: int,
        device: str = "cuda:0",
        schedule: Optional[LinearNoiseSchedule] = None,
    ):
        self.model = model
        self.mask_token_id = mask_token_id
        self.device = device
        self.schedule = schedule or LinearNoiseSchedule()

    @torch.no_grad()
    def sample_ddpm(
        self,
        batch_size: int,
        seq_length: int,
        num_steps: int = 128,
        temperature: float = 1.0,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """
        DDPM-style sampling with caching.

        Starts from fully masked sequence and iteratively unmasks tokens
        using the posterior distribution.

        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of each sequence
            num_steps: Number of denoising steps
            temperature: Sampling temperature (1.0 = standard, 0.0 = greedy)
            eps: Minimum time value

        Returns:
            Generated token IDs [B, L]
        """
        device = self.device

        # Start from fully masked sequence
        x = torch.full(
            (batch_size, seq_length), self.mask_token_id, dtype=torch.long, device=device
        )

        timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)

        for i in range(num_steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            # Get model predictions (log-probabilities over vocabulary)
            t_batch = t_cur.expand(batch_size)
            logits = self.model(x, t_batch)  # [B, L, V]

            # Set mask token logit to -inf (never predict mask)
            logits[:, :, self.mask_token_id] = float("-inf")

            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
            else:
                # Greedy: one-hot at argmax
                probs = torch.zeros_like(logits)
                probs.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)

            # Posterior probability computation
            mask_prob_t = t_cur
            mask_prob_s = t_next
            unmask_prob = (mask_prob_t - mask_prob_s) / mask_prob_t

            # For each masked position, decide whether to unmask
            is_masked = x == self.mask_token_id
            unmask_decision = torch.rand_like(x, dtype=torch.float) < unmask_prob

            # Sample new tokens for positions that get unmasked
            new_tokens = torch.multinomial(
                probs.view(-1, probs.size(-1)), num_samples=1
            ).view(batch_size, seq_length)

            # Apply: unmask where both (currently masked) and (decided to unmask)
            should_unmask = is_masked & unmask_decision
            x = torch.where(should_unmask, new_tokens, x)

        # Final step: unmask any remaining masked positions with argmax
        is_masked = x == self.mask_token_id
        if is_masked.any():
            t_batch = torch.full((batch_size,), eps, device=device)
            logits = self.model(x, t_batch)
            logits[:, :, self.mask_token_id] = float("-inf")
            final_tokens = logits.argmax(dim=-1)
            x = torch.where(is_masked, final_tokens, x)

        return x

    @torch.no_grad()
    def sample_topk(
        self,
        batch_size: int,
        seq_length: int,
        num_steps: int = 128,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
    ) -> torch.Tensor:
        """
        Top-k confidence-based sampling (LLaDA-style).

        At each step, predict all tokens, then unmask the most confident ones.

        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of each sequence
            num_steps: Number of denoising steps
            temperature: Gumbel noise temperature (0 = greedy)
            remasking: Strategy for choosing which tokens to unmask
                       ('low_confidence' or 'random')

        Returns:
            Generated token IDs [B, L]
        """
        device = self.device

        # Start from fully masked
        x = torch.full(
            (batch_size, seq_length), self.mask_token_id, dtype=torch.long, device=device
        )

        # Precompute how many tokens to unmask at each step
        tokens_per_step = _compute_unmask_schedule(seq_length, num_steps)

        for i in range(num_steps):
            is_masked = x == self.mask_token_id
            num_masked = is_masked.sum(dim=1)

            if num_masked.max() == 0:
                break  # All tokens unmasked

            # Compute masking ratio for model conditioning
            t_batch = num_masked.float() / seq_length
            logits = self.model(x, t_batch)  # [B, L, V]
            logits[:, :, self.mask_token_id] = float("-inf")

            # Sample predicted tokens
            if temperature > 0:
                # Gumbel-Max trick
                gumbel_noise = -torch.log(
                    -torch.log(torch.rand_like(logits, dtype=torch.float64) + 1e-20) + 1e-20
                )
                noisy_logits = logits.double() + temperature * gumbel_noise
                predicted = noisy_logits.argmax(dim=-1)
            else:
                predicted = logits.argmax(dim=-1)

            # Compute confidence scores
            if remasking == "low_confidence":
                probs = F.softmax(logits, dim=-1)
                confidence = probs.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
            else:  # random
                confidence = torch.rand(batch_size, seq_length, device=device)

            # Only consider masked positions
            confidence = torch.where(is_masked, confidence, torch.tensor(-1.0, device=device))

            # Select top-k most confident predictions to unmask
            k = min(tokens_per_step[i], num_masked.min().item())
            if k > 0:
                _, topk_indices = confidence.topk(k, dim=1)
                unmask_mask = torch.zeros_like(is_masked)
                unmask_mask.scatter_(1, topk_indices, True)
                unmask_mask = unmask_mask & is_masked

                x = torch.where(unmask_mask, predicted, x)

        # Final pass: unmask any remaining
        is_masked = x == self.mask_token_id
        if is_masked.any():
            t_batch = is_masked.float().sum(dim=1) / seq_length
            logits = self.model(x, t_batch)
            logits[:, :, self.mask_token_id] = float("-inf")
            x = torch.where(is_masked, logits.argmax(dim=-1), x)

        return x

    def sample(
        self,
        batch_size: int,
        seq_length: int,
        num_steps: int = 128,
        strategy: str = "ddpm_cache",
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Sample sequences using the specified strategy."""
        if strategy == "ddpm_cache":
            return self.sample_ddpm(batch_size, seq_length, num_steps, temperature)
        elif strategy == "topk":
            return self.sample_topk(batch_size, seq_length, num_steps, temperature, **kwargs)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")


class MDLM(BaseDiffusionAlgorithm):
    """
    Masked Diffusion Language Model.

    Implements the SUBS parameterization from the MDLM paper, where the
    continuous-time ELBO simplifies to a reweighted masked language modeling loss.

    Args:
        model: The denoising model that takes (token_ids, t) and returns logits [B, L, V]
        mask_token_id: ID of the mask token in the vocabulary
        vocab_size: Size of the vocabulary (excluding mask token if mask_token_id == vocab_size)
        seq_length: Fixed sequence length for generation
        device: Device to use
        sampling_eps: Minimum time value to avoid edge cases at t=0
        antithetic_sampling: Use stratified time sampling for lower variance
    """

    def __init__(
        self,
        model: nn.Module,
        mask_token_id: int,
        vocab_size: int,
        seq_length: int = 128,
        device: str = "cuda:0",
        sampling_eps: float = 1e-3,
        antithetic_sampling: bool = True,
    ):
        self.model = model
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.device = device
        self.sampling_eps = sampling_eps
        self.antithetic_sampling = antithetic_sampling

        self.schedule = LinearNoiseSchedule(eps=sampling_eps)
        self.forward_process = MDLMForwardProcess(mask_token_id)
        self.sampler = MDLMSampler(
            model=model,
            mask_token_id=mask_token_id,
            device=device,
            schedule=self.schedule,
        )

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps with optional antithetic (stratified) sampling.

        Returns t in [eps, 1] for each example in the batch.
        """
        eps_t = torch.rand(batch_size, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(batch_size, device=device).float() / batch_size
            eps_t = (eps_t / batch_size + offset) % 1.0
        # Map to [sampling_eps, 1]
        t = self.sampling_eps + (1 - self.sampling_eps) * eps_t
        return t

    def compute_loss(
        self,
        x_0: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the MDLM SUBS loss.

        This is a reweighted cross-entropy loss computed only at masked positions:
            L = E_t [ w(t) * sum_{masked positions} -log p(x_0_i | x_t) ]

        Args:
            x_0: Clean token IDs [B, L]
            attention_mask: Binary mask for valid positions [B, L] (1 = valid, 0 = padding)

        Returns:
            Scalar loss value
        """
        device = x_0.device
        batch_size, seq_length = x_0.shape

        if attention_mask is None:
            attention_mask = torch.ones_like(x_0, dtype=torch.float)

        # 1. Sample random timesteps
        t = self._sample_t(batch_size, device)  # [B]

        # 2. Forward process: mask tokens
        x_t = self.forward_process.mask_tokens(x_0, t)  # [B, L]

        # 3. Model prediction: get logits
        logits = self.model(x_t, t)  # [B, L, V]

        # 4. Apply SUBS parameterization
        # Set mask token logit to -inf so model never predicts [MASK]
        logits[:, :, self.mask_token_id] = float("-inf")

        # 5. Compute per-token cross-entropy loss
        # logits: [B, L, V], x_0: [B, L]
        log_probs = F.log_softmax(logits, dim=-1)
        token_nll = -log_probs.gather(
            dim=-1, index=x_0.unsqueeze(-1)
        ).squeeze(-1)  # [B, L]

        # 6. Only count loss at masked positions
        is_masked = (x_t == self.mask_token_id).float()

        # 7. Apply loss weight w(t) = 1/t
        loss_weight = self.schedule.loss_weight(t)  # [B]
        weighted_nll = token_nll * is_masked * loss_weight.unsqueeze(1) * attention_mask

        # 8. Normalize by number of valid tokens
        num_valid = (attention_mask).sum().clamp(min=1)
        loss = weighted_nll.sum() / num_valid

        return loss

    def prepare_step_supervision(self, batch, **kwargs):
        """
        Interface for compatibility with the existing training loop.

        For MDLM, we override the standard diffusion training to compute
        the loss directly rather than separating into inputs/targets.
        Returns (inputs_dict, None) where inputs_dict contains everything
        needed for loss computation.
        """
        if isinstance(batch, dict):
            x_0 = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", torch.ones_like(x_0, dtype=torch.float)).to(self.device)
        elif hasattr(batch, "x"):
            x_0 = batch.x.to(self.device).long()
            attention_mask = torch.ones_like(x_0, dtype=torch.float)
        else:
            x_0 = batch.to(self.device).long()
            attention_mask = torch.ones_like(x_0, dtype=torch.float)

        return {"x_0": x_0, "attention_mask": attention_mask}, None

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 4,
        seq_length: Optional[int] = None,
        num_steps: int = 128,
        strategy: str = "ddpm_cache",
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text token sequences."""
        seq_length = seq_length or self.seq_length
        return self.sampler.sample(
            batch_size=batch_size,
            seq_length=seq_length,
            num_steps=num_steps,
            strategy=strategy,
            temperature=temperature,
            **kwargs,
        )


def _compute_unmask_schedule(seq_length: int, num_steps: int) -> List[int]:
    """
    Compute how many tokens to unmask at each step for top-k sampling.

    Distributes seq_length tokens evenly across num_steps steps.
    """
    tokens_per_step = []
    remaining = seq_length
    for i in range(num_steps):
        steps_left = num_steps - i
        k = remaining // steps_left
        tokens_per_step.append(k)
        remaining -= k
    return tokens_per_step
