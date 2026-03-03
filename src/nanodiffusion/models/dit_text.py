"""
Diffusion Transformer for discrete text (DiT-Text).

A bidirectional transformer that takes masked token sequences and predicts
original tokens. Reuses the AdaLN-Zero conditioning pattern from the image DiT
but adapted for 1D token sequences.

The architecture follows MDLM's DIT:
- Token embedding (instead of patch embedding)
- Sinusoidal positional encoding (1D)
- Transformer blocks with AdaLN-Zero conditioning on timestep
- Output projection to vocabulary logits
"""

import math
import torch
import torch.nn as nn
from .dit import TimestepEmbedder


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TextDiTBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero conditioning for text.
    Uses standard multi-head self-attention (bidirectional).
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        # AdaLN modulation: 6 vectors (shift, scale, gate for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings [B, L, D]
            c: Conditioning vector [B, D] (from timestep embedding)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # Self-attention with AdaLN
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h

        # MLP with AdaLN
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class TextDiTFinalLayer(nn.Module):
    """Final layer: AdaLN + linear projection to vocabulary logits."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class DiTText(nn.Module):
    """
    Diffusion Transformer for discrete text generation.

    Takes masked token IDs and a timestep, returns logits over the vocabulary.

    Args:
        vocab_size: Size of the vocabulary (including mask token)
        max_seq_length: Maximum sequence length
        hidden_size: Transformer hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim = hidden_size * mlp_ratio
        dropout: Dropout rate
        time_conditioning: Whether to condition on timestep via AdaLN.
            If False, the model infers time from the number of masked tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int = 256,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        time_conditioning: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.time_conditioning = time_conditioning

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_size)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_length, hidden_size)
        )

        # Timestep embedding (reuse DiT's sinusoidal + MLP)
        if time_conditioning:
            self.t_embedder = TimestepEmbedder(hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TextDiTBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Output projection
        self.final_layer = TextDiTFinalLayer(hidden_size, vocab_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Xavier init for linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize token embedding
        nn.init.normal_(self.token_embed.weight, std=0.02)

        # Initialize positional embedding with sinusoidal pattern
        pos = torch.arange(self.max_seq_length, dtype=torch.float).unsqueeze(1)
        dim = torch.arange(0, self.hidden_size, 2, dtype=torch.float)
        pe = torch.zeros(self.max_seq_length, self.hidden_size)
        pe[:, 0::2] = torch.sin(pos / (10000 ** (dim / self.hidden_size)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (dim / self.hidden_size)))
        self.pos_embed.data.copy_(pe.unsqueeze(0))

        # Initialize timestep embedding
        if self.time_conditioning:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Token IDs [B, L] (may contain mask tokens)
            t: Timestep values [B] in range [0, 1]

        Returns:
            Logits over vocabulary [B, L, V]
        """
        B, L = x.shape

        # Token + positional embedding
        h = self.token_embed(x) + self.pos_embed[:, :L, :]

        # Timestep conditioning
        if self.time_conditioning:
            # Scale t to a larger range for the sinusoidal embedder
            c = self.t_embedder(t * 1000)  # [B, D]
        else:
            # Zero conditioning (model infers time from mask ratio)
            c = torch.zeros(B, self.hidden_size, device=x.device)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, c)

        # Output projection
        logits = self.final_layer(h, c)  # [B, L, V]

        return logits


# ─── Configuration presets ───────────────────────────────────────────────────

def DiTText_T(vocab_size, max_seq_length=256, **kwargs):
    """Tiny: ~2M params, good for testing."""
    return DiTText(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        hidden_size=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        **kwargs,
    )


def DiTText_S(vocab_size, max_seq_length=256, **kwargs):
    """Small: ~25M params."""
    return DiTText(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        **kwargs,
    )


def DiTText_B(vocab_size, max_seq_length=256, **kwargs):
    """Base: ~110M params."""
    return DiTText(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs,
    )
