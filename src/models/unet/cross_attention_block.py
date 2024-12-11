import torch.nn as nn
import xformers

from src.models.unet.qkv_attention import QKVAttention, QKVAttentionLegacy
from src.models.util import (
    conv_nd,
    zero_module,
    normalization,
    checkpoint,
)


class CrossAttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.q_linear = conv_nd(1, channels, channels, 1)  # Query projection
        self.kv_linear = nn.Linear(channels, channels * 2)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), True)

    def _forward(self, x, context=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # Flatten spatial dimensions

        # context = context.reshape(b, c, -1)  # Flatten context if not already
        q = self.q_linear(self.norm(x))  # Query
        print(f"context.shape: {context.shape}")
        # print(f"Norm context.shape: {self.norm(context).shape}")
        kv = self.kv_linear(self.norm(context))  # Key and Value
        # kv = self.kv_linear(context)  # Key and Value
        k, v = kv.chunk(2, dim=1)  # Split Key and Value

        # Reshape for multi-head attention
        bs, width, length = q.shape
        q = q.reshape(bs * self.num_heads, width // self.num_heads, length)
        k = k.reshape(bs * self.num_heads, width // self.num_heads, length)
        v = v.reshape(bs * self.num_heads, width // self.num_heads, length)

        h = self.attention(q, k, v)  # Attention
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
