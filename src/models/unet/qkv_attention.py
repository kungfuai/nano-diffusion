import torch as th
import torch.nn as nn
import math

from src.models.util import count_flops_attn


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, k, v):
        """
        Apply QKV attention.
        :param q: Query tensor of shape [N x (H * C) x T].
        :param k: Key tensor of shape [N x (H * C) x S].
        :param v: Value tensor of shape [N x (H * C) x S].
        :return: Output tensor of shape [N x (H * C) x T].
        """
        bs, width, length_q = q.shape
        _, _, length_kv = k.shape

        assert width % self.n_heads == 0
        ch = width // self.n_heads

        # Reshape and split into heads
        q = q.view(bs * self.n_heads, ch, length_q)
        k = k.view(bs * self.n_heads, ch, length_kv)
        v = v.view(bs * self.n_heads, ch, length_kv)

        # Scaled dot-product attention
        scale = 1 / math.sqrt(ch)
        attn_weights = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # Attention scores
        attn_weights = th.softmax(attn_weights, dim=-1)

        # Compute attention output
        output = th.einsum("bts,bcs->bct", attn_weights, v)

        # Merge heads back
        return output.view(bs, -1, length_q)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
