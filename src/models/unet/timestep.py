from abc import abstractmethod
import torch.nn as nn

from src.models.unet.cross_attention_block import CrossAttentionBlock


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, CrossAttentionBlock):
                x = layer(x, context)
            else:
                print(f"layer: {layer}")
                x = layer(x)
        return x
