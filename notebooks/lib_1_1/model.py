import math
import torch
import torch.nn as nn
from typing import List


class Model1(nn.Module):
    """
    Basic model that concatenates time with input directly.
    """
    def __init__(self, hidden_features: List[int], dim_in: int = 2, dim_out: int = 2, num_timesteps: int = 1000):
        """
        Args:
            hidden_features: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        input_dim = dim_in + 1  # The input dimension (t and x combined)
        self.num_timesteps = num_timesteps
        
        for hidden_dim in hidden_features:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, dim_out))
        layers.append(nn.Tanh())
        
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            t: Timestep tensor
            x: Input tensor
            
        Returns:
            Model output
        """
        t = t / self.num_timesteps
        t = t.reshape(-1, 1)
        return self.net(torch.cat([t, x], 1))


class MLP(nn.Sequential):
    """
    Helper class for Model2 implementing a basic MLP.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int],
    ):
        layers = []
        for a, b in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            layers.extend([nn.Linear(a, b), nn.ELU()])
        super().__init__(*layers[:-1])


class Model2(nn.Module):
    """
    Model using sinusoidal time embeddings.
    """
    def __init__(self, dim_in: int = 2, dim_out: int = 2, num_timesteps: int = 1000, freqs: int = 16, hidden_features: List[int] = None):
        """
        Args:
            features: Number of input/output features
            freqs: Number of frequency components for time embedding
            hidden_features: List of hidden layer dimensions
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        if hidden_features is None:
            hidden_features = [512, 512]

        self.net = MLP(2 * freqs + dim_in, dim_out, hidden_features)
        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            t: Timestep tensor
            x: Input tensor
            
        Returns:
            Model output
        """
        t = t / self.num_timesteps
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(*x.shape[:-1], -1)
        return self.net(torch.cat((t, x), dim=-1))


class TimeEmbedding(nn.Module):
    """
    Time embedding module for Model3.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: Input timesteps
            dim: Desired dimension of the embeddings
            max_period: Maximum period of the embedding
            
        Returns:
            Time embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class TimeLinear(nn.Module):
    """
    Time-modulated linear layer for Model3.
    """
    def __init__(self, dim_in: int, dim_out: int, num_timesteps: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_timesteps = num_timesteps
        self.time_embedding = TimeEmbedding(dim_out)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        alpha = self.time_embedding(t).view(-1, self.dim_out)
        return alpha * x


class Model3(nn.Module):
    """
    Model using time-modulated linear layers.
    """
    def __init__(
        self, 
        hidden_features: List[int], 
        num_timesteps: int, 
        dim_in: int = 2, 
        dim_out: int = 2,
    ):
        super().__init__()
        layers = []
        dims = [dim_in] + hidden_features

        # Build MLP layers with time-dependent linear layers
        for i in range(len(dims)-1):
            layers.append(TimeLinear(dims[i], dims[i+1], num_timesteps))
            layers.append(nn.ReLU())

        # Final layer to output noise prediction
        layers.append(TimeLinear(dims[-1], dim_out, num_timesteps))

        self.net = nn.Sequential(*layers)
        self.num_timesteps = num_timesteps

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            if isinstance(layer, TimeLinear):
                x = layer(x, t)
            else:
                x = layer(x)
        return x
