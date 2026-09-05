from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """Bias-free linear map storing weights as [out_features, in_features]."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        # BEGIN SOLUTION
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        # END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN SOLUTION
        return x @ self.weight.transpose(-1, -2)
        # END SOLUTION


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        # BEGIN SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        # END SOLUTION

    def forward(self, token_ids: Tensor) -> Tensor:
        # BEGIN SOLUTION
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("embedding indices must be integer tensors")
        return self.weight[token_ids]
        # END SOLUTION


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        # BEGIN SOLUTION
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        # END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN SOLUTION
        input_dtype = x.dtype
        x_float = x.float()
        inverse_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * inverse_rms * self.weight.float()).to(input_dtype)
        # END SOLUTION


class LayerNorm(nn.Module):
    """Provided original-style baseline; students do not implement this class."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        variance = (xf - mean).square().mean(dim=-1, keepdim=True)
        normalized = (xf - mean) * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float() + self.bias.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    """Adjacent-pair rotary embedding with cached sine and cosine tables."""

    def __init__(self, theta: float, head_dim: int, max_seq_len: int) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        self.theta = theta
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        # BEGIN SOLUTION
        pair_ids = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inverse_frequencies = theta ** (-pair_ids / head_dim)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)
        # END SOLUTION

    def apply(self, x: Tensor, positions: Tensor) -> Tensor:  # type: ignore[override]
        """Rotate an input whose final axes are [..., sequence, head_dim]."""
        # BEGIN SOLUTION
        if x.shape[-1] != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}, got {x.shape[-1]}")
        if positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("positions must be an integer tensor")
        if positions.numel() and (positions.min() < 0 or positions.max() >= self.max_seq_len):
            raise ValueError("position is outside the configured RoPE cache")
        cos = self.cos_cached[positions].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[positions].to(dtype=x.dtype, device=x.device)
        position_batch_dims = positions.ndim - 1
        input_batch_dims = x.ndim - 2
        if position_batch_dims > input_batch_dims:
            raise ValueError("positions have more batch dimensions than x")
        singleton_dims = (1,) * (input_batch_dims - position_batch_dims)
        broadcast_shape = (*positions.shape[:-1], *singleton_dims, positions.shape[-1], self.head_dim // 2)
        cos = cos.reshape(broadcast_shape)
        sin = sin.reshape(broadcast_shape)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
        # END SOLUTION

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        return self.apply(x, positions)


class SinusoidalPositionalEncoding(nn.Module):
    """Provided parameter-free additive positional baseline."""

    def __init__(self, d_model: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        dimensions = torch.arange(0, d_model, 2, dtype=torch.float32)
        inverse_frequencies = theta ** (-dimensions / d_model)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]
        table = torch.zeros(max_seq_len, d_model)
        table[:, 0::2] = angles.sin()
        table[:, 1::2] = angles.cos()
        self.register_buffer("table", table, persistent=False)

    def forward(self, positions: Tensor, *, dtype: torch.dtype) -> Tensor:
        return self.table[positions].to(dtype=dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, *, init_std: float = 0.02) -> None:
        super().__init__()
        # BEGIN SOLUTION
        self.gate = Linear(d_model, d_ff, init_std=init_std)
        self.up = Linear(d_model, d_ff, init_std=init_std)
        self.down = Linear(d_ff, d_model, init_std=init_std)
        # END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN SOLUTION
        gate = self.gate(x)
        silu_gate = gate * torch.sigmoid(gate)
        return self.down(silu_gate * self.up(x))
        # END SOLUTION


class ReLUFFN(nn.Module):
    """Provided original-style, approximately parameter-matched FFN baseline."""

    def __init__(self, d_model: int, d_ff: int, *, init_std: float = 0.02) -> None:
        super().__init__()
        self.up = Linear(d_model, d_ff, init_std=init_std)
        self.down = Linear(d_ff, d_model, init_std=init_std)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(torch.clamp_min(self.up(x), 0.0))


def swiglu_width(d_model: int, multiple_of: int = 64) -> int:
    """Round 8/3 d_model upward to a hardware-friendly multiple."""
    return math.ceil((8 * d_model / 3) / multiple_of) * multiple_of
