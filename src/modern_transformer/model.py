from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .layers import (
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
    ReLUFFN,
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
    SwiGLU,
)


def _stable_softmax(x: Tensor, dim: int = -1) -> Tensor:
    shifted = x - x.amax(dim=dim, keepdim=True)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_q_heads = config.n_q_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.queries_per_kv = config.queries_per_kv
        # BEGIN SOLUTION
        self.q_proj = Linear(config.d_model, config.n_q_heads * config.head_dim, init_std=config.init_std)
        self.k_proj = Linear(config.d_model, config.n_kv_heads * config.head_dim, init_std=config.init_std)
        self.v_proj = Linear(config.d_model, config.n_kv_heads * config.head_dim, init_std=config.init_std)
        self.out_proj = Linear(config.n_q_heads * config.head_dim, config.d_model, init_std=config.init_std)
        self.rope = (
            RotaryEmbedding(config.rope_theta, config.head_dim, config.max_seq_len)
            if config.position_style == "rope"
            else None
        )
        # END SOLUTION

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        # BEGIN SOLUTION
        if x.ndim != 3:
            raise ValueError("attention input must have shape [batch, sequence, d_model]")
        batch, sequence, _ = x.shape
        if sequence > self.config.max_seq_len:
            raise ValueError("input exceeds max_seq_len")
        if positions is None:
            positions = torch.arange(sequence, device=x.device)

        q = self.q_proj(x).view(batch, sequence, self.n_kv_heads, self.queries_per_kv, self.head_dim)
        k = self.k_proj(x).view(batch, sequence, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, sequence, self.n_kv_heads, self.head_dim)

        # [batch, kv_head, query_group, sequence, head_dim]
        q = q.permute(0, 2, 3, 1, 4)
        # [batch, kv_head, sequence, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.rope is not None:
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        scores = torch.einsum("bhgtd,bhsd->bhgts", q, k) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(sequence, sequence, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal_mask, -torch.inf)
        probabilities = _stable_softmax(scores, dim=-1)
        grouped = torch.einsum("bhgts,bhsd->bhgtd", probabilities, v)
        merged = grouped.permute(0, 3, 1, 2, 4).reshape(batch, sequence, self.config.d_model)
        return self.out_proj(merged)
        # END SOLUTION


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.attention = CausalSelfAttention(config)
        if config.ffn_style == "swiglu":
            self.ffn: nn.Module = SwiGLU(config.d_model, config.d_ff, init_std=config.init_std)
        else:
            self.ffn = ReLUFFN(config.d_model, config.d_ff, init_std=config.init_std)
        if config.norm_style == "pre_rms":
            self.attention_norm: nn.Module = RMSNorm(config.d_model, config.norm_eps)
            self.ffn_norm: nn.Module = RMSNorm(config.d_model, config.norm_eps)
        else:
            self.attention_norm = LayerNorm(config.d_model, config.norm_eps)
            self.ffn_norm = LayerNorm(config.d_model, config.norm_eps)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        # BEGIN SOLUTION
        if self.config.norm_style == "pre_rms":
            x = x + self.attention(self.attention_norm(x), positions)
            x = x + self.ffn(self.ffn_norm(x))
            return x
        x = self.attention_norm(x + self.attention(x, positions))
        return self.ffn_norm(x + self.ffn(x))
        # END SOLUTION


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # BEGIN SOLUTION
        self.token_embedding = Embedding(config.vocab_size, config.d_model, init_std=config.init_std)
        self.position_embedding = (
            SinusoidalPositionalEncoding(config.d_model, config.max_seq_len, config.rope_theta)
            if config.position_style == "sinusoidal"
            else None
        )
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm: nn.Module = (
            RMSNorm(config.d_model, config.norm_eps) if config.norm_style == "pre_rms" else nn.Identity()
        )
        self.lm_head = Linear(config.d_model, config.vocab_size, init_std=config.init_std)
        # END SOLUTION

    def forward(self, token_ids: Tensor) -> Tensor:
        # BEGIN SOLUTION
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        sequence = token_ids.shape[1]
        if sequence > self.config.max_seq_len:
            raise ValueError("input exceeds max_seq_len")
        positions = torch.arange(sequence, device=token_ids.device)
        x = self.token_embedding(token_ids)
        if self.position_embedding is not None:
            x = x + self.position_embedding(positions, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, positions)
        return self.lm_head(self.final_norm(x))
        # END SOLUTION

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def kv_cache_bytes(
    config: ModelConfig,
    *,
    batch_size: int,
    sequence_length: int,
    bytes_per_element: int = 2,
) -> int:
    """Analytic K+V cache size for all layers, without allocating a cache."""
    return (
        2
        * config.n_layers
        * batch_size
        * sequence_length
        * config.n_kv_heads
        * config.head_dim
        * bytes_per_element
    )
