from __future__ import annotations

import torch

from modern_transformer.config import ModelConfig
from modern_transformer.layers import RotaryEmbedding
from modern_transformer.model import CausalSelfAttention


def test_rope_position_zero_and_norm_preservation() -> None:
    rope = RotaryEmbedding(theta=10_000.0, head_dim=8, max_seq_len=32)
    x = torch.randn(2, 3, 5, 8)
    positions = torch.arange(5)
    rotated = rope(x, positions)
    torch.testing.assert_close(rotated[..., 0, :], x[..., 0, :])
    torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1), rtol=1e-5, atol=1e-5)


def test_rope_dot_product_depends_on_relative_position() -> None:
    rope = RotaryEmbedding(theta=10_000.0, head_dim=8, max_seq_len=64)
    q = torch.randn(1, 1, 8)
    k = torch.randn(1, 1, 8)
    score_a = (rope(q, torch.tensor([7])) * rope(k, torch.tensor([19]))).sum()
    score_b = (rope(q, torch.tensor([13])) * rope(k, torch.tensor([25]))).sum()
    torch.testing.assert_close(score_a, score_b, rtol=2e-5, atol=2e-5)


def test_rope_supports_batched_position_ids_and_head_axis() -> None:
    rope = RotaryEmbedding(theta=10_000.0, head_dim=8, max_seq_len=32)
    x = torch.randn(2, 3, 4, 8)
    positions = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    actual = rope(x, positions)
    expected = torch.stack(
        [torch.stack([rope(x[b, h], positions[b]) for h in range(3)]) for b in range(2)]
    )
    torch.testing.assert_close(actual, expected)


def _explicit_repeat_reference(module: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    batch, sequence, _ = x.shape
    hq, hkv, dh = module.n_q_heads, module.n_kv_heads, module.head_dim
    q = module.q_proj(x).view(batch, sequence, hq, dh).permute(0, 2, 1, 3)
    k = module.k_proj(x).view(batch, sequence, hkv, dh).permute(0, 2, 1, 3)
    v = module.v_proj(x).view(batch, sequence, hkv, dh).permute(0, 2, 1, 3)
    positions = torch.arange(sequence)
    if module.rope is not None:
        q = module.rope(q, positions)
        k = module.rope(k, positions)
    k = k.repeat_interleave(module.queries_per_kv, dim=1)
    v = v.repeat_interleave(module.queries_per_kv, dim=1)
    scores = torch.einsum("bhtd,bhsd->bhts", q, k) / dh**0.5
    mask = torch.ones(sequence, sequence, dtype=torch.bool).tril()
    probabilities = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
    output = torch.einsum("bhts,bhsd->bhtd", probabilities, v)
    return module.out_proj(output.permute(0, 2, 1, 3).reshape(batch, sequence, -1))


@torch.no_grad()
def test_gqa_matches_explicit_kv_repeat() -> None:
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=31, max_seq_len=16, d_model=32, n_layers=1, n_q_heads=4, n_kv_heads=2, d_ff=64)
    attention = CausalSelfAttention(config)
    x = torch.randn(2, 7, 32)
    torch.testing.assert_close(attention(x), _explicit_repeat_reference(attention, x), rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_attention_is_strictly_causal() -> None:
    config = ModelConfig(vocab_size=31, max_seq_len=16, d_model=32, n_layers=1, n_q_heads=4, n_kv_heads=1, d_ff=64)
    attention = CausalSelfAttention(config)
    x = torch.randn(1, 8, 32)
    changed = x.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 20
    torch.testing.assert_close(attention(x)[:, :5], attention(changed)[:, :5], rtol=1e-5, atol=1e-5)


def test_mha_and_mqa_boundary_shapes() -> None:
    x = torch.randn(2, 5, 32)
    for kv_heads in (1, 4):
        config = ModelConfig(vocab_size=31, max_seq_len=8, d_model=32, n_layers=1, n_q_heads=4, n_kv_heads=kv_heads, d_ff=64)
        assert CausalSelfAttention(config)(x).shape == x.shape
