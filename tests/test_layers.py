from __future__ import annotations

import torch

from modern_transformer.layers import Embedding, Linear, RMSNorm, SwiGLU


def test_linear_matches_explicit_matrix_multiply() -> None:
    layer = Linear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]))
    x = torch.tensor([[[2.0, -1.0, 0.5]]], requires_grad=True)
    expected = x @ layer.weight.T
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert x.grad is not None


def test_embedding_lookup_and_gradient() -> None:
    embedding = Embedding(7, 3)
    token_ids = torch.tensor([[1, 4], [4, 2]])
    output = embedding(token_ids)
    assert output.shape == (2, 2, 3)
    output.sum().backward()
    assert embedding.weight.grad is not None
    assert embedding.weight.grad[0].abs().sum() == 0
    torch.testing.assert_close(embedding.weight.grad[4], torch.full((3,), 2.0))


def test_rmsnorm_matches_float64_reference_and_preserves_dtype() -> None:
    torch.manual_seed(0)
    layer = RMSNorm(6, eps=1e-5).to(dtype=torch.float16)
    x = torch.randn(2, 3, 6, dtype=torch.float16)
    actual = layer(x)
    reference = x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-5)
    reference = reference * layer.weight.double()
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual.float(), reference.float(), rtol=2e-3, atol=2e-3)


def test_swiglu_formula_and_parameter_count() -> None:
    layer = SwiGLU(4, 6)
    x = torch.randn(2, 3, 4)
    gate = x @ layer.gate.weight.T
    expected = ((gate * torch.sigmoid(gate)) * (x @ layer.up.weight.T)) @ layer.down.weight.T
    torch.testing.assert_close(layer(x), expected)
    assert sum(p.numel() for p in layer.parameters()) == 3 * 4 * 6

