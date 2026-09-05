from __future__ import annotations

import torch

from modern_transformer.config import ModelConfig
from modern_transformer.generation import generate
from modern_transformer.model import TransformerLM


@torch.no_grad()
def test_greedy_generation_shape_and_determinism() -> None:
    model = TransformerLM(ModelConfig(vocab_size=23, max_seq_len=12, d_model=16, n_layers=1, n_q_heads=2, n_kv_heads=1, d_ff=32))
    prompt = torch.tensor([[1, 2, 3]])
    first = generate(model, prompt, max_new_tokens=5, temperature=0)
    second = generate(model, prompt, max_new_tokens=5, temperature=0)
    assert first.shape == (1, 8)
    torch.testing.assert_close(first, second)

