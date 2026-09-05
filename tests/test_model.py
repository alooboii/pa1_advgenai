from __future__ import annotations

import torch

from modern_transformer.config import ModelConfig, load_experiment_config
from modern_transformer.model import TransformerLM, kv_cache_bytes


def test_model_forward_backward() -> None:
    config = ModelConfig(vocab_size=101, max_seq_len=16, d_model=32, n_layers=2, n_q_heads=4, n_kv_heads=2, d_ff=64)
    model = TransformerLM(config)
    tokens = torch.randint(0, 101, (3, 12))
    logits = model(tokens)
    assert logits.shape == (3, 12, 101)
    logits.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_baseline_parameter_count() -> None:
    config = load_experiment_config("configs/baseline.yaml")
    assert TransformerLM(config.model).parameter_count() == 4_917_504


def test_all_architecture_variants_run() -> None:
    for path in (
        "configs/ablation_post_layernorm.yaml",
        "configs/ablation_sinusoidal.yaml",
        "configs/ablation_relu.yaml",
        "configs/ablation_mha.yaml",
    ):
        config = load_experiment_config(path)
        logits = TransformerLM(config.model)(torch.randint(0, config.model.vocab_size, (1, 8)))
        assert logits.shape == (1, 8, config.model.vocab_size)
        logits.mean().backward()


def test_kv_cache_accounting() -> None:
    config = ModelConfig(d_model=256, n_layers=4, n_q_heads=8, n_kv_heads=2)
    assert kv_cache_bytes(config, batch_size=1, sequence_length=128, bytes_per_element=2) == 131_072
