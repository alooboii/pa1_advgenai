from __future__ import annotations

import pytest

from modern_transformer.config import ModelConfig, load_experiment_config


def test_baseline_config_and_fingerprint_are_stable() -> None:
    config = load_experiment_config("configs/baseline.yaml")
    assert config.model.head_dim == 32
    assert config.model.queries_per_kv == 4
    assert config.train.sequence_length == 128
    assert len(config.fingerprint) == 12


def test_config_override() -> None:
    config = load_experiment_config(
        "configs/baseline.yaml",
        ["train.max_steps=12", "train.warmup_steps=2", "train.amp=false"],
    )
    assert config.train.max_steps == 12
    assert config.train.amp is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 250, "n_q_heads": 8},
        {"n_q_heads": 8, "n_kv_heads": 3},
        {"d_model": 252, "n_q_heads": 12, "n_kv_heads": 3},
    ],
)
def test_invalid_head_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)
