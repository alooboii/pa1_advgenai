from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from modern_transformer.checkpoint import load_checkpoint, save_checkpoint
from modern_transformer.config import ModelConfig, load_experiment_config
from modern_transformer.data import get_batch
from modern_transformer.model import TransformerLM
from modern_transformer.optim import AdamW, cross_entropy


@pytest.mark.private
def test_no_banned_high_level_implementations() -> None:
    source = "\n".join(path.read_text() for path in Path("src/modern_transformer").glob("*.py"))
    banned = [
        "nn.Linear(",
        "nn.Embedding(",
        "nn.RMSNorm(",
        "F.cross_entropy(",
        "torch.optim.AdamW(",
        "scaled_dot_product_attention(",
        "nn.MultiheadAttention(",
    ]
    for needle in banned:
        assert needle not in source


@pytest.mark.private
def test_instructor_only_original_composite_runs() -> None:
    config = load_experiment_config("configs/original_composite.yaml")
    model = TransformerLM(config.model)
    output = model(torch.randint(0, config.model.vocab_size, (1, 8)))
    assert output.shape == (1, 8, config.model.vocab_size)


@pytest.mark.private
def test_checkpoint_restores_exact_next_update(tmp_path: Path) -> None:
    config = ModelConfig(vocab_size=29, max_seq_len=8, d_model=16, n_layers=1, n_q_heads=2, n_kv_heads=1, d_ff=32)
    torch.manual_seed(3)
    model = TransformerLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(99)
    tokens = np.arange(400, dtype=np.uint16) % config.vocab_size

    x, y = get_batch(tokens, 3, 8, "cpu", generator)
    loss = cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoint = tmp_path / "state.pt"
    save_checkpoint(checkpoint, model=model, optimizer=optimizer, step=1, config={"test": True}, data_generator=generator)

    expected_model = copy.deepcopy(model)
    expected_optimizer = AdamW(expected_model.parameters(), lr=1e-3)
    expected_optimizer.load_state_dict(optimizer.state_dict())
    expected_generator = torch.Generator()
    expected_generator.set_state(generator.get_state())
    x_expected, y_expected = get_batch(tokens, 3, 8, "cpu", expected_generator)
    expected_loss = cross_entropy(expected_model(x_expected), y_expected)
    expected_loss.backward()
    expected_optimizer.step()

    restored_model = TransformerLM(config)
    restored_optimizer = AdamW(restored_model.parameters(), lr=1e-3)
    restored_generator = torch.Generator()
    payload = load_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        data_generator=restored_generator,
    )
    assert payload["step"] == 1
    x_actual, y_actual = get_batch(tokens, 3, 8, "cpu", restored_generator)
    actual_loss = cross_entropy(restored_model(x_actual), y_actual)
    actual_loss.backward()
    restored_optimizer.step()
    torch.testing.assert_close(x_actual, x_expected)
    torch.testing.assert_close(actual_loss, expected_loss)
    for actual, expected in zip(restored_model.parameters(), expected_model.parameters()):
        torch.testing.assert_close(actual, expected)


@pytest.mark.private
@pytest.mark.slow
def test_model_can_overfit_one_batch() -> None:
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=17, max_seq_len=8, d_model=32, n_layers=2, n_q_heads=4, n_kv_heads=2, d_ff=64)
    model = TransformerLM(config)
    optimizer = AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    inputs = torch.randint(0, config.vocab_size, (4, 8))
    targets = torch.roll(inputs, shifts=-1, dims=1)
    initial = cross_entropy(model(inputs), targets).item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.15 * initial


@pytest.mark.private
def test_release_builder_removes_private_material_and_compiles() -> None:
    subprocess.run([sys.executable, "scripts/build_student_release.py"], check=True)
    release = Path("dist/student_assignment")
    assert (release / "README.md").exists()
    assert not (release / "instructor_tests").exists()
    assert not any(path.suffix in {".pt", ".bin"} for path in release.rglob("*"))
    subprocess.run([sys.executable, "-m", "compileall", "-q", str(release)], check=True)
