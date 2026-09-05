from __future__ import annotations

import math

import numpy as np
import torch

from modern_transformer.data import get_batch
from modern_transformer.optim import AdamW, cross_entropy


def test_cross_entropy_matches_reference_for_large_logits() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 11, dtype=torch.float64) * 1000
    targets = torch.randint(0, 11, (2, 3))
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    torch.testing.assert_close(cross_entropy(logits, targets), expected)


def test_adamw_one_step_matches_assignment_equations() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float64)
    optimizer = AdamW([parameter], lr=0.1, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.2)
    optimizer.step()
    gradient = torch.tensor([0.25, -0.5], dtype=torch.float64)
    m = 0.1 * gradient
    v = 0.05 * gradient.square()
    decayed = torch.tensor([1.0, -2.0], dtype=torch.float64) * (1 - 0.1 * 0.2)
    step_size = 0.1 * math.sqrt(1 - 0.95) / (1 - 0.9)
    expected = decayed - step_size * m / (v.sqrt() + 1e-8)
    torch.testing.assert_close(parameter, expected)


def test_get_batch_is_deterministic_and_shifted() -> None:
    tokens = np.arange(1000, dtype=np.uint16)
    generator_a = torch.Generator().manual_seed(9)
    generator_b = torch.Generator().manual_seed(9)
    inputs_a, targets_a = get_batch(tokens, 5, 12, "cpu", generator_a)
    inputs_b, targets_b = get_batch(tokens, 5, 12, "cpu", generator_b)
    torch.testing.assert_close(inputs_a, inputs_b)
    torch.testing.assert_close(targets_a, targets_b)
    torch.testing.assert_close(inputs_a[:, 1:], targets_a[:, :-1])

