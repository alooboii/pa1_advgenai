from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def capture_rng_state(data_generator: torch.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if data_generator is not None:
        state["data_generator"] = data_generator.get_state()
    return state


def restore_rng_state(state: dict[str, Any], data_generator: torch.Generator | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # A checkpoint loaded with map_location="cuda" or "mps" moves every tensor,
    # including RNG byte tensors. Generator state APIs require CPU byte tensors.
    torch.set_rng_state(state["torch"].detach().cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in state["cuda"]])
    if data_generator is not None and "data_generator" in state:
        data_generator.set_state(state["data_generator"].detach().cpu())


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    data_generator: torch.Generator | None = None,
    scaler: Any = None,
) -> None:
    # BEGIN SOLUTION
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "rng_state": capture_rng_state(data_generator),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    # END SOLUTION


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    data_generator: torch.Generator | None = None,
    scaler: Any = None,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    # BEGIN SOLUTION
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng_state"], data_generator)
    return payload
    # END SOLUTION
