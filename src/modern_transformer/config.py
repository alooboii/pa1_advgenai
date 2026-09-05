from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


NormStyle = Literal["pre_rms", "post_layer"]
PositionStyle = Literal["rope", "sinusoidal"]
FFNStyle = Literal["swiglu", "relu"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    max_seq_len: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_q_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 704
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    norm_style: NormStyle = "pre_rms"
    position_style: PositionStyle = "rope"
    ffn_style: FFNStyle = "swiglu"
    init_std: float = 0.02

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_q_heads": self.n_q_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_ff": self.d_ff,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.d_model % self.n_q_heads:
            raise ValueError("d_model must be divisible by n_q_heads")
        if self.n_q_heads % self.n_kv_heads:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.norm_style not in ("pre_rms", "post_layer"):
            raise ValueError(f"unknown norm_style: {self.norm_style}")
        if self.position_style not in ("rope", "sinusoidal"):
            raise ValueError(f"unknown position_style: {self.position_style}")
        if self.ffn_style not in ("swiglu", "relu"):
            raise ValueError(f"unknown ffn_style: {self.ffn_style}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_q_heads

    @property
    def queries_per_kv(self) -> int:
        return self.n_q_heads // self.n_kv_heads


@dataclass(frozen=True)
class DataConfig:
    train_path: str = "data/tinystories/train.bin"
    validation_path: str = "data/tinystories/validation.bin"


@dataclass(frozen=True)
class TrainConfig:
    run_name: str = "baseline"
    output_dir: str = "runs"
    device: str = "auto"
    seed: int = 42
    sequence_length: int = 128
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_steps: int = 1500
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 500
    amp: bool = True
    amp_dtype: Literal["float16", "bfloat16"] = "float16"
    resume: bool = True

    def __post_init__(self) -> None:
        if self.sequence_length <= 0 or self.batch_size <= 0 or self.max_steps <= 0:
            raise ValueError("sequence_length, batch_size, and max_steps must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if not (0 <= self.warmup_steps < self.max_steps):
            raise ValueError("warmup_steps must be in [0, max_steps)")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if self.train.sequence_length > self.model.max_seq_len:
            raise ValueError("training sequence_length exceeds model.max_seq_len")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"cyclic config inheritance involving {path}")
    seen.add(path)
    payload = yaml.safe_load(path.read_text()) or {}
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    parent_payload = _read_yaml((path.parent / parent).resolve(), seen)
    return _deep_merge(parent_payload, payload)


def _parse_override(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return value


def _apply_override(payload: dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ValueError(f"override must be key=value, got {item!r}")
    dotted_key, raw_value = item.split("=", 1)
    keys = dotted_key.split(".")
    cursor = payload
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            raise KeyError(f"unknown override path: {dotted_key}")
        cursor = cursor[key]
    if keys[-1] not in cursor:
        raise KeyError(f"unknown override key: {dotted_key}")
    cursor[keys[-1]] = _parse_override(raw_value)


def load_experiment_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    payload = _read_yaml(Path(path))
    for item in overrides or []:
        _apply_override(payload, item)
    return ExperimentConfig(
        model=ModelConfig(**payload.get("model", {})),
        data=DataConfig(**payload.get("data", {})),
        train=TrainConfig(**payload.get("train", {})),
    )
