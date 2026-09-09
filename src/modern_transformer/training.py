from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .data import EXPECTED_VOCAB_SIZE, get_batch, load_metadata, load_token_array
from .model import TransformerLM
from .optim import AdamW, clip_gradients, cosine_learning_rate, cross_entropy


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch environment")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_at_step(
    config: ExperimentConfig,
    step: int,
    *,
    elapsed_seconds: float | None = None,
) -> float:
    train = config.train
    if train.max_duration_seconds is not None and elapsed_seconds is not None:
        progress = min(max(elapsed_seconds / train.max_duration_seconds, 0.0), 1.0)
        warmup = train.duration_warmup_fraction
        if warmup > 0 and progress < warmup:
            return train.learning_rate * progress / warmup
        cosine_progress = (progress - warmup) / (1.0 - warmup)
        cosine_progress = min(max(cosine_progress, 0.0), 1.0)
        minimum = train.learning_rate * train.min_lr_ratio
        return minimum + 0.5 * (1.0 + math.cos(math.pi * cosine_progress)) * (
            train.learning_rate - minimum
        )
    return cosine_learning_rate(
        step,
        max_learning_rate=train.learning_rate,
        min_learning_rate=train.learning_rate * train.min_lr_ratio,
        warmup_steps=train.warmup_steps,
        cosine_cycle_steps=train.max_steps,
    )


def _autocast_context(config: ExperimentConfig, device: torch.device):
    enabled = config.train.amp and device.type == "cuda"
    if not enabled:
        return nullcontext()
    dtype = torch.float16 if config.train.amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def evaluate_loss(
    model: TransformerLM,
    tokens: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    batches: int,
    device: torch.device,
    seed: int,
    config: ExperimentConfig,
) -> float:
    """Return deterministic mean validation loss and restore model mode."""
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets = get_batch(tokens, batch_size, context_length, device, generator)
        with _autocast_context(config, device):
            losses.append(cross_entropy(model(inputs), targets).float().item())
    model.train(was_training)
    return float(np.mean(losses))


def train_step(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    tokens: np.ndarray,
    *,
    config: ExperimentConfig,
    device: torch.device,
    generator: torch.Generator,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    """Perform one optimizer step, including configured gradient accumulation."""
    # BEGIN SOLUTION
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    accumulation_steps = config.train.gradient_accumulation_steps
    for _ in range(accumulation_steps):
        inputs, targets = get_batch(
            tokens,
            config.train.batch_size,
            config.train.sequence_length,
            device,
            generator,
        )
        with _autocast_context(config, device):
            microbatch_loss = cross_entropy(model(inputs), targets)
            scaled_loss = microbatch_loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        accumulated_loss += scaled_loss.detach().float().item()

    scaler.unscale_(optimizer)
    gradient_norm = clip_gradients(model.parameters(), config.train.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    tokens_processed = (
        config.train.batch_size
        * config.train.sequence_length
        * config.train.gradient_accumulation_steps
    )
    return {
        "loss": accumulated_loss,
        "gradient_norm": gradient_norm,
        "tokens": float(tokens_processed),
    }
    # END SOLUTION


def _environment() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "git_commit": commit,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def train(config: ExperimentConfig) -> dict[str, Any]:
    """Run or resume training and produce checkpoints, metrics, and summary."""
    # BEGIN SOLUTION
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)
    data_metadata = load_metadata(config.data.dataset_dir)
    data_vocab_size = data_metadata.get("tokenizer", {}).get("vocab_size")
    if data_vocab_size != EXPECTED_VOCAB_SIZE or config.model.vocab_size != data_vocab_size:
        raise ValueError(
            f"model vocabulary ({config.model.vocab_size}) does not match prepared data ({data_vocab_size})"
        )
    run_dir = Path(config.train.output_dir) / config.train.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_path = run_dir / "checkpoint_last.pt"
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (run_dir / "environment.json").write_text(json.dumps(_environment(), indent=2) + "\n")

    train_tokens = load_token_array(config.data.train_path)
    validation_tokens = load_token_array(config.data.validation_path)
    model = TransformerLM(config.model).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.beta1, config.train.beta2),
        eps=config.train.eps,
        weight_decay=config.train.weight_decay,
    )
    amp_enabled = config.train.amp and device.type == "cuda" and config.train.amp_dtype == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    data_generator = torch.Generator().manual_seed(config.train.seed + 10_000)
    start_step = 0
    if config.train.resume and checkpoint_path.exists():
        payload = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            data_generator=data_generator,
            scaler=scaler,
            map_location=device,
        )
        if payload["config"] != config.to_dict():
            raise ValueError("checkpoint configuration does not match requested run")
        start_step = int(payload["step"])
        if start_step >= config.train.max_steps:
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                return json.loads(summary_path.read_text())
            raise ValueError("completed checkpoint exists without a summary.json")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    tokens_processed = start_step * config.train.batch_size * config.train.sequence_length * config.train.gradient_accumulation_steps
    final_train_loss = float("nan")
    final_validation_loss = float("nan")

    completed_steps = start_step
    for step in range(start_step, config.train.max_steps):
        step_started = time.perf_counter()
        elapsed_before_step = time.perf_counter() - started
        lr = learning_rate_at_step(
            config,
            step,
            elapsed_seconds=elapsed_before_step,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        step_metrics = train_step(
            model,
            optimizer,
            train_tokens,
            config=config,
            device=device,
            generator=data_generator,
            scaler=scaler,
        )
        gradient_norm = step_metrics["gradient_norm"]
        step_tokens = int(step_metrics["tokens"])
        tokens_processed += step_tokens
        duration = time.perf_counter() - step_started
        final_train_loss = step_metrics["loss"]
        completed_steps = step + 1
        elapsed_after_step = time.perf_counter() - started
        duration_reached = (
            config.train.max_duration_seconds is not None
            and elapsed_after_step >= config.train.max_duration_seconds
        )

        should_evaluate = (
            completed_steps % config.train.eval_interval == 0
            or completed_steps == config.train.max_steps
            or duration_reached
        )
        if should_evaluate:
            final_validation_loss = evaluate_loss(
                model,
                validation_tokens,
                batch_size=config.train.batch_size,
                context_length=config.train.sequence_length,
                batches=config.train.eval_batches,
                device=device,
                seed=config.train.seed + 20_000,
                config=config,
            )
        if completed_steps % config.train.log_interval == 0 or should_evaluate:
            record = {
                "step": completed_steps,
                "tokens": tokens_processed,
                "train_loss": final_train_loss,
                "validation_loss": final_validation_loss if should_evaluate else None,
                "learning_rate": lr,
                "gradient_norm": gradient_norm,
                "tokens_per_second": step_tokens / max(duration, 1e-9),
                "elapsed_seconds": time.perf_counter() - started,
            }
            _append_jsonl(metrics_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if not math.isfinite(final_train_loss):
            raise FloatingPointError(f"training diverged at step {step + 1}")
        if (
            completed_steps % config.train.checkpoint_interval == 0
            or completed_steps == config.train.max_steps
            or duration_reached
        ):
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=completed_steps,
                config=config.to_dict(),
                data_generator=data_generator,
                scaler=scaler,
            )
        if duration_reached:
            break

    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    summary = {
        "run_name": config.train.run_name,
        "config_fingerprint": config.fingerprint,
        "parameter_count": model.parameter_count(),
        "steps": completed_steps,
        "tokens": tokens_processed,
        "final_train_loss": final_train_loss,
        "validation_context": config.train.sequence_length,
        "validation_loss": final_validation_loss,
        f"validation_loss_{config.train.sequence_length}": final_validation_loss,
        f"perplexity_{config.train.sequence_length}": math.exp(final_validation_loss)
        if math.isfinite(final_validation_loss)
        else None,
        "elapsed_seconds": elapsed,
        "mean_tokens_per_second": (tokens_processed - start_step * step_tokens) / max(elapsed, 1e-9),
        "peak_memory_bytes": peak_memory,
        "device": str(device),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
    # END SOLUTION
