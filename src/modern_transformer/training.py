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
from .data import get_batch, load_token_array
from .model import TransformerLM
from .optim import AdamW, cross_entropy


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


def learning_rate_at_step(config: ExperimentConfig, step: int) -> float:
    train = config.train
    if step < train.warmup_steps:
        return train.learning_rate * (step + 1) / train.warmup_steps
    progress = (step - train.warmup_steps) / max(1, train.max_steps - train.warmup_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    factor = train.min_lr_ratio + (1 - train.min_lr_ratio) * cosine
    return train.learning_rate * factor


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
    # BEGIN SOLUTION
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)
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

    for step in range(start_step, config.train.max_steps):
        step_started = time.perf_counter()
        lr = learning_rate_at_step(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(config.train.gradient_accumulation_steps):
            inputs, targets = get_batch(
                train_tokens,
                config.train.batch_size,
                config.train.sequence_length,
                device,
                data_generator,
            )
            with _autocast_context(config, device):
                loss = cross_entropy(model(inputs), targets) / config.train.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accumulated_loss += loss.detach().float().item()

        scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        step_tokens = config.train.batch_size * config.train.sequence_length * config.train.gradient_accumulation_steps
        tokens_processed += step_tokens
        duration = time.perf_counter() - step_started
        final_train_loss = accumulated_loss

        should_evaluate = (step + 1) % config.train.eval_interval == 0 or step + 1 == config.train.max_steps
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
        if (step + 1) % config.train.log_interval == 0 or should_evaluate:
            record = {
                "step": step + 1,
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
        if (step + 1) % config.train.checkpoint_interval == 0 or step + 1 == config.train.max_steps:
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step + 1,
                config=config.to_dict(),
                data_generator=data_generator,
                scaler=scaler,
            )

    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    summary = {
        "run_name": config.train.run_name,
        "config_fingerprint": config.fingerprint,
        "parameter_count": model.parameter_count(),
        "steps": config.train.max_steps,
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
