from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from modern_transformer.checkpoint import load_checkpoint
from modern_transformer.config import load_experiment_config
from modern_transformer.data import load_token_array
from modern_transformer.generation import generate
from modern_transformer.model import TransformerLM, kv_cache_bytes
from modern_transformer.training import evaluate_loss, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate loss, generation, and GQA resource accounting")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="assets/tokenizer.json")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    config = load_experiment_config(args.config, args.overrides)
    device = resolve_device(config.train.device)
    model = TransformerLM(config.model).to(device)
    payload = load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()
    validation = load_token_array(config.data.validation_path)
    evaluations: dict[str, object] = {}
    for context_length in sorted({config.train.sequence_length, config.model.max_seq_len}):
        batch_size = max(1, config.train.batch_size * config.train.sequence_length // context_length)
        loss = evaluate_loss(
            model,
            validation,
            batch_size=batch_size,
            context_length=context_length,
            batches=config.train.eval_batches,
            device=device,
            seed=config.train.seed + 30_000 + context_length,
            config=config,
        )
        evaluations[f"validation_loss_{context_length}"] = loss
        evaluations[f"perplexity_{context_length}"] = math.exp(loss)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt).ids
    if device.type in {"cpu", "cuda"}:
        sample_generator = torch.Generator(device=device.type).manual_seed(config.train.seed + 40_000)
    else:
        torch.manual_seed(config.train.seed + 40_000)
        sample_generator = None
    generated = generate(
        model,
        torch.tensor([prompt_ids], dtype=torch.long, device=device),
        max_new_tokens=args.max_new_tokens,
        temperature=0.8,
        top_k=50,
        eos_token_id=tokenizer.token_to_id("<|endoftext|>"),
        generator=sample_generator,
    )
    evaluations["prompt"] = args.prompt
    evaluations["generation"] = tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
    evaluations["checkpoint_step"] = payload["step"]
    evaluations["parameter_count"] = model.parameter_count()
    evaluations["kv_cache_bytes"] = {
        str(length): kv_cache_bytes(config.model, batch_size=1, sequence_length=length)
        for length in (128, 256, 1024)
    }
    output_path = Path(args.checkpoint).parent / "evaluation.json"
    output_path.write_text(json.dumps(evaluations, indent=2) + "\n")
    print(json.dumps(evaluations, indent=2))


if __name__ == "__main__":
    main()
