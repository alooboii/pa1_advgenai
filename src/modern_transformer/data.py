from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from tokenizers import Tokenizer


TINYSTORIES_DATASET = "roneneldan/TinyStories"
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINYSTORIES_FILES = {
    "train": "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    "validation": "data/validation-00000-of-00001-869c898b519ad725.parquet",
}


def load_token_array(path: str | Path) -> np.memmap:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"token file not found: {path}; run prepare_data.py first")
    return np.memmap(path, mode="r", dtype=np.uint16)


def get_batch(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    # BEGIN SOLUTION
    if len(tokens) <= context_length:
        raise ValueError("token array must be longer than context_length")
    starts = torch.randint(
        0,
        len(tokens) - context_length,
        (batch_size,),
        generator=generator,
    )
    rows = np.stack([np.asarray(tokens[i : i + context_length + 1], dtype=np.int64) for i in starts.tolist()])
    batch = torch.from_numpy(rows).to(device=device, dtype=torch.long)
    return batch[:, :-1], batch[:, 1:]
    # END SOLUTION


def _documents_from_fixture(path: Path) -> Iterator[str]:
    text = path.read_text(encoding="utf-8")
    documents = [item.strip() for item in text.split("<|endoftext|>") if item.strip()]
    while True:
        yield from documents


def _documents_from_huggingface(split: str) -> Iterator[str]:
    if split not in TINYSTORIES_FILES:
        raise ValueError(f"unknown split: {split}")
    relative_path = TINYSTORIES_FILES[split]
    url = (
        f"https://huggingface.co/datasets/{TINYSTORIES_DATASET}/resolve/"
        f"{TINYSTORIES_REVISION}/{relative_path}"
    )
    with tempfile.TemporaryDirectory(prefix=f"tinystories-{split}-") as temporary_dir:
        local_path = Path(temporary_dir) / Path(relative_path).name
        request = urllib.request.Request(url, headers={"User-Agent": "modern-transformer-assignment/0.1"})
        print(f"Downloading pinned {split} shard to temporary storage...", flush=True)
        with urllib.request.urlopen(request) as response, local_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        parquet = pq.ParquetFile(local_path)
        for batch in parquet.iter_batches(batch_size=1024, columns=["text"]):
            for document in batch.column(0).to_pylist():
                yield document


def _write_tokens(
    documents: Iterator[str],
    tokenizer: Tokenizer,
    output_path: Path,
    target_tokens: int,
    eot_id: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with output_path.open("wb") as stream:
            for document in documents:
                token_ids = tokenizer.encode(document).ids + [eot_id]
                remaining = target_tokens - written
                if remaining <= 0:
                    break
                array = np.asarray(token_ids[:remaining], dtype=np.uint16)
                stream.write(array.tobytes())
                written += len(array)
                if written >= target_tokens:
                    break
    finally:
        close = getattr(documents, "close", None)
        if close is not None:
            close()
    return written


def prepare_from_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    payload = yaml.safe_load(config_path.read_text())
    tokenizer_path = (config_path.parent / payload["tokenizer_path"]).resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eot_token = payload.get("eot_token", "<|endoftext|>")
    eot_id = tokenizer.token_to_id(eot_token)
    if eot_id is None:
        raise ValueError(f"tokenizer is missing {eot_token!r}")
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max + 1:
        raise ValueError("uint16 storage requires vocabulary <= 65,536")

    source = payload["source"]
    fixture_path = (config_path.parent / payload["fixture_path"]).resolve() if source == "fixture" else None
    train_documents = (
        _documents_from_fixture(fixture_path)
        if fixture_path is not None
        else _documents_from_huggingface("train")
    )
    validation_documents = (
        _documents_from_fixture(fixture_path)
        if fixture_path is not None
        else _documents_from_huggingface("validation")
    )
    train_path = Path(payload["train_path"])
    validation_path = Path(payload["validation_path"])
    train_count = _write_tokens(train_documents, tokenizer, train_path, int(payload["train_tokens"]), eot_id)
    validation_count = _write_tokens(
        validation_documents,
        tokenizer,
        validation_path,
        int(payload["validation_tokens"]),
        eot_id,
    )
    metadata = {
        "source": source,
        "dataset": TINYSTORIES_DATASET if source == "huggingface" else str(fixture_path),
        "revision": TINYSTORIES_REVISION if source == "huggingface" else None,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "vocab_size": tokenizer.get_vocab_size(),
        "eot_id": eot_id,
        "train_tokens": train_count,
        "validation_tokens": validation_count,
        "dtype": "uint16",
    }
    metadata_path = train_path.parent / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
