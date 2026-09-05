"""Instructor utility for rebuilding the committed TinyStories tokenizer."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from modern_transformer.data import _documents_from_huggingface


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="assets/tokenizer.json")
    parser.add_argument("--documents", type=int, default=50_000)
    args = parser.parse_args()

    stream = _documents_from_huggingface("train")
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=4096,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    iterator = itertools.islice(stream, args.documents)
    tokenizer.train_from_iterator(iterator, trainer=trainer, length=args.documents)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output), pretty=True)
    print(f"saved {tokenizer.get_vocab_size()}-token tokenizer to {output}")


if __name__ == "__main__":
    main()
