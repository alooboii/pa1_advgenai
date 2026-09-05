# Instructor Guide

This file is private and is intentionally excluded from the student release.

## Repository lifecycle

1. Develop and validate locally with `uv run pytest`.
2. Build `dist/student_assignment` with the release builder.
3. Create the public Git repository from that generated directory, not from the
   instructor repository or its Git history.
4. Never publish `instructor_tests`, solution-marked source, feasibility logs, or
   checkpoints.

## Local gate

```bash
uv sync
uv run pytest
uv run python prepare_data.py --config configs/data_debug.yaml
uv run python train.py --config configs/debug.yaml
uv run python evaluate.py --config configs/debug.yaml \
  --checkpoint runs/debug/checkpoint_last.pt
uv run python scripts/build_student_release.py
```

The reference debug run should finish in under two minutes on a current laptop CPU.

## Cloud feasibility run

Clone the private instructor repository into Colab or Kaggle, select a GPU, and run:

```bash
uv sync
uv run pytest
uv run python prepare_data.py --config configs/data.yaml
uv run python scripts/run_instructor_sweep.py --phase all
```

The sweep is resumable because each run has a stable name and training checkpoints
automatically. It performs the learning-rate screen, three-seed architecture screen,
full runs, evaluation, aggregation, and safe result packaging.

Return `artifacts/feasibility-results.tar.gz`. It contains only JSON/JSONL metrics,
environment records, summaries, plots, and tables—never model state or data.

For Kaggle, upload `notebooks/kaggle_instructor_feasibility.ipynb` and edit its
single parameters cell. It supports a private GitHub repository through a Kaggle
Secret named `GITHUB_TOKEN`, or an attached private Kaggle Dataset containing the
repository. The default `all` phase performs the commands above. To split the sweep
across sessions, run `lr`, `screen`, and `full` in order and restore `data/`, `runs/`,
and `artifacts/` from the prior private notebook output.

## Release decision

Require all four reversions when the baseline plus four student runs total at most
90 minutes on the measured T4-class GPU and peak memory remains below 12 GB.
Otherwise assign two reversions per student and distribute instructor results for
the remaining comparisons.

Use the short-run three-seed spread when judging whether a final difference is
larger than run-to-run noise. Do not present an architecture as universally superior
if its only measured advantage is at one seed or one scale.

## Suggested schedule

- Day 1: setup, tensor warmup, Linear/Embedding/RMSNorm
- Day 2: RoPE and its tests
- Day 3: GQA
- Day 4: SwiGLU, block, and full LM
- Day 5: loss, AdamW, data, and checkpointing
- Weekend: cloud runs and report
