# Modern Transformer From Scratch

This is the instructor repository for a one-week implementation and ablation
assignment. The source tree is runnable as a complete reference solution. A
release script converts marked solution regions into student TODOs and excludes
private material.

## Instructor quick start

```bash
uv sync
uv run pytest
uv run python prepare_data.py --config configs/data_debug.yaml
uv run python train.py --config configs/debug.yaml
uv run python evaluate.py --config configs/debug.yaml \
  --checkpoint runs/debug/checkpoint_last.pt
```

Build and audit the public release:

```bash
uv run python scripts/build_student_release.py
```

The full assignment is in `docs/ASSIGNMENT.md`. Do not publish this repository
directly: it contains instructor solutions and private tests.

