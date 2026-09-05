# Private Grading Guide

## Automated implementation — 65 points

- Linear and Embedding: 5
- RMSNorm: 6
- RoPE: 9
- SwiGLU and parameter matching: 6
- GQA and causal masking: 11
- Transformer block and LM: 8
- Cross-entropy and AdamW: 10
- Batch sampling, checkpointing, and generation: 10

Use private tests for unseen tensor shapes, dtype behavior, gradients, invalid
configuration, prefix invariance, exact optimizer equations, and checkpoint resume.
Do not grade wall-clock numerical equality.

## Report — 30 points

- Correct derivations and resource accounting: 8
- Controlled method and reproducibility: 6
- Accurate tables and plots: 6
- Component-by-component interpretation: 8
- Concise lineage with primary citations: 2

Award full credit for a well-supported null result. Deduct for changing multiple
configuration fields without disclosure, comparing unequal token budgets, plotting
loss against steps when batch sizes differ, or making large-model causal claims from
the small experiment.

## Engineering quality — 5 points

Award for readable shape documentation, small commits, a clean repository, and a
successful fresh `uv sync`. Data, checkpoints, or large run outputs in Git lose this
section.
