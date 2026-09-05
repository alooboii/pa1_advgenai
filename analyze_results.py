from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

# Analysis is a batch command, including when it is launched from Jupyter.
# Kaggle exports its inline backend through MPLBACKEND, but that backend lives in
# the notebook environment and is not necessarily installed in the uv project.
# Force Matplotlib's non-interactive backend before importing pyplot so plotting
# does not depend on the parent notebook's environment.
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment logs into tables and learning curves")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="artifacts/analysis")
    args = parser.parse_args()
    run_root = Path(args.runs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    summaries = []
    curves: dict[str, list[dict]] = {}
    for summary_path in sorted(run_root.glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text())
        evaluation_path = run_dir / "evaluation.json"
        if evaluation_path.exists():
            summary.update(json.loads(evaluation_path.read_text()))
        summaries.append(summary)
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            curves[summary["run_name"]] = _load_jsonl(metrics_path)
    if not summaries:
        raise SystemExit(f"no summaries found under {run_root}")

    fieldnames = sorted({key for row in summaries for key in row if not isinstance(row[key], (dict, list))})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in summaries)

    fig, axis = plt.subplots(figsize=(8, 5))
    for name, records in curves.items():
        evaluated = [row for row in records if row.get("validation_loss") is not None]
        if evaluated:
            axis.plot(
                [row["tokens"] for row in evaluated],
                [row["validation_loss"] for row in evaluated],
                marker="o",
                markersize=2,
                label=name,
            )
    axis.set_xlabel("Training tokens")
    axis.set_ylabel("Validation loss (nats/token)")
    axis.set_title("Architecture ablations at a fixed token budget")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "validation_curves.png", dpi=180)
    plt.close(fig)

    grouped: dict[str, list[float]] = defaultdict(list)
    for summary in summaries:
        value = summary.get("validation_loss")
        if isinstance(value, (int, float)) and np.isfinite(value):
            base_name = summary["run_name"].rsplit("_seed", 1)[0]
            grouped[base_name].append(float(value))
    screening = {
        name: {"n": len(values), "mean": float(np.mean(values)), "std": float(np.std(values))}
        for name, values in grouped.items()
    }
    (output / "screening_statistics.json").write_text(json.dumps(screening, indent=2) + "\n")

    print(f"wrote analysis to {output}")


if __name__ == "__main__":
    main()
