from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


SAFE_NAMES = {"config.json", "environment.json", "metrics.jsonl", "summary.json", "evaluation.json"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Package small feasibility artifacts without checkpoints or data")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="artifacts/feasibility-results.tar.gz")
    args = parser.parse_args()
    run_root = Path(args.runs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(run_root.glob("*/*")):
            if path.name in SAFE_NAMES and path.is_file():
                archive.add(path, arcname=path.relative_to(run_root.parent))
        for path in sorted(Path("artifacts/analysis").glob("*")):
            if path.is_file():
                archive.add(path, arcname=path)
        status = Path("artifacts/sweep_status.json")
        if status.exists():
            archive.add(status, arcname=status)
    print(f"created {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
