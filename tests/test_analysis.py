from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_analysis_ignores_inherited_notebook_backend(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "example"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"run_name": "example", "validation_loss": 2.5}) + "\n"
    )
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"tokens": 128, "validation_loss": 2.5}) + "\n"
    )
    output = tmp_path / "analysis"
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "analyze_results.py"),
            "--runs",
            str(tmp_path / "runs"),
            "--output",
            str(output),
        ],
        check=True,
        cwd=tmp_path,
        env=environment,
    )

    assert (output / "summary.csv").is_file()
    assert (output / "validation_curves.png").is_file()
    assert (output / "screening_statistics.json").is_file()
