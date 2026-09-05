from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _run(config: str, overrides: list[str]) -> int:
    command = [sys.executable, "train.py", "--config", config]
    for item in overrides:
        command.extend(["--set", item])
    print("RUN", " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def _evaluate(config: str, run_name: str) -> int:
    checkpoint = Path("runs") / run_name / "checkpoint_last.pt"
    if not checkpoint.exists():
        return 1
    return subprocess.run(
        [
            sys.executable,
            "evaluate.py",
            "--config",
            config,
            "--checkpoint",
            str(checkpoint),
        ],
        check=False,
    ).returncode


def _summary(run_name: str) -> dict | None:
    path = Path("runs") / run_name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def _screen_overrides(name: str, steps: int, warmup: int, seed: int, lr: float) -> list[str]:
    return [
        f"train.run_name={name}",
        f"train.max_steps={steps}",
        f"train.warmup_steps={warmup}",
        f"train.seed={seed}",
        f"train.learning_rate={lr}",
        f"train.eval_interval={steps}",
        f"train.checkpoint_interval={steps}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable instructor calibration sweep")
    parser.add_argument("--manifest", default="configs/sweeps/instructor.yaml")
    parser.add_argument("--phase", choices=["lr", "screen", "full", "all"], default="all")
    args = parser.parse_args()
    manifest = yaml.safe_load(Path(args.manifest).read_text())
    status: dict[str, object] = {"failures": []}

    if args.phase in {"lr", "all"}:
        for learning_rate in manifest["learning_rates"]:
            name = f"lr_{learning_rate:g}".replace(".", "p")
            code = _run(
                manifest["architectures"]["baseline"],
                _screen_overrides(
                    name,
                    manifest["screen_steps"],
                    manifest["screen_warmup_steps"],
                    42,
                    learning_rate,
                ),
            )
            if code:
                status["failures"].append(name)

    candidates = []
    for learning_rate in manifest["learning_rates"]:
        name = f"lr_{learning_rate:g}".replace(".", "p")
        result = _summary(name)
        if result and result.get("validation_loss") is not None:
            candidates.append((result["validation_loss"], learning_rate))
    if not candidates:
        raise SystemExit("No successful LR screen summaries. Run --phase lr first.")
    best_lr = min(candidates)[1]
    status["selected_learning_rate"] = best_lr
    print(f"selected learning rate: {best_lr:g}")

    if args.phase in {"screen", "all"}:
        for architecture, config_path in manifest["architectures"].items():
            for seed in manifest["screen_seeds"]:
                name = f"screen_{architecture}_seed{seed}"
                code = _run(
                    config_path,
                    _screen_overrides(
                        name,
                        manifest["screen_steps"],
                        manifest["screen_warmup_steps"],
                        seed,
                        best_lr,
                    ),
                )
                if code:
                    status["failures"].append(name)

    if args.phase in {"full", "all"}:
        completed_full_runs: list[tuple[str, str]] = []
        for architecture, config_path in manifest["architectures"].items():
            name = f"full_{architecture}"
            code = _run(
                config_path,
                [
                    f"train.run_name={name}",
                    f"train.learning_rate={best_lr}",
                    f"train.max_steps={manifest['full_steps']}",
                ],
            )
            if code:
                status["failures"].append(name)
                if architecture == "post_layernorm":
                    rescue_name = "full_post_layernorm_rescue"
                    rescue_code = _run(
                        config_path,
                        [
                            f"train.run_name={rescue_name}",
                            f"train.learning_rate={best_lr / 2}",
                            f"train.max_steps={manifest['full_steps']}",
                        ],
                    )
                    if rescue_code:
                        status["failures"].append(rescue_name)
                    else:
                        completed_full_runs.append((config_path, rescue_name))
            else:
                completed_full_runs.append((config_path, name))
        for config_path, run_name in completed_full_runs:
            if _evaluate(config_path, run_name):
                status["failures"].append(f"evaluate:{run_name}")

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/sweep_status.json").write_text(json.dumps(status, indent=2) + "\n")
    if args.phase in {"full", "all"}:
        subprocess.run([sys.executable, "analyze_results.py"], check=False)
        subprocess.run([sys.executable, "scripts/analyze_feasibility.py"], check=False)
        subprocess.run([sys.executable, "scripts/package_results.py"], check=False)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
