from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_instructor_feasibility.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "kaggle": {"accelerator": "gpu", "internet": True},
    }
    notebook.cells = [
        markdown(
            """
            # Modern Transformer — Instructor Kaggle Feasibility Runner

            This notebook runs the **private instructor calibration**, not the student assignment. It deliberately keeps model and training code in the repository so the experiment remains testable, reviewable, and resumable.

            It performs:

            1. GPU and environment validation
            2. private repository acquisition
            3. locked dependency installation and the full test suite
            4. pinned TinyStories token-array preparation
            5. a small CUDA smoke test
            6. learning-rate screening, three-seed architecture screens, and full ablations
            7. aggregation, feasibility gates, plots, and safe result packaging

            Before running, enable a **GPU accelerator** and **Internet** in Kaggle. For a private GitHub repository, create a Kaggle Secret named `GITHUB_TOKEN` with read access. Alternatively, attach the repository as a private Kaggle Dataset and select `SOURCE_MODE = "kaggle_dataset"` below.

            The default `SWEEP_PHASE = "all"` is the one-click path. For multiple Kaggle sessions, run `lr`, then `screen`, then `full`, saving the notebook output after each phase and attaching the prior output to the next session via `RESTORE_STATE_DIR`.
            """
        ),
        markdown(
            """
            ## 1. Parameters

            Edit only this cell for a normal run. The notebook targets the private `alooboii/pa1_advgenai` repository on `main`. For the final calibration, replace `main` with the exact commit SHA you want to freeze.
            """
        ),
        code(
            """
            from pathlib import Path

            # Source acquisition: "git" or "kaggle_dataset".
            SOURCE_MODE = "git"
            GITHUB_REPOSITORY = "alooboii/pa1_advgenai"
            GITHUB_REF = "main"
            KAGGLE_DATASET_SOURCE = "/kaggle/input/YOUR-PRIVATE-REPOSITORY-DATASET"

            # Optional path to a previous Kaggle output containing data/, runs/,
            # and artifacts/ from this project. Leave blank for a fresh session.
            RESTORE_STATE_DIR = ""

            PROJECT_DIR = Path("/kaggle/working/modern-transformer-assignment")
            SWEEP_PHASE = "all"  # one of: "lr", "screen", "full", "all"

            REQUIRE_CUDA = True
            RUN_TESTS = True
            RUN_DEBUG_SMOKE = True
            PREPARE_FULL_DATA = True

            assert SOURCE_MODE in {"git", "kaggle_dataset"}
            assert SWEEP_PHASE in {"lr", "screen", "full", "all"}
            print({
                "source_mode": SOURCE_MODE,
                "project_dir": str(PROJECT_DIR),
                "sweep_phase": SWEEP_PHASE,
                "restore_state": RESTORE_STATE_DIR or None,
            })
            """
        ),
        markdown(
            """
            ## 2. Runtime preflight

            The calibration target is a T4-class GPU. The notebook stops early if CUDA is unavailable, unless `REQUIRE_CUDA` is explicitly disabled for setup-only debugging.
            """
        ),
        code(
            """
            import os
            import platform
            import shutil
            import subprocess
            import sys
            import time

            print("Python:", sys.version.replace("\\n", " "))
            print("Platform:", platform.platform())
            print("Working directory:", Path.cwd())

            nvidia_smi = shutil.which("nvidia-smi")
            if nvidia_smi:
                subprocess.run(
                    [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                    check=True,
                )
            elif REQUIRE_CUDA:
                raise RuntimeError("No NVIDIA runtime detected. Enable a Kaggle GPU accelerator and restart the session.")
            """
        ),
        markdown(
            """
            ## 3. Acquire the instructor repository

            Git authentication is passed without printing the token. A Dataset source is copied from Kaggle's read-only input mount. On reruns, ignored experiment outputs in the working repository are preserved.
            """
        ),
        code(
            """
            import base64
            from collections import deque

            COMMAND_ENV = os.environ.copy()
            COMMAND_ENV["UV_CACHE_DIR"] = "/kaggle/working/.uv-cache"
            COMMAND_ENV["UV_LINK_MODE"] = "copy"
            COMMAND_ENV["PYTHONUNBUFFERED"] = "1"
            COMMAND_ENV["GIT_TERMINAL_PROMPT"] = "0"
            # Kaggle injects a sitecustomize module through PYTHONPATH. It imports
            # packages from Kaggle's system environment (including wrapt), which
            # should not leak into the lockfile-managed uv environment.
            COMMAND_ENV.pop("PYTHONPATH", None)
            COMMAND_ENV["PYTHONNOUSERSITE"] = "1"


            def run_command(arguments, *, cwd=None, env=None, label=None):
                'Run a bounded command and echo the non-sensitive invocation.'
                shown = label or " ".join(map(str, arguments))
                print(f"$ {shown}", flush=True)
                completed = subprocess.run(
                    list(map(str, arguments)),
                    cwd=cwd,
                    env=env or COMMAND_ENV,
                    text=True,
                    check=False,
                )
                if completed.returncode:
                    raise RuntimeError(f"command failed with exit code {completed.returncode}: {shown}")
                return completed


            def run_logged(arguments, log_path, *, cwd=None):
                'Save complete output while printing compact progress to the notebook.'
                log_path = Path(log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                shown = " ".join(map(str, arguments))
                print(f"$ {shown}", flush=True)
                recent = deque(maxlen=30)
                started = time.monotonic()
                with log_path.open("a", encoding="utf-8") as log_stream:
                    process = subprocess.Popen(
                        list(map(str, arguments)),
                        cwd=cwd,
                        env=COMMAND_ENV,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        log_stream.write(line)
                        log_stream.flush()
                        stripped = line.rstrip()
                        recent.append(stripped)
                        # Training emits JSON every few steps. Show evaluation points
                        # and orchestration milestones without bloating notebook output.
                        show_line = (
                            stripped.startswith("RUN ")
                            or stripped.startswith("selected learning rate")
                            or stripped.startswith("wrote ")
                            or stripped.startswith("created ")
                            or '"failures"' in stripped
                            or " passed" in stripped
                            or " failed" in stripped
                            or ('"validation_loss":' in stripped and '"validation_loss": null' not in stripped)
                        )
                        if show_line:
                            print(stripped, flush=True)
                    return_code = process.wait()
                elapsed = time.monotonic() - started
                print(f"Finished in {elapsed / 60:.1f} minutes; full log: {log_path}", flush=True)
                if return_code:
                    print("\\nLast log lines:")
                    print("\\n".join(recent))
                    raise RuntimeError(f"command failed with exit code {return_code}: {shown}")


            def git_arguments(token):
                if not token:
                    return ["git"]
                credential = base64.b64encode(f"x-access-token:{token}".encode()).decode()
                return ["git", "-c", f"http.extraHeader=AUTHORIZATION: basic {credential}"]


            if SOURCE_MODE == "git":
                if "YOUR_" in GITHUB_REPOSITORY:
                    raise ValueError("Set GITHUB_REPOSITORY in the parameters cell before continuing.")
                github_token = None
                try:
                    from kaggle_secrets import UserSecretsClient
                    github_token = UserSecretsClient().get_secret("GITHUB_TOKEN")
                except Exception:
                    # This is expected for public repositories or local notebook validation.
                    pass
                repository_url = f"https://github.com/{GITHUB_REPOSITORY}.git"
                git = git_arguments(github_token)
                if not (PROJECT_DIR / ".git").exists():
                    PROJECT_DIR.parent.mkdir(parents=True, exist_ok=True)
                    run_command(
                        [*git, "clone", repository_url, str(PROJECT_DIR)],
                        label=f"git clone https://github.com/{GITHUB_REPOSITORY}.git {PROJECT_DIR}",
                    )
                is_commit = len(GITHUB_REF) >= 7 and all(
                    character in "0123456789abcdefABCDEF" for character in GITHUB_REF
                )
                if is_commit:
                    run_command([*git, "fetch", "origin"], cwd=PROJECT_DIR, label="git fetch origin")
                    run_command(["git", "checkout", "--detach", GITHUB_REF], cwd=PROJECT_DIR)
                else:
                    run_command(
                        [*git, "fetch", "origin", GITHUB_REF],
                        cwd=PROJECT_DIR,
                        label=f"git fetch origin {GITHUB_REF}",
                    )
                    run_command(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=PROJECT_DIR)
            else:
                dataset_source = Path(KAGGLE_DATASET_SOURCE)
                if "YOUR-" in KAGGLE_DATASET_SOURCE or not dataset_source.exists():
                    raise FileNotFoundError(
                        "Set KAGGLE_DATASET_SOURCE to the attached private repository directory."
                    )
                shutil.copytree(dataset_source, PROJECT_DIR, dirs_exist_ok=True)

            required_paths = ["pyproject.toml", "uv.lock", "train.py", "scripts/run_instructor_sweep.py"]
            missing = [name for name in required_paths if not (PROJECT_DIR / name).exists()]
            if missing:
                raise FileNotFoundError(f"repository is missing required paths: {missing}")

            if RESTORE_STATE_DIR:
                state_root = Path(RESTORE_STATE_DIR)
                if not state_root.exists():
                    raise FileNotFoundError(f"RESTORE_STATE_DIR does not exist: {state_root}")
                for directory_name in ("data", "runs", "artifacts"):
                    source = state_root / directory_name
                    if source.exists():
                        shutil.copytree(source, PROJECT_DIR / directory_name, dirs_exist_ok=True)
                        print(f"Restored {directory_name}/ from {source}")

            os.chdir(PROJECT_DIR)
            if (PROJECT_DIR / ".git").exists():
                commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True
                ).strip()
            else:
                commit = "dataset-source (no Git metadata)"
            print("Project:", PROJECT_DIR)
            print("Source revision:", commit)
            """
        ),
        markdown(
            """
            ## 4. Install the locked environment

            `uv sync --frozen` makes the cloud environment use the repository lockfile. The second check confirms that CUDA is visible from the project environment—not merely from Kaggle's system Python.
            """
        ),
        code(
            """
            if shutil.which("uv") is None:
                run_command([sys.executable, "-m", "pip", "install", "--quiet", "uv"])

            run_logged(
                ["uv", "sync", "--frozen"],
                PROJECT_DIR / "artifacts" / "kaggle_uv_sync.log",
                cwd=PROJECT_DIR,
            )
            run_command(
                [
                    "uv", "run", "python", "-c",
                    (
                        "import torch; "
                        "print('torch:', torch.__version__); "
                        "print('cuda_available:', torch.cuda.is_available()); "
                        "print('cuda_version:', torch.version.cuda); "
                        "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); "
                        "assert torch.cuda.is_available()"
                        if REQUIRE_CUDA
                        else
                        "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
                    ),
                ],
                cwd=PROJECT_DIR,
            )
            """
        ),
        markdown(
            """
            ## 5. Run correctness tests

            These include component formulas, architectural invariants, gradients, deterministic sampling, checkpoint replay, generation, and CPU integration checks. Do not start an expensive sweep if this gate fails.
            """
        ),
        code(
            """
            if RUN_TESTS:
                run_logged(
                    ["uv", "run", "pytest", "-q"],
                    PROJECT_DIR / "artifacts" / "kaggle_pytest.log",
                    cwd=PROJECT_DIR,
                )
            else:
                print("Skipped by RUN_TESTS=False")
            """
        ),
        markdown(
            """
            ## 6. Prepare pinned TinyStories arrays

            The repository downloads only the pinned Parquet shards into temporary storage, tokenizes approximately 25M training and 1M validation tokens, and retains only `uint16` arrays plus provenance metadata. Existing arrays with the expected byte sizes are reused.
            """
        ),
        code(
            """
            import json

            train_array = PROJECT_DIR / "data" / "tinystories" / "train.bin"
            validation_array = PROJECT_DIR / "data" / "tinystories" / "validation.bin"
            metadata_path = PROJECT_DIR / "data" / "tinystories" / "metadata.json"


            def full_data_ready():
                expected_bytes = {
                    train_array: 25_000_000 * 2,
                    validation_array: 1_000_000 * 2,
                }
                if any(not path.exists() or path.stat().st_size != size for path, size in expected_bytes.items()):
                    return False
                if not metadata_path.exists():
                    return False
                metadata = json.loads(metadata_path.read_text())
                return (
                    metadata.get("revision") == "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
                    and metadata.get("train_tokens") == 25_000_000
                    and metadata.get("validation_tokens") == 1_000_000
                    and metadata.get("dtype") == "uint16"
                )


            if PREPARE_FULL_DATA and not full_data_ready():
                run_logged(
                    ["uv", "run", "python", "prepare_data.py", "--config", "configs/data.yaml"],
                    PROJECT_DIR / "artifacts" / "kaggle_prepare_data.log",
                    cwd=PROJECT_DIR,
                )
            elif PREPARE_FULL_DATA:
                print("Pinned token arrays already exist with the expected provenance and sizes.")
            else:
                print("Skipped by PREPARE_FULL_DATA=False")

            if PREPARE_FULL_DATA and not full_data_ready():
                raise RuntimeError("full data preparation did not produce the expected pinned arrays")
            if metadata_path.exists():
                print(json.dumps(json.loads(metadata_path.read_text()), indent=2))
            """
        ),
        markdown(
            """
            ## 7. CUDA smoke test

            This exercises preparation, training, checkpointing, evaluation, and generation with a tiny configuration before the calibration budget is committed. It is stable across reruns and resumes from its completed checkpoint.
            """
        ),
        code(
            """
            if RUN_DEBUG_SMOKE:
                run_command(
                    ["uv", "run", "python", "prepare_data.py", "--config", "configs/data_debug.yaml"],
                    cwd=PROJECT_DIR,
                )
                run_logged(
                    [
                        "uv", "run", "python", "train.py", "--config", "configs/debug.yaml",
                        "--set", "train.device=cuda",
                        "--set", "train.run_name=kaggle_debug",
                    ],
                    PROJECT_DIR / "artifacts" / "kaggle_debug.log",
                    cwd=PROJECT_DIR,
                )
                run_command(
                    [
                        "uv", "run", "python", "evaluate.py",
                        "--config", "configs/debug.yaml",
                        "--checkpoint", "runs/kaggle_debug/checkpoint_last.pt",
                        "--set", "train.device=cuda",
                        "--set", "train.run_name=kaggle_debug",
                        "--max-new-tokens", "32",
                    ],
                    cwd=PROJECT_DIR,
                )
                debug_summary = json.loads((PROJECT_DIR / "runs" / "kaggle_debug" / "summary.json").read_text())
                print(json.dumps(debug_summary, indent=2))
            else:
                print("Skipped by RUN_DEBUG_SMOKE=False")
            """
        ),
        markdown(
            """
            ## 8. Inspect the calibration plan

            The manifest fixes the four learning rates, screening seeds, token budgets, and six instructor architectures. Student-facing feasibility is based on the modern baseline plus four single-component reversions; the composite original model is contextual instructor evidence.
            """
        ),
        code(
            """
            import yaml

            manifest_path = PROJECT_DIR / "configs" / "sweeps" / "instructor.yaml"
            manifest = yaml.safe_load(manifest_path.read_text())
            architecture_count = len(manifest["architectures"])
            planned = {
                "learning_rate_runs": len(manifest["learning_rates"]),
                "screen_runs": architecture_count * len(manifest["screen_seeds"]),
                "full_runs": architecture_count,
                "screen_steps_per_run": manifest["screen_steps"],
                "full_steps_per_run": manifest["full_steps"],
                "architectures": list(manifest["architectures"]),
            }
            print(yaml.safe_dump(planned, sort_keys=False))

            if PREPARE_FULL_DATA and not full_data_ready():
                raise RuntimeError("full data arrays are not ready")
            if REQUIRE_CUDA and nvidia_smi is None:
                raise RuntimeError("CUDA preflight failed")
            """
        ),
        markdown(
            """
            ## 9. Run or resume the selected sweep phase

            This is the long-running cell. Complete logs are written to `artifacts/kaggle_sweep_<phase>.log`; notebook output only shows orchestration milestones and validation points. Run names and checkpoints are stable, so rerunning the cell resumes interrupted training when the requested configuration is unchanged.

            Phase dependencies:

            - `lr`: no prior phase required
            - `screen`: requires restored or current `lr_*` summaries
            - `full`: requires restored or current `lr_*` summaries
            - `all`: performs everything in order
            """
        ),
        code(
            """
            sweep_log = PROJECT_DIR / "artifacts" / f"kaggle_sweep_{SWEEP_PHASE}.log"
            run_logged(
                [
                    "uv", "run", "python", "scripts/run_instructor_sweep.py",
                    "--phase", SWEEP_PHASE,
                ],
                sweep_log,
                cwd=PROJECT_DIR,
            )
            """
        ),
        markdown(
            """
            ## 10. Aggregate and inspect results

            Full sweeps automatically run these commands. They are repeated safely here so partial phases also produce a compact table and downloadable metrics bundle.
            """
        ),
        code(
            """
            run_command(["uv", "run", "python", "analyze_results.py"], cwd=PROJECT_DIR)

            full_summaries = list((PROJECT_DIR / "runs").glob("full_*/summary.json"))
            if full_summaries:
                run_command(
                    ["uv", "run", "python", "scripts/analyze_feasibility.py"],
                    cwd=PROJECT_DIR,
                )
            run_command(
                ["uv", "run", "python", "scripts/package_results.py"],
                cwd=PROJECT_DIR,
            )

            import csv
            from IPython.display import Image, Markdown, display

            summary_path = PROJECT_DIR / "artifacts" / "analysis" / "summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as stream:
                summary_rows = list(csv.DictReader(stream))

            preferred_columns = [
                "run_name", "parameter_count", "tokens", "validation_loss",
                "validation_loss_256", "mean_tokens_per_second",
                "peak_memory_bytes", "elapsed_seconds", "device",
            ]
            available_columns = [
                column for column in preferred_columns
                if any(row.get(column) not in (None, "") for row in summary_rows)
            ]
            table_lines = [
                "| " + " | ".join(available_columns) + " |",
                "| " + " | ".join(["---"] * len(available_columns)) + " |",
            ]
            for row in summary_rows:
                table_lines.append("| " + " | ".join(row.get(column, "") for column in available_columns) + " |")
            display(Markdown("\\n".join(table_lines)))

            plot_path = PROJECT_DIR / "artifacts" / "analysis" / "validation_curves.png"
            if plot_path.exists():
                display(Image(filename=str(plot_path)))

            decision_path = PROJECT_DIR / "artifacts" / "analysis" / "feasibility_decision.json"
            if decision_path.exists():
                decision = json.loads(decision_path.read_text())
                print("Feasibility decision:")
                print(json.dumps(decision, indent=2))
            else:
                print("No feasibility decision yet; it requires full_* summaries.")
            """
        ),
        markdown(
            """
            ## 11. Export the safe evidence bundle

            The archive contains metrics, configuration fingerprints, environment records, evaluations, plots, and decisions. It intentionally excludes model checkpoints and token data. Download this file and return it for validation and final assignment calibration.

            For cross-session resume, use **Save Version** with outputs. In the next session, attach that version's output and set `RESTORE_STATE_DIR` to the directory containing this project's `data/`, `runs/`, and `artifacts/` directories.
            """
        ),
        code(
            """
            import hashlib
            from IPython.display import FileLink, display

            source_bundle = PROJECT_DIR / "artifacts" / "feasibility-results.tar.gz"
            export_bundle = Path("/kaggle/working/feasibility-results.tar.gz")
            if not source_bundle.exists():
                raise FileNotFoundError("result bundle was not created")
            shutil.copy2(source_bundle, export_bundle)
            digest = hashlib.sha256(export_bundle.read_bytes()).hexdigest()
            print(f"Bundle: {export_bundle}")
            print(f"Size: {export_bundle.stat().st_size:,} bytes")
            print(f"SHA-256: {digest}")
            display(FileLink(str(export_bundle)))
            """
        ),
        markdown(
            """
            ## Completion checklist

            A complete instructor run should leave:

            - `artifacts/sweep_status.json` with no unexpected failures
            - `artifacts/analysis/summary.csv`
            - `artifacts/analysis/validation_curves.png`
            - `artifacts/analysis/screening_statistics.json`
            - `artifacts/analysis/feasibility_decision.json`
            - `/kaggle/working/feasibility-results.tar.gz`

            Return the final archive. Do **not** publish the working directory because it contains the instructor solution, private tests, data arrays, and checkpoints.
            """
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build_notebook()
