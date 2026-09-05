from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the instructor feasibility gates")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--analysis", default="artifacts/analysis")
    args = parser.parse_args()
    runs = Path(args.runs)
    analysis = Path(args.analysis)
    summaries = []
    for path in runs.glob("full_*/summary.json"):
        row = json.loads(path.read_text())
        evaluation = path.parent / "evaluation.json"
        if evaluation.exists():
            row.update(json.loads(evaluation.read_text()))
        summaries.append(row)
    full_by_name = {row["run_name"].removeprefix("full_"): row for row in summaries}
    student_names = ["baseline", "post_layernorm", "sinusoidal", "relu", "mha"]
    available = [full_by_name[name] for name in student_names if name in full_by_name]
    screening_path = analysis / "screening_statistics.json"
    screening = json.loads(screening_path.read_text()) if screening_path.exists() else {}
    total_seconds = sum(float(row.get("elapsed_seconds") or 0.0) for row in available)
    peak_values = [int(row["peak_memory_bytes"]) for row in available if row.get("peak_memory_bytes") is not None]
    peak_memory = max(peak_values, default=None)
    baseline = full_by_name.get("baseline")
    retained: dict[str, dict[str, object]] = {}
    for name in student_names[1:]:
        variant = full_by_name.get(name)
        if baseline is None or variant is None:
            retained[name] = {"retain": name == "mha", "reason": "missing full result"}
            continue
        loss_delta = abs(float(variant["validation_loss"]) - float(baseline["validation_loss"]))
        throughput_delta = abs(float(variant["mean_tokens_per_second"]) / float(baseline["mean_tokens_per_second"]) - 1.0)
        baseline_long = baseline.get("validation_loss_256")
        variant_long = variant.get("validation_loss_256")
        length_effect = (
            abs((float(variant_long) - float(variant["validation_loss"])) - (float(baseline_long) - float(baseline["validation_loss"])))
            if baseline_long is not None and variant_long is not None
            else 0.0
        )
        screen_base = screening.get("screen_baseline", {})
        screen_variant = screening.get(f"screen_{name}", {})
        noise = 2.0 * (float(screen_base.get("std", 0.0)) ** 2 + float(screen_variant.get("std", 0.0)) ** 2) ** 0.5
        quality_signal = loss_delta >= max(0.03, noise)
        efficiency_signal = throughput_delta >= 0.10
        length_signal = length_effect >= 0.03
        retained[name] = {
            "retain": name == "mha" or quality_signal or efficiency_signal or length_signal,
            "loss_delta": loss_delta,
            "screening_noise_threshold": noise,
            "throughput_fraction_delta": throughput_delta,
            "length_generalization_delta": length_effect,
            "reason": "GQA resource comparison is mandatory" if name == "mha" else None,
        }
    within_runtime = len(available) == 5 and total_seconds <= 90 * 60
    within_memory = peak_memory is None or peak_memory < 12 * 1024**3
    all_retained = all(item["retain"] for item in retained.values())
    decision = {
        "student_matrix_runtime_seconds": total_seconds,
        "peak_memory_bytes": peak_memory,
        "within_90_minute_runtime": within_runtime,
        "within_12_gib_memory": within_memory,
        "comparisons": retained,
        "recommended_requirement": "all_four"
        if within_runtime and within_memory and all_retained
        else "baseline_plus_two_assigned",
    }
    analysis.mkdir(parents=True, exist_ok=True)
    output = analysis / "feasibility_decision.json"
    output.write_text(json.dumps(decision, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
