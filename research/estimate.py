"""Measured pilot extrapolation, not a promise or scientific experiment result."""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import ROOT, atomic_json
from .data import build_data
from .runner import validate_config
from .validate import validate_run


def estimate(pilot, output):
    pilot, output = Path(pilot), Path(output)
    validate_run(pilot, "both")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    full = json.loads((ROOT / "configs/full.json").read_text())
    validate_config(full)
    data = build_data(full, output / "full_data_manifest")
    ntrain = len(data["y"]["train"])
    baseline = json.loads((pilot / "train/s1/fp32/metadata.json").read_text())
    at = json.loads((pilot / "train/s1/at_fp32/metadata.json").read_text())
    train_estimates = {}
    for name, record in (("fp32", baseline), ("at_fp32", at)):
        # Includes validation/model-selection overhead from each measured epoch.
        train_estimates[name] = record["training_seconds"] * ntrain / record["training_rows"] * full["training"]["epochs"] / record["epochs"] * len(full["seeds"])
    pilot_cfg = json.loads((pilot / "run.json").read_text())["config"]
    pilot_frozen = json.loads((pilot / "protocol_selection/frozen.json").read_text())["selected"]
    max_candidate = max(full["attack_candidates"], key=lambda v: v["steps"] * v["restarts"])
    pgd_factor = (1 + max_candidate["steps"] * max_candidate["restarts"]) / (1 + pilot_frozen["steps"] * pilot_frozen["restarts"])
    evaluation_seconds = 0.0
    for partition in ("validation", "test"):
        target_n = len(data["y"][partition])
        for file in (pilot / partition / "evaluation").glob("s1/*/metrics.json"):
            rows = json.loads(file.read_text())
            main = file.parent.name in ("fp32", "ptq_all_linear", "at_fp32", "at_ptq_all_linear")
            eps_count = sum(e > 0 for e in full["epsilons"]) if main else 1
            for kind in ("fgsm", "pgd"):
                timing_per_example = [r["attack_seconds"] / r["n"] for r in rows if r["attack"] == kind and r["epsilon"] > 0]
                evaluation_seconds += float(np.mean(timing_per_example)) * target_n * eps_count * len(full["seeds"]) * (pgd_factor if kind == "pgd" else 1)
            # Measured non-attack evaluation/benchmark/diagnostic overhead.
            elapsed = json.loads((file.parent / "timing.json").read_text())["evaluation_seconds"]
            other = max(0, elapsed - sum(r["attack_seconds"] for r in rows))
            evaluation_seconds += other * target_n / rows[0]["n"] * len(full["seeds"])
        for file in (pilot / partition / "transfer").glob("s1/*/metrics.json"):
            row = json.loads(file.read_text())[0]
            evaluation_seconds += row["attack_seconds"] / row["n"] * target_n * pgd_factor * len(full["seeds"])
    central = sum(train_estimates.values()) + evaluation_seconds
    measured_bytes = sum(p.stat().st_size for p in pilot.rglob("*") if p.is_file())
    scaled_predictions = measured_bytes * len(full["seeds"]) * len(data["y"]["test"]) / len(json.loads((pilot / "data/manifest.json").read_text())["selected_row_ids"]["test"]) * len(full["epsilons"]) / len(pilot_cfg["epsilons"])
    rss = max(json.loads(p.read_text())["process_rss_bytes"] for p in pilot.glob("*/evaluation/s1/*/benchmark.json"))
    record = {"basis": str(pilot.resolve()), "full_config": "configs/full.json",
              "full_split_counts": {k: len(v) for k, v in data["y"].items()},
              "measured_baseline_training_seconds": baseline["training_seconds"],
              "measured_at_training_seconds": at["training_seconds"],
              "measured_training_rows": baseline["training_rows"],
              "full_training_seconds_estimate": train_estimates,
              "full_evaluation_seconds_estimate": evaluation_seconds,
              "full_total_hours_central": central / 3600,
              "planning_hours_range": [central / 3600, central / 3600 * 4],
              "pgd_scaling_factor": pgd_factor,
              "full_pgd_budget_assumption": max_candidate,
              "measured_pilot_output_bytes": measured_bytes,
              "conservative_full_output_storage_bytes": int(scaled_predictions),
              "measured_peak_observed_process_rss_bytes": rss,
              "planning_process_ram_bytes": int(max(rss * 2, rss + 4 * 50 * 39 * max(len(data['y']['validation']), len(data['y']['test'])) * 4)),
              "limitations": ["linear per-example/epoch extrapolation; small pilot is not a full-epoch measurement",
                              "one-thread laptop thermal throttling and background processes can increase runtime",
                              "planning range is heuristic, not a confidence interval",
                              "excludes optional QAT/reproduction and installation; includes both validation/test phases",
                              "storage extrapolation deliberately scales all pilot files and overestimates fixed checkpoints",
                              "RAM is sampled process RSS plus conservative allowance, not measured full-study peak"]}
    atomic_json(output / "estimate.json", record)
    print(json.dumps(record, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pilot")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    estimate(a.pilot, a.output)


if __name__ == "__main__":
    main()
