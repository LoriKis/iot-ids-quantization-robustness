"""Milestone 2A pilot-to-campaign estimate: scaling of measured pilot seconds, not a measured full run.

Reads the artifacts of a finished bounded pilot (artifacts/m2a-pilot) and the frozen split sizes, scales
per-row training time (each run's first step discarded), per-example attack time per (source, attack), PTQ and
efficiency time to the full campaign, and reports a point estimate with a range. The range covers the pilot-to-full
scaling error (+-50 %), the pgd50x5 budget scenario, and extra allowance for thermal throttling and RAM paging on
this laptop CPU. All numbers are planning figures with stated assumptions.
Design: docs/MILESTONE2_PROTOCOL.md section G (full-campaign estimate).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import psutil

from .common import ROOT, atomic_json

BANNER = ("Pilot results are non-scientific development checks (1 seed, 2 epochs, small subsets, validation data only) "
          "and carry no accuracy or robustness conclusion.")
SCALING_ERROR = 0.5       # +-50 % pilot-to-full scaling error (protocol G)
THROTTLE_PAGING_FACTOR = 1.25  # extra allowance on the high side only
FORWARD_UNIT, BACKWARD_UNIT = 1.0, 2.0  # nominal cost assumption of protocol D: forward 1, forward+backward 3
PER_EXAMPLE_CSV_BYTES = 60  # assumed bytes per example per unit in a future per-example CSV


# ------------------------------------------------------------------ training cost
def _rows_per_step(log: dict) -> list:
    rows = list(log["batch_sizes_epoch_1"]) * int(log["epochs"])
    if len(rows) != len(log["step_seconds"]):
        raise ValueError("train_log.json step_seconds do not match the epoch/batch structure")
    return rows


def per_row_step_seconds(log: dict) -> float:
    """Seconds per training row, discarding the first step (warm-up and allocation noise)."""
    rows, steps = _rows_per_step(log), log["step_seconds"]
    if len(steps) < 2:
        raise ValueError("need at least two steps to discard the first")
    return float(sum(steps[1:]) / sum(rows[1:]))


def nominal_step_cost(counters: dict) -> float:
    """Pass-count cost of one update in units of one forward: forward+backward = 3, input-only attack gradient = 3."""
    fb = FORWARD_UNIT + BACKWARD_UNIT
    return (counters["train_forward_backward_per_step"] * fb + counters["attack_grad_passes_per_step"] * fb
            + counters["attack_scoring_forwards_per_step"] * FORWARD_UNIT)


def training_cost_ratio(v1_log: dict, v3_log: dict) -> dict:
    """AT/ordinary cost: measured wall-clock per step (first step discarded) and the nominal pass-count ratio."""
    a, b = v1_log["step_seconds"], v3_log["step_seconds"]
    if len(a) != len(b):
        raise ValueError("V1 and V3 must have the same number of steps")
    measured = float(sum(b[1:]) / sum(a[1:]))
    nominal = nominal_step_cost(v3_log["pass_counters_per_step"]) / nominal_step_cost(v1_log["pass_counters_per_step"])
    return {"measured_wall_clock_ratio": measured, "nominal_pass_count_ratio": nominal,
            "steps_compared": len(a) - 1, "first_step_discarded": True,
            "v1_step_seconds_after_first": a[1:], "v3_step_seconds_after_first": b[1:],
            "v3_attack_seconds_after_first": v3_log["attack_seconds"][1:],
            "note": "a 2-epoch pilot has about 12 updates; the ratio is indicative only. The wall-clock ratio is the "
                    "headline number, the nominal ratio (forward = 1, forward+backward = 3) is a cross-check."}


# ------------------------------------------------------------------ attack cost
def attack_seconds_per_example(rows: list) -> dict:
    """{(source, attack): {'eps0': s/example, 'eps_positive': s/example}} from pilot attack rows."""
    buckets: dict = {}
    for r in rows:
        key = (r["attack_source"], r["attack"])
        bucket = buckets.setdefault(key, {"eps0": [], "eps_positive": []})
        bucket["eps0" if r["epsilon"] == 0 else "eps_positive"].append(r["attack_seconds"] / r["n"])
    return {k: {n: (float(np.mean(v)) if v else 0.0) for n, v in b.items()} for k, b in buckets.items()}


def budget_ratio(cells: list) -> dict:
    """Measured pgd50x5/pgd20 time ratio over the E4 plumbing cells."""
    pairs = [(c["seconds_pgd20"], c["seconds_pgd50x5"]) for c in cells if c.get("seconds_pgd20") and c.get("seconds_pgd50x5")]
    if not pairs:
        return {"ratio": None, "n_cells": 0}
    ratios = [b / a for a, b in pairs]
    return {"ratio": float(sum(b for _, b in pairs) / sum(a for a, _ in pairs)), "min": min(ratios), "max": max(ratios),
            "n_cells": len(pairs)}


# ------------------------------------------------------------------ estimate
def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def estimate(pilot: Path | str, split_path: Path | str | None = None, *, free_disk_bytes: int | None = None) -> dict:
    pilot = Path(pilot)
    split = _json(split_path or ROOT / "reports/milestone2a/split/split_m2.json")
    config = _json(ROOT / "configs/m2/training.json")
    campaign = _json(ROOT / "configs/m2/campaign.json")
    n_train, n_val, n_test = (split["partition_rows"][p] for p in ("train", "validation", "test"))
    seeds, epochs, batch = len(campaign["seeds"]), config["training"]["epochs"], config["training"]["batch_size"]
    updates = math.ceil(n_train / batch) * epochs

    v1, v3 = _json(pilot / "models/V1/train_log.json"), _json(pilot / "models/V3/train_log.json")
    ratio = training_cost_ratio(v1, v3)
    train_v1 = per_row_step_seconds(v1) * n_train * epochs
    train_v3 = per_row_step_seconds(v3) * n_train * epochs

    runtime = _json(pilot / "pilot_runtime.json")
    ptq_each = runtime["ptq_seconds"]
    ptq_total = seeds * (ptq_each["V2"] + ptq_each["V4"])

    rows = _json(pilot / "attack_results.json")["rows"]
    unit_cost = attack_seconds_per_example(rows)
    eps_positive = sum(e > 0 for e in campaign["epsilon_grid"])
    pair_sources = ["fp32_direct"] + [s for s in campaign["matrix"]["sources_by_target"]["int8"]]
    # one FP32 target and one INT8 target per pair, two pairs (V1->V2, V3->V4) per seed
    n_pairs = 2

    def eval_seconds(n_examples: int, pgd_factor: float) -> float:
        total = 0.0
        for source in pair_sources:
            for attack in ("fgsm", "pgd20"):
                c = unit_cost[(source, attack)]
                factor = pgd_factor if attack == "pgd20" else 1.0
                total += factor * (c["eps_positive"] * eps_positive + c["eps0"]) * n_examples
        return total * seeds * n_pairs

    cells = _json(pilot / "attack_budget_check.json")["cell_costs"]
    factor = budget_ratio(cells)
    pgd50_factor = factor["ratio"] or 4.0
    e4_subset = runtime["campaign_e4_subset"]
    e4_seconds = sum(c["seconds_pgd50x5"] / c["n"] + c["seconds_pgd20"] / c["n"] for c in cells) * e4_subset["n_subset"]
    ordinary_scenario = {"validation_attacks": eval_seconds(n_val, 1.0), "test_attacks": eval_seconds(n_test, 1.0)}
    heavy_scenario = {"validation_attacks": eval_seconds(n_val, pgd50_factor), "test_attacks": eval_seconds(n_test, pgd50_factor)}

    eff = _json(pilot / "efficiency/efficiency.json")
    processes = [e for e in eff["run_order"] if e["status"] == "ok"]
    wall = float(np.mean([e["finished_unix"] - e["started_unix"] for e in processes])) if processes else 0.0
    spec, eff_cfg = eff["warmup_and_timed"], _json(ROOT / "configs/m2/efficiency.json")
    extra = [((eff_cfg["batch1"]["timed"] + eff_cfg["batch1"]["warmup"] - spec["batch1"]["timed"] - spec["batch1"]["warmup"])
              * v["batch1_median_ms"]["median"]
              + (eff_cfg["batch256"]["timed"] + eff_cfg["batch256"]["warmup"] - spec["batch256"]["timed"] - spec["batch256"]["warmup"])
              * v["batch256_median_ms"]["median"]) / 1e3
             for v in eff["summary"].values() if v["batch1_median_ms"]]
    extra_per_process = float(np.mean(extra)) if extra else 0.0
    efficiency_seconds = len(eff_cfg["variants"]) * eff_cfg["repetitions"]["full"] * (wall + extra_per_process)

    fixed = (train_v1 + train_v3) * seeds + ptq_total + e4_seconds + efficiency_seconds
    central = fixed + ordinary_scenario["validation_attacks"] + ordinary_scenario["test_attacks"]
    heavy = fixed + heavy_scenario["validation_attacks"] + heavy_scenario["test_attacks"]
    low_s, high_s = central * (1 - SCALING_ERROR), heavy * (1 + SCALING_ERROR) * THROTTLE_PAGING_FACTOR

    unit_files = [p for p in (pilot / "attacks").rglob("attack_*.json")]
    unit_bytes = float(np.mean([p.stat().st_size for p in unit_files])) if unit_files else 0.0
    units_per_partition = seeds * n_pairs * 2 * len(campaign["epsilon_grid"]) * len(pair_sources)
    checkpoint_bytes = sum(p.stat().st_size for p in (pilot / "models").rglob("*.pt")) * seeds
    csv_bytes = (n_val + n_test) * units_per_partition / 2 * PER_EXAMPLE_CSV_BYTES
    storage = checkpoint_bytes + units_per_partition * 2 * unit_bytes + csv_bytes
    pilot_bytes = sum(p.stat().st_size for p in pilot.rglob("*") if p.is_file())
    free = free_disk_bytes if free_disk_bytes is not None else psutil.disk_usage(str(ROOT)).free

    def hours(seconds):
        return seconds / 3600

    return {
        "banner": BANNER, "basis": str(pilot), "kind": "planning estimate scaled from pilot timings; not a measured full run",
        "split": {"train_rows": n_train, "validation_rows": n_val, "test_rows": n_test},
        "campaign": {"seeds": seeds, "epochs": epochs, "batch_size": batch, "updates_per_run": updates,
                     "required_training_runs": 2 * seeds, "required_ptq_units": 2 * seeds,
                     "evaluation_partitions": ["validation", "test"],
                     "epsilon_values_positive": eps_positive, "sources_per_pair": pair_sources},
        "pilot_measurements": {
            "v1_seconds_per_row_after_first_step": per_row_step_seconds(v1),
            "v3_seconds_per_row_after_first_step": per_row_step_seconds(v3),
            "training_cost_ratio": {k: ratio[k] for k in ("measured_wall_clock_ratio", "nominal_pass_count_ratio", "steps_compared")},
            "ptq_seconds": ptq_each, "attack_seconds_per_example": {f"{s}|{a}": v for (s, a), v in sorted(unit_cost.items())},
            "pgd50x5_over_pgd20": factor, "efficiency_mean_process_seconds": wall},
        "components_seconds": {
            "training_v1_all_seeds": train_v1 * seeds, "training_v3_all_seeds": train_v3 * seeds, "ptq_all_units": ptq_total,
            "step0_e4_budget_check": e4_seconds, "efficiency_harness": efficiency_seconds,
            "attacks_pgd20_validation": ordinary_scenario["validation_attacks"], "attacks_pgd20_test": ordinary_scenario["test_attacks"],
            "attacks_pgd50x5_validation": heavy_scenario["validation_attacks"], "attacks_pgd50x5_test": heavy_scenario["test_attacks"]},
        "total_hours": {"point_estimate_pgd20": hours(central), "if_pgd50x5_everywhere": hours(heavy),
                        "planning_range": [hours(low_s), hours(high_s)],
                        "range_rule": f"low = point x (1 - {SCALING_ERROR}); high = pgd50x5 scenario x (1 + {SCALING_ERROR}) x {THROTTLE_PAGING_FACTOR} (thermal throttling and RAM paging allowance)"},
        "disk": {"estimated_campaign_bytes": int(storage), "budget_bytes": 2 * 1024 ** 3, "pilot_output_bytes": int(pilot_bytes),
                 "free_disk_bytes": int(free), "within_budget": storage < 2 * 1024 ** 3},
        "assumptions": [
            "single process, 2 training threads, 1 attack and efficiency thread (as in the pilot)",
            "training time scales linearly with rows after discarding each run's first step; the pilot has about 12 updates",
            "attack time per example is the pilot mean for each (source, attack), separately for epsilon 0 and epsilon > 0; the pilot's last batch is partial",
            "evaluation covers validation and test once each (test belongs to a later milestone) with the full-split row counts",
            "pgd50x5 scenario multiplies pgd20 time by the pilot E4 plumbing ratio (measured on a 54-row subset)",
            "optional V5 QAT arms and the per-position sensitivity check are excluded",
            "planning figures only: the pilot models are nearly untrained, so attack cost per example may differ for trained models"],
        "framing": "This is a new group-disjoint evaluation partition within a previously examined dataset, not a completely untouched external benchmark. The dataset was seen by Milestone 1; the earlier test partition was evaluated; identical feature vectors with conflicting labels remain in the data."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    record = estimate(args.pilot)
    atomic_json(args.out, record)
    print(json.dumps({"total_hours": record["total_hours"], "disk": record["disk"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
