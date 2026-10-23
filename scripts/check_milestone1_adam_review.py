"""Collect and independently check compact evidence for corrected Adam."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1, 7, 13, 29]
CONDITIONS = ["Clean", "FGSM", "PGD"]
FILES = [
    "metrics.csv", "per_class.csv", "predictions.csv", "split_ids.json",
    "preprocessing.json", "preprocessing.npz", "confusion_matrices.npz",
    "verification.json", "training_complete.json", "runtime.json", "manifest.json",
]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, *, tol=1e-9):
    assert math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol), (
        actual, expected
    )


def collect(run_dir: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        source = run_dir / f"seed-{seed}"
        target = output / "evidence" / f"seed-{seed}"
        target.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            shutil.copy2(source / name, target / name)
    for name in ("configuration.json", "environment_actual.json", "fingerprint.json",
                 "data_range_report.json", "split_overlap_audit.json"):
        shutil.copy2(run_dir / name, output / name)
    (output / "evidence" / "files_sha256.json").write_text(
        json.dumps({
            str(path.relative_to(output / "evidence")): sha(path)
            for path in sorted((output / "evidence").rglob("*")) if path.is_file()
        }, indent=2) + "\n", encoding="utf-8"
    )


def check(package: Path) -> None:
    cfg = load(package / "configuration.json")
    assert cfg["seeds"] == SEEDS
    all_metrics = []
    for seed in SEEDS:
        base = package / "evidence" / f"seed-{seed}"
        manifest = load(base / "manifest.json")
        split = load(base / "split_ids.json")
        train, validation, test = (set(split[key]) for key in ("train", "validation", "test"))
        assert len(train) == 20997 and len(validation) == 4500 and len(test) == 4500
        assert not train & validation and not train & test and not validation & test
        assert split["dataset_sha256"] == cfg["dataset"]["sha256"]
        done = load(base / "training_complete.json")
        assert (done["epochs"], done["total_optimizer_steps"]) == (25, 1050)
        assert sum(done["batch_sizes_epoch_1"]) == 20997
        assert done["total_optimizer_steps"] * 512 - (512 - 5) * 25 == 524925
        verification = load(base / "verification.json")
        assert verification["test_rows_evaluated"] == 4500
        assert verification["evaluation_weights_unchanged"]
        assert verification["same_examples_all_conditions"]
        assert verification["checkpoint_reload"]["bitwise_equal"]
        predictions = rows(base / "predictions.csv")
        metrics = rows(base / "metrics.csv")
        assert len(predictions) == 13500 and len(metrics) == 3
        clean = [r for r in predictions if r["condition"] == "Clean"]
        assert [r["row_id"] for r in clean] == [str(x) for x in split["test"]]
        clean_by_id = {r["row_id"]: r for r in clean}
        for condition in CONDITIONS:
            pred = [r for r in predictions if r["condition"] == condition]
            assert [r["row_id"] for r in pred] == [r["row_id"] for r in clean]
            metric = next(r for r in metrics if r["condition"] == condition)
            assert metric["checkpoint_sha256"] == manifest["checkpoint.pt"]
            y = [int(r["label"]) for r in pred]
            p = [int(r["prediction"]) for r in pred]
            assert all(0 <= x < 27 for x in y + p)
            correct = sum(actual == predicted for actual, predicted in zip(y, p))
            close(metric["accuracy"], correct / 4500)
            clean_correct = [clean_by_id[r["row_id"]]["prediction"] == r["label"] for r in pred]
            denominator = sum(clean_correct)
            numerator = sum(ok and r["prediction"] != r["label"] for ok, r in zip(clean_correct, pred))
            assert int(metric["asr_numerator"]) == numerator
            assert int(metric["asr_denominator"]) == denominator
            close(metric["asr_percent"], 100 * numerator / denominator if denominator else 0)
            all_metrics.append(metric)
        for name in ("metrics.csv", "predictions.csv", "verification.json"):
            assert manifest[name] == sha(base / name), (seed, name)
    summary = rows(package / "summary.csv")
    assert len(summary) == 15
    for record in summary:
        values = [float(r[record["metric"]]) for r in all_metrics
                  if r["condition"] == record["condition"]]
        assert int(record["n"]) == 4
        close(record["mean"], statistics.mean(values))
        close(record["sample_std"], statistics.stdev(values))
    print("PASS: 4 seeds, 12 condition rows, 54,000 aligned predictions, split/step/hash checks, metric recomputation, sample SD")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/milestone1-paper-described-adam-rerun")
    parser.add_argument("--package", type=Path, default=ROOT / "reports/milestone1-paper-described-adam")
    args = parser.parse_args()
    if args.collect:
        collect(args.run_dir.resolve(), args.package.resolve())
    check(args.package.resolve())
