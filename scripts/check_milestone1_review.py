"""Collect compact evidence, or check the review repository using only stdlib.

Never loads models/data tensors, trains, attacks, or changes original run records.
Default mode is read-only. --collect copies measured evidence from a local run.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/milestone1-artifact"
SEEDS = [1, 7, 13, 29]
CONDITIONS = ["Clean", "FGSM", "PGD"]
FILES = ["metrics.csv", "per_class.csv", "predictions.csv", "split_ids.json",
         "preprocessing.json", "preprocessing.npz", "confusion_matrices.npz",
         "verification.json", "training_complete.json", "runtime.json",
         "training_adapter.py", "manifest.json"]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def collect(run):
    completion = load(REPORT / "completion.json")
    assert completion["verified_completed_seeds"] == SEEDS
    assert completion["four_seed_package_complete"] and not completion["incomplete"]
    assert load(run / "configuration.json") == load(REPORT / "configuration.json")
    evidence = REPORT / "evidence"
    for seed in SEEDS:
        source = run / f"seed-{seed}"
        target = evidence / f"seed-{seed}"
        target.mkdir(parents=True, exist_ok=True)
        original_manifest = load(source / "manifest.json")
        for name in FILES:
            src, dst = source / name, target / name
            if name != "manifest.json":
                assert sha(src) == original_manifest[name], (seed, name)
            if dst.exists():
                assert sha(src) == sha(dst), f"Refusing to overwrite different evidence: {dst}"
            else:
                shutil.copy2(src, dst)
    provenance = load(REPORT / "provenance.json")
    provenance["checkpoint_locations"] = {
        str(s): f"runs/{run.name}/seed-{s}/checkpoint.pt" for s in SEEDS}
    provenance["paper_source"] = "excluded local PDF; obtain paper using DOI in REPORT.md"
    environment = load(REPORT / "environment.json")
    environment["executable"] = ".venv/Scripts/python.exe (original local interpreter)"
    environment.pop("git_status", None)
    completion["full_run_location"] = f"runs/{run.name}"
    metadata = {
        "description": "Portable derivative of original completion/environment/provenance records; originals retained locally, unchanged.",
        "original_records_sha256": {n: sha(REPORT / n) for n in
                                     ["completion.json", "environment.json", "provenance.json"]},
        "completion": completion, "environment": environment, "provenance": provenance,
    }
    write_json(REPORT / "review_metadata.json", metadata)
    write_json(evidence / "files_sha256.json", {
        p.relative_to(evidence).as_posix(): sha(p)
        for seed in SEEDS for p in sorted((evidence / f"seed-{seed}").iterdir())})


def close(a, b):
    assert math.isfinite(float(a)) and math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), (a, b)


def check_results():
    cfg = load(REPORT / "configuration.json")
    assert cfg == load(ROOT / "configs/milestone1_artifact_full.json")
    assert cfg["seeds"] == SEEDS and cfg["dataset"]["classes"] == 27
    meta = load(REPORT / "review_metadata.json")
    for name, expected in load(REPORT / "evidence/files_sha256.json").items():
        assert sha(REPORT / "evidence" / name) == expected, name
    all_metrics = rows(REPORT / "per_seed.csv")
    assert len(all_metrics) == 12
    assert {(int(r["seed"]), r["condition"]) for r in all_metrics} == {
        (s, c) for s in SEEDS for c in CONDITIONS}
    train_seconds = eval_seconds = 0
    for seed in SEEDS:
        base = REPORT / f"evidence/seed-{seed}"
        original_manifest = load(base / "manifest.json")
        assert sha(base / "manifest.json") == meta["provenance"]["seed_manifests"][str(seed)]
        for name in FILES:
            if name != "manifest.json":
                assert sha(base / name) == original_manifest[name], (seed, name)
        split = load(base / "split_ids.json")
        tr, va, ev = map(set, [split["train"], split["validation"], split["evaluation"]])
        assert len(tr) == 23997 and len(va) == 6000 and len(ev) == 3000
        assert not (tr & va) and tr | va == set(range(29997)) and ev <= va
        assert not split["test"] and split["split_seed"] == seed and split["cap_seed"] == 42
        assert split["dataset_sha256"] == cfg["dataset"]["sha256"]
        done = load(base / "training_complete.json")
        assert (done["epochs"], done["optimizer_steps"], done["samples_processed"]) == (15, 705, 359955)
        verification = load(base / "verification.json")
        assert verification["checkpoint_sha256"] == meta["provenance"]["checkpoint_sha256"][str(seed)]
        assert len(rows(base / "per_class.csv")) == 81
        for key in ["epsilon_zero_fgsm_identity", "epsilon_zero_pgd_identity",
                    "input_gradient_finite_nonzero", "evaluation_weights_unchanged",
                    "same_examples_all_conditions", "checkpoint_reload_logits_identical"]:
            assert verification[key] is True
        predictions = rows(base / "predictions.csv")
        assert len(predictions) == 9000
        grouped = {c: [r for r in predictions if r["condition"] == c] for c in CONDITIONS}
        clean = {int(r["row_id"]): r for r in grouped["Clean"]}
        assert len(clean) == 3000 and set(clean) == ev
        seed_metrics = rows(base / "metrics.csv")
        assert len(seed_metrics) == 3
        for condition in CONDITIONS:
            pred = grouped[condition]
            assert [int(r["row_id"]) for r in pred] == split["evaluation"]
            assert [r["row_id"] for r in pred] == [r["row_id"] for r in grouped["Clean"]]
            metric = next(r for r in all_metrics if int(r["seed"]) == seed and r["condition"] == condition)
            source_metric = next(r for r in seed_metrics if r["condition"] == condition)
            assert int(metric["rows"]) == 3000
            assert metric.keys() == source_metric.keys()
            for key in metric:
                if key in ["condition", "checkpoint_sha256"]:
                    assert metric[key] == source_metric[key]
                else:
                    close(metric[key], source_metric[key])
            cm = [[0] * 27 for _ in range(27)]
            numerator = denominator = 0
            for r in pred:
                row_id, y, p = map(int, [r["row_id"], r["label"], r["prediction"]])
                assert 0 <= y < 27 and 0 <= p < 27
                assert r["label"] == clean[row_id]["label"]
                assert r["checkpoint_sha256"] == verification["checkpoint_sha256"] == metric["checkpoint_sha256"]
                cm[y][p] += 1
                correct = clean[row_id]["prediction"] == r["label"]
                denominator += correct
                numerator += correct and p != y
            support = [sum(r) for r in cm]
            predicted = [sum(cm[i][j] for i in range(27)) for j in range(27)]
            trace = sum(cm[i][i] for i in range(27))
            f1 = []
            for i in range(27):
                precision = cm[i][i] / predicted[i] if predicted[i] else 0
                recall = cm[i][i] / support[i] if support[i] else 0
                value = 2 * precision * recall / (precision + recall) if precision + recall else 0
                f1.append(value)
                pc = next(r for r in rows(base / "per_class.csv") if r["condition"] == condition and int(r["class_id"]) == i)
                for name, actual in [("precision", precision), ("recall", recall), ("f1-score", value), ("support", support[i])]:
                    close(pc[name], actual)
            mcc = (trace * 3000 - sum(a*b for a, b in zip(support, predicted))) / math.sqrt(
                (3000**2 - sum(x*x for x in support)) * (3000**2 - sum(x*x for x in predicted)))
            for key, value in [("accuracy", trace/3000), ("macro_f1", statistics.mean(f1)),
                               ("weighted_f1", sum(a*b for a, b in zip(f1, support))/3000), ("mcc", mcc),
                               ("asr_numerator", numerator), ("asr_denominator", denominator)]:
                close(metric[key], value)
            if denominator:
                close(metric["asr_percent"], 100*numerator/denominator)
            else:
                assert metric["asr_percent"] == ""  # no manufactured zero
        runtime = load(base / "runtime.json")
        train_seconds += runtime["training_seconds_accumulated"]
        eval_seconds += sum(runtime["evaluation_seconds"].values())
    for metric in rows(REPORT / "summary.csv"):
        values = [float(r[metric["metric"]]) for r in all_metrics if r["condition"] == metric["condition"]]
        assert int(metric["n"]) == 4
        close(metric["mean"], statistics.mean(values))
        close(metric["sample_std"], statistics.stdev(values))
    close(meta["completion"]["training_seconds"], train_seconds)
    close(meta["completion"]["evaluation_seconds"], eval_seconds)
    report = (REPORT / "REPORT.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    for condition in CONDITIONS:
        records = [r for r in rows(REPORT / "summary.csv") if r["condition"] == condition]
        accuracy = next(r for r in records if r["metric"] == "accuracy")
        assert f'{100*float(accuracy["mean"]):.2f} ± {100*float(accuracy["sample_std"]):.2f}' in readme
        for r in records:
            scale = 100 if r["metric"] in ["accuracy", "macro_f1", "weighted_f1"] else 1
            assert f'{scale*float(r["mean"]):.3f} ± {scale*float(r["sample_std"]):.3f}' in report
    assert "Unresolved / not demonstrated" in report


def check_links():
    files = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]).decode().split("\0")
    checked = 0
    for name in set(files):
        path = ROOT / name
        if path.suffix.lower() != ".md" or not path.is_file():
            continue
        for link in re.findall(r'!?\[[^\]]*\]\(([^)\n]+)\)', path.read_text(encoding="utf-8-sig")):
            link = unquote(link.split("#", 1)[0].strip().strip("<>"))
            if not link or re.match(r"[a-z]+:", link, re.I):
                continue
            assert (path.parent / link).exists(), (name, link)
            checked += 1
    return checked


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without -O: review assertions must remain enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="Copy compact evidence and derive portable metadata; requires full local records")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/milestone1-artifact-full")
    args = parser.parse_args()
    if args.collect:
        collect(args.run_dir.resolve())
    check_results()
    print(f"PASS: 4 seeds, 12 condition rows, 36000 aligned predictions, per-class metrics, sample SD, runtime, evidence hashes; {check_links()} local Markdown links")
