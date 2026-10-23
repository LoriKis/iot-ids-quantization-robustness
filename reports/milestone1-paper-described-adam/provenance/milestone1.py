"""Milestone 1 target gate and bounded smoke of released Transformer helpers.

Route A is unresolved and cannot train. Route B smoke is a component check,
with explicit budget/seeding changes; it is never a Table 4 reproduction.
This module does not import or change the corrected research pipeline.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PIN = "e73911bbdb241f3fb6c5de346139010c4aadefd4"
DATA_SHA = "0d869e334cc9a2795c96996fbde8fb04a8acac61f5e67c5d46aee2b0717a53ed"
TARGET = ROOT / "configs/milestone1_target.json"
REQUIRED = {
    "configuration.json", "preprocessing.json", "preprocessing.npz",
    "split_ids.json", "dataset.json", "checkpoint.pt", "predictions.npz",
    "predictions.csv", "metrics.json", "runtime.json", "verification.json",
    "provenance.json", "raw.log", "smoke_inputs.npz",
}


def read_target(path=TARGET):
    target = json.loads(Path(path).read_text(encoding="utf-8"))
    if target.get("schema_version") != 1 or target.get("dataset") != "CICIoT2023" or target.get("model") != "Standard Transformer":
        raise ValueError("Unsupported Milestone 1 target configuration")
    if target.get("status") != "unresolved" or not target.get("unresolved"):
        raise ValueError("This entry point only supports the documented unresolved target; review the scientific evidence before implementing route A")
    return target


def validate_request(route, smoke, dry_run):
    read_target()
    if route == "paper" and not dry_run:
        raise ValueError("Exact paper reproduction is blocked: Table 4 mode, data/splits, metrics, seeds, sequence construction and checkpoint rule remain unresolved. See docs/MILESTONE1_REFERENCE.md. No training started.")
    if route == "artifact" and not (smoke or dry_run):
        raise ValueError("Only --smoke is executable this turn; full-budget training is disabled")
    if smoke and dry_run:
        raise ValueError("Choose --smoke or --dry-run")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def verify_bundle(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not REQUIRED.issubset(manifest):
        raise ValueError("Incomplete smoke manifest")
    for name, digest in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha(path) != digest:
            raise ValueError(f"Missing or changed smoke artifact: {name}")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if set(metrics) != {"Clean", "FGSM", "PGD"}:
        raise ValueError("Expected clean/FGSM/PGD from one checkpoint")
    splits = json.loads((root / "split_ids.json").read_text(encoding="utf-8"))
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    all_ids = splits["train"] + splits["validation"]
    if len(all_ids) != dataset["rows"] or sorted(all_ids) != list(range(dataset["rows"])):
        raise ValueError("Split identifiers must partition the released CSV")
    if not set(splits["smoke_train"]).issubset(splits["train"]) or not set(splits["smoke_evaluation"]).issubset(splits["validation"]):
        raise ValueError("Smoke identifiers do not match source partitions")
    with (root / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    expected = {(attack, rid) for attack in metrics for rid in splits["smoke_evaluation"]}
    actual = {(r["attack"], int(r["row_id"])) for r in records}
    if len(records) != len(expected) or actual != expected:
        raise ValueError("Incomplete or duplicate predictions")
    if any(r["checkpoint_sha256"] != manifest["checkpoint.pt"] for r in records) or any(m["checkpoint_sha256"] != manifest["checkpoint.pt"] for m in metrics.values()):
        raise ValueError("Attack results must share the saved checkpoint")
    return manifest


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_smoke(root):
    # Scientific packages are deliberately lazy: the route-A gate works before imports.
    import numpy as np
    import pandas as pd
    import torch
    import sklearn
    from sklearn.model_selection import train_test_split

    upstream = ROOT / "upstream/PAVE_artifact"
    if git("rev-parse", "HEAD", cwd=upstream) != PIN or git("status", "--porcelain", "--untracked-files=all", cwd=upstream):
        raise ValueError("Pinned upstream must remain at the expected clean revision")
    data_path = upstream / "data/balanced_CICIoT.csv"
    if sha(data_path) != DATA_SHA:
        raise ValueError("Unexpected released CICIoT data hash")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "status.json", {"state": "running", "claim": "B component smoke only; NOT a scientific reproduction"})
    try:
        with (root / "raw.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            _smoke(root, upstream, data_path, np, pd, torch, sklearn, train_test_split)
        write_json(root / "manifest.json", {name: sha(root / name) for name in sorted(REQUIRED)})
        verify_bundle(root)
        if git("status", "--porcelain", "--untracked-files=all", cwd=upstream):
            raise RuntimeError("Pinned upstream changed during smoke")
        write_json(root / "status.json", {"state": "complete", "claim": "B component smoke only; NOT Table 4", "manifest_verified": True})
    except BaseException as exc:
        write_json(root / "status.json", {"state": "failed", "error": str(exc)})
        raise


def _smoke(root, upstream, data_path, np, pd, torch, sklearn, train_test_split):
    started = time.perf_counter()
    print("B COMPONENT SMOKE ONLY: CICIoT / Standard Transformer / one seed / one epoch", flush=True)
    reference = root / "reference/code"
    shutil.copytree(upstream / "code", reference, ignore=shutil.ignore_patterns("__pycache__"))
    # Import side effects and bytecode stay outside the pinned checkout.
    os.environ["AGT_OUT"] = str(root / "upstream_auxiliary")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference))
    infra = importlib.import_module("sota_compare")
    infra.BAL = str(data_path.parent)  # path compatibility only; no source changes
    infra.DEV = "cpu"
    torch.set_num_threads(2)
    torch.manual_seed(1)
    np.random.seed(1)
    config = {
        "route": "B component smoke", "paper_reproduction": False,
        "dataset": "CICIoT", "mode": "multiclass", "model": "StandardTransformerIDS",
        "seed": 1, "epochs": 1, "train_cap": 54, "eval_cap": 27,
        "train_batch_size": 18, "eval_attack_batch_size": 9, "device": "cpu", "torch_threads": 2,
        "optimizer": "upstream train_seq: Adam(lr=0.001, weight_decay=0)",
        "preprocessing": "upstream get_data: whole-file StandardScaler, stratified 80/20",
        "training_sequence": "upstream: repeat row 50 times plus Gaussian std=0.01",
        "evaluation_sequence": "upstream expand: repeat row 50 times without noise",
        "fgsm_epsilon": infra.EPS, "pgd": infra.PGD,
        "pgd_random_start": False, "domain_clipping": False,
        "checkpoint_rule": "final epoch (smoke only)",
        "explicit_deviations_from_full_release_orchestration": [
            "Only Standard Transformer helper path; no surrogate, AGT, GRU, defenses or other datasets",
            "One epoch, stratified 54/27 row caps and reduced batch sizes",
            "Explicit torch/numpy seed 1 and CPU threads 2; release orchestration does not seed torch",
            "Copied code, redirected AGT_OUT and data path; no changes to upstream source",
            "Added saving/reload/checksums and distinctly named metrics; no Table 4 metric equivalence assumed"
        ]
    }
    write_json(root / "configuration.json", config)
    data_started = time.perf_counter()
    d = infra.get_data("CICIoT", 1, "multiclass")
    frame = pd.read_csv(data_path)
    classes = list(map(str, d["class_names"]))
    y_all = np.array([classes.index(str(label)) for label in frame.Label])
    tr_ids, va_ids = train_test_split(np.arange(len(frame)), test_size=0.2, random_state=1, stratify=y_all)
    numeric = frame.drop(columns="Label").apply(pd.to_numeric, errors="coerce").fillna(0).replace([np.inf, -np.inf], 0)
    scaled = d["scaler"].transform(numeric.values)
    assert np.array_equal(y_all[tr_ids], d["y_train"]) and np.array_equal(y_all[va_ids], d["y_val"])
    assert np.array_equal(scaled[tr_ids], d["X_train_raw"]) and np.array_equal(scaled[va_ids], d["X_val_raw"])
    def select(y, cap):
        positions = np.arange(len(y))[:, None]
        return infra.cap_aligned(positions, positions, y, n=cap, seed=42)[0].ravel()
    ti = select(d["y_train"], 54)
    vi = select(d["y_val"], 27)
    xt, yt = d["X_train_seq"][ti], np.asarray(d["y_train"])[ti]
    xc, yv = infra.expand(d["X_val_raw"][vi]), np.asarray(d["y_val"])[vi]
    assert (len(xt), len(xc), d["n_classes"], d["n_features"]) == (54, 27, 27, 39)
    assert not set(tr_ids).intersection(va_ids)
    write_json(root / "split_ids.json", {"row_id_definition": "zero-based CSV data row; header excluded; duplicates keep distinct IDs", "dataset_sha256": DATA_SHA, "train": tr_ids.tolist(), "validation": va_ids.tolist(), "test": None, "smoke_train": tr_ids[ti].tolist(), "smoke_evaluation": va_ids[vi].tolist(), "no_held_out_test_in_release": True})
    write_json(root / "dataset.json", {"sha256": DATA_SHA, "rows": len(frame), "features": d["feature_names"], "classes": dict(enumerate(classes)), "class_counts": {str(k): int(v) for k,v in frame.Label.value_counts().items()}, "benign_fraction": float((frame.Label == "Benign_Final").mean())})
    sc = d["scaler"]
    write_json(root / "preprocessing.json", {"type": "sklearn.preprocessing.StandardScaler", "fit_scope": "all 29997 rows BEFORE split (released behavior, not paper)", "n_samples_seen": int(sc.n_samples_seen_), "feature_order": d["feature_names"], "cleaning": "numeric coerce, missing/infinite -> 0"})
    np.savez_compressed(root / "preprocessing.npz", mean=sc.mean_, scale=sc.scale_, var=sc.var_)
    with np.load(root / "preprocessing.npz") as restored:
        restored_clean = ((numeric.values[va_ids[vi]] - restored["mean"]) / restored["scale"]).astype(np.float32)
        np.testing.assert_array_equal(restored_clean, xc[:, 0, :])
    np.savez_compressed(root / "smoke_inputs.npz", train=xt, train_labels=yt, clean=xc, labels=yv, train_ids=tr_ids[ti], evaluation_ids=va_ids[vi])
    data_seconds = time.perf_counter() - data_started
    model_config = dict(input_dim=39, d_model=96, n_heads=4, n_layers=3, d_ff=384, n_classes=27, max_seq_len=50, dropout=0.1)
    model = infra.StandardTransformerIDS(**model_config).to("cpu")
    train_started = time.perf_counter()
    infra.train_seq(model, xt, yt, epochs=1, bs=18)
    train_seconds = time.perf_counter() - train_started
    torch.save({"state_dict": model.state_dict(), "model_config": model_config, "claim": "B component smoke only", "epoch": 1}, root / "checkpoint.pt")
    before = infra.seq_logits(model, xc, bs=9)
    saved = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True)
    loaded = infra.StandardTransformerIDS(**saved["model_config"]).to("cpu")
    loaded.load_state_dict(saved["state_dict"], strict=True)
    loaded.eval()
    np.testing.assert_array_equal(before, infra.seq_logits(loaded, xc, bs=9))
    checkpoint_sha = sha(root / "checkpoint.pt")
    outputs, durations, logits = {}, {}, {}
    clean_pred = before.argmax(1)
    for name in ["Clean", "FGSM", "PGD"]:
        tick = time.perf_counter()
        attacked = xc if name == "Clean" else infra.seq_fgsm(loaded, xc, yv, infra.EPS, bs=9) if name == "FGSM" else infra.seq_pgd(loaded, xc, yv, **infra.PGD, bs=9)
        if name != "Clean":
            assert float(np.max(np.abs(attacked - xc))) <= infra.EPS + 1e-5
        logits[name] = infra.seq_logits(loaded, attacked, bs=9)
        pred = logits[name].argmax(1)
        correct = clean_pred == yv
        numerator = int(np.count_nonzero(correct & (pred != yv)))
        denominator = int(correct.sum())
        raw = infra.metrics(yv, pred, 27)
        # ASR names are definitions, not assertions about Table 4.
        outputs[name] = {"upstream_metrics": raw, "upstream_f1_average": "weighted", "checkpoint_sha256": checkpoint_sha,
                         "correct_clean_to_wrong_count": numerator, "clean_correct_count": denominator,
                         "asr_clean_correct_percent": 100 * numerator / denominator if denominator else None,
                         "prediction_change_all_percent": float((pred != clean_pred).mean()) * 100,
                         "adversarial_error_all_percent": float((pred != yv).mean()) * 100,
                         "paper_asr_definition_verified": False, "rows": len(yv)}
        durations[name] = time.perf_counter() - tick
        print(name, json.dumps(outputs[name]), flush=True)
    np.savez_compressed(root / "predictions.npz", labels=yv, row_ids=va_ids[vi], **logits)
    records = []
    for name, values in logits.items():
        records += [{"attack": name, "row_id": int(rid), "label": int(label), "prediction": int(pred), "checkpoint_sha256": checkpoint_sha} for rid,label,pred in zip(va_ids[vi], yv, values.argmax(1))]
    pd.DataFrame(records).to_csv(root / "predictions.csv", index=False)
    write_json(root / "metrics.json", outputs)
    write_json(root / "runtime.json", {"data_seconds": data_seconds, "train_seconds": train_seconds, "evaluation_seconds": durations, "total_after_imports_seconds": time.perf_counter() - started, "train_rows": 54, "train_batches": 3, "evaluation_rows": 27, "full_reference_estimate_seconds": None, "estimate_status": "provisional only: reference sample size, sequences, checkpoint protocol and mode unresolved; smoke batch sizes differ from full reference. No defensible full-seed estimate."})
    write_json(root / "verification.json", {"dataset_model_selection": "CICIoT / 27-class Standard Transformer only", "actual_upstream_split_rows_verified": True, "checkpoint_reload_logits_identical": True, "same_checkpoint_all_attacks": True, "attack_linf_bound_verified": True, "eval_rows_per_class": 1, "training_rows_per_class": 2, "scientific_result": False})
    write_json(root / "provenance.json", {"local_revision": git("rev-parse", "HEAD"), "local_dirty_status": git("status", "--porcelain"), "upstream_revision": PIN, "dataset_sha256": DATA_SHA, "python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "entry_point_sha256": sha(Path(__file__)), "upstream_code_sha256": {p.relative_to(upstream).as_posix(): sha(p) for p in sorted((upstream / "code").rglob("*.py"))}, "target_sha256": sha(TARGET), "command": sys.argv})
    print("Smoke checks passed; artifacts saved. Paper target remains unresolved.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=["paper", "artifact"], default="paper")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--verify", type=Path, help="Verify saved smoke checksums without training")
    args = parser.parse_args()
    if args.verify:
        verify_bundle(args.verify)
        print("Smoke bundle checksums and required artifacts verified")
        return
    try:
        validate_request(args.route, args.smoke, args.dry_run)
        if args.dry_run:
            print(json.dumps({"route": args.route, "training_started": False, "target": read_target(), "artifact_smoke_budget": "54 train / 27 validation, 1 epoch, seed 1, Standard Transformer only"}, indent=2))
        elif args.run_dir is None:
            parser.error("--run-dir is required for artifact smoke")
        else:
            run_smoke(args.run_dir)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
