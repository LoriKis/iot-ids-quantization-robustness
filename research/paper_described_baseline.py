"""Executable runner for Milestone 1 Paper-Described Configuration.

Follows settings described in the PAVE paper as closely as evidence allows:
- MinMaxScaler fitted on training partition ONLY (Section 5.2)
- Stratified 70% train / 15% val / 15% test split (split_seed=2025, Section 5.1)
- AdamW lr=0.001, weight_decay=0.0001 (Section 5.1)
- 25 epochs, batch size 512 (Section 5.1)
- FGSM: eps=0.10, domain clipping to normalized feature bounds [0, 1] (Section 5.4)
- PGD: eps=0.10, 7 steps, alpha=0.025 (eps/4), random start, domain clipping (Section 5.4)
- Standard Transformer 96/4/3/384 (Section 5.3 / Table 3)
- Evaluated on full 4,500 held-out test rows across seeds 1, 7, 13, 29

Multiclass 27-class experiment with documented assumptions.
Never asserted to be an exact reproduction of Table 4.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import psutil
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from .milestone1 import ROOT, PIN, DATA_SHA, Tee, git, sha

CLAIM = "Paper-described configuration with documented assumptions; multiclass 27-class experiment; not a verified Table 4 reproduction."
DEFAULT_CONFIG = ROOT / "configs/milestone1_paper_described.json"


def atomic_json(path: Path | str, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_save(path: Path | str, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, val in model.state_dict().items():
        h.update(key.encode())
        h.update(val.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def read_config(path: Path | str) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = (
        cfg["dataset"]["sha256"],
        cfg["seeds"],
        cfg["split_seed"],
        cfg["training"]["epochs"],
        cfg["training"]["batch_size"],
        cfg["training"]["learning_rate"],
        cfg["training"]["weight_decay"],
        cfg["evaluation"]["test_rows"],
    )
    if expected != (DATA_SHA, [1, 7, 13, 29], 2025, 25, 512, 0.001, 0.0001, 4500):
        raise ValueError("Expected the frozen paper-described configuration, not modified settings")
    return cfg


def import_reference(root: Path, cfg: dict):
    upstream = ROOT / "upstream/PAVE_artifact"
    if git("rev-parse", "HEAD", cwd=upstream) != PIN or git("status", "--porcelain", "--untracked-files=all", cwd=upstream):
        raise ValueError("Pinned upstream changed")
    if sha(upstream / "data/balanced_CICIoT.csv") != DATA_SHA:
        raise ValueError("Dataset hash changed")
    reference = root / "reference"
    if not reference.exists():
        shutil.copytree(upstream / "code", reference / "code", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(upstream / "LICENSE", reference / "LICENSE")
    sys.dont_write_bytecode = True
    if str(reference / "code") not in sys.path:
        sys.path.insert(0, str(reference / "code"))
    os.environ["AGT_OUT"] = str(root / "upstream_auxiliary")
    infra = importlib.import_module("sota_compare")
    infra.BAL = str(upstream / "data")
    infra.DEV = "cpu"
    return infra


class MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.peak = 0
        self.min_available = psutil.virtual_memory().available
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, psutil.Process().memory_info().rss)
            self.min_available = min(self.min_available, psutil.virtual_memory().available)
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def audit_splits(features_df: pd.DataFrame, labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    """Audit duplicate feature overlap between partitions without modifying the split."""
    feat_tuples = [tuple(row) for row in features_df.values]
    train_feat_set = set(feat_tuples[i] for i in train_idx)
    val_feat_set = set(feat_tuples[i] for i in val_idx)
    test_feat_set = set(feat_tuples[i] for i in test_idx)

    test_in_train = sum(1 for i in test_idx if feat_tuples[i] in train_feat_set)
    val_in_train = sum(1 for i in val_idx if feat_tuples[i] in train_feat_set)
    test_in_val = sum(1 for i in test_idx if feat_tuples[i] in val_feat_set)

    labeled_tuples = [(tuple(row), label) for row, label in zip(features_df.values, labels)]
    train_labeled_set = set(labeled_tuples[i] for i in train_idx)
    test_in_train_labeled = sum(1 for i in test_idx if labeled_tuples[i] in train_labeled_set)

    return {
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "test_rows": len(test_idx),
        "unique_feature_patterns_total": len(set(feat_tuples)),
        "unique_feature_patterns_train": len(train_feat_set),
        "unique_feature_patterns_val": len(val_feat_set),
        "unique_feature_patterns_test": len(test_feat_set),
        "test_rows_matching_train_features": test_in_train,
        "test_rows_matching_train_features_percent": 100.0 * test_in_train / len(test_idx),
        "test_rows_matching_train_labeled": test_in_train_labeled,
        "test_rows_matching_train_labeled_percent": 100.0 * test_in_train_labeled / len(test_idx),
        "val_rows_matching_train_features": val_in_train,
        "val_rows_matching_train_features_percent": 100.0 * val_in_train / len(val_idx),
        "note": "Row IDs disjoint by construction; overlap reflects duplicate records present in source CSV",
    }


def prepare_data(infra, cfg: dict, root: Path):
    """Prepare single stratified 70/15/15 split and fit MinMaxScaler on training partition only."""
    csv_path = Path(infra.BAL) / "balanced_CICIoT.csv"
    if sha(csv_path) != DATA_SHA:
        raise ValueError("Dataset hash mismatch")
    df = pd.read_csv(csv_path)
    assert len(df) == cfg["dataset"]["rows"]

    classes = sorted(df["Label"].unique().tolist())
    assert classes == cfg["dataset"]["class_names"]
    le = LabelEncoder()
    labels = le.fit_transform(df["Label"])

    numeric = df.drop(columns="Label").apply(pd.to_numeric, errors="coerce").fillna(0.0).replace([np.inf, -np.inf], 0.0)
    assert list(numeric.columns) == cfg["dataset"]["feature_names"]
    assert numeric.shape[1] == cfg["dataset"]["features"]

    split_seed = cfg["split_seed"]
    indices = np.arange(len(df))
    test_count = cfg["evaluation"]["test_rows"]  # 4500
    val_count = cfg["partition"]["validation_rows"]  # 4500
    train_count = cfg["partition"]["train_rows"]  # 20997

    # Two-pass stratified split:
    train_val_idx, test_idx = train_test_split(indices, test_size=test_count, random_state=split_seed, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_count, random_state=split_seed, stratify=labels[train_val_idx])

    assert len(train_idx) == train_count and len(val_idx) == val_count and len(test_idx) == test_count
    assert not (set(train_idx) & set(val_idx))
    assert not (set(train_idx) & set(test_idx))
    assert not (set(val_idx) & set(test_idx))
    assert set(train_idx) | set(val_idx) | set(test_idx) == set(indices)

    # Duplicate overlap audit
    audit = audit_splits(numeric, labels, train_idx, val_idx, test_idx)
    atomic_json(root / "split_overlap_audit.json", audit)

    split_metadata = {
        "split_seed": split_seed,
        "train_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "test_rows": len(test_idx),
        "train": train_idx.tolist(),
        "validation": val_idx.tolist(),
        "test": test_idx.tolist(),
        "dataset_sha256": DATA_SHA,
        "row_id_semantics": "zero-based CSV row; disjoint row IDs across partitions",
        "overlap_audit": audit,
    }
    atomic_json(root / "split_ids.json", split_metadata)

    # FIT MINMAXSCALER ON TRAINING DATA ONLY
    scaler = MinMaxScaler()
    X_tr_raw = numeric.iloc[train_idx].values
    X_va_raw = numeric.iloc[val_idx].values
    X_te_raw = numeric.iloc[test_idx].values

    scaler.fit(X_tr_raw)

    # Check constant features in training
    zero_var_features = np.where(scaler.data_min_ == scaler.data_max_)[0]
    assert len(zero_var_features) == 0, f"Constant features detected in training: {zero_var_features}"

    X_tr_norm = scaler.transform(X_tr_raw).astype(np.float32)
    X_va_norm = scaler.transform(X_va_raw).astype(np.float32)
    X_te_norm = scaler.transform(X_te_raw).astype(np.float32)

    # Record out-of-training-range occurrences
    va_out = int(((X_va_norm < 0) | (X_va_norm > 1)).sum())
    te_out = int(((X_te_norm < 0) | (X_te_norm > 1)).sum())

    preprocessing_meta = {
        "type": "MinMaxScaler",
        "fit_scope": "training partition ONLY (Section 5.2)",
        "train_rows_seen": int(scaler.n_samples_seen_),
        "features": list(numeric.columns),
        "class_mapping": dict(enumerate(classes)),
        "data_min_train": scaler.data_min_.tolist(),
        "data_max_train": scaler.data_max_.tolist(),
        "validation_values_outside_0_1": va_out,
        "test_values_outside_0_1": te_out,
        "constant_features": len(zero_var_features),
        "missing_values_imputation": "filled with 0.0 prior to scaling",
        "domain_clipping_bounds": "[0.0, 1.0] per feature in normalized space",
    }
    atomic_json(root / "preprocessing.json", preprocessing_meta)
    np.savez_compressed(
        root / "preprocessing.npz",
        data_min=scaler.data_min_,
        data_max=scaler.data_max_,
        data_range=scaler.data_range_,
        scale=scaler.scale_,
        min=scaler.min_,
    )

    data_bundle = {
        "df": df,
        "numeric": numeric,
        "labels": labels,
        "classes": classes,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "X_tr_norm": X_tr_norm,
        "y_tr": labels[train_idx],
        "X_va_norm": X_va_norm,
        "y_va": labels[val_idx],
        "X_te_norm": X_te_norm,
        "y_te": labels[test_idx],
        "scaler": scaler,
    }
    return data_bundle


def fgsm_attack(model: torch.nn.Module, x_seq: np.ndarray, y: np.ndarray, eps: float = 0.10, batch_size: int = 64) -> np.ndarray:
    """FGSM with per-feature domain clipping to [0, 1] in normalized sequence space. model.eval() throughout."""
    model.eval()
    adv_chunks = []
    yt_all = torch.tensor(y, dtype=torch.long)
    for i in range(0, len(x_seq), batch_size):
        xb = torch.tensor(x_seq[i : i + batch_size], dtype=torch.float32).requires_grad_(True)
        yb = yt_all[i : i + batch_size]
        logits = model(xb)["logits"]
        loss = F.cross_entropy(logits, yb)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            x_adv = xb + eps * xb.grad.sign()
            # Domain clipping to normalized bounds [0, 1]
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            # Projection verification
            x_adv = torch.clamp(x_adv, xb - eps, xb + eps)
        adv_chunks.append(x_adv.cpu().numpy())
    return np.concatenate(adv_chunks, axis=0)


def pgd_attack(
    model: torch.nn.Module,
    x_seq: np.ndarray,
    y: np.ndarray,
    seed: int,
    eps: float = 0.10,
    alpha: float = 0.025,
    iters: int = 7,
    batch_size: int = 64,
) -> np.ndarray:
    """PGD (Section 5.4: eps=0.10, 7 steps, alpha=eps/4=0.025, random start, domain clipping [0,1]). model.eval() throughout."""
    model.eval()
    adv_chunks = []
    yt_all = torch.tensor(y, dtype=torch.long)
    for batch_idx, i in enumerate(range(0, len(x_seq), batch_size)):
        x0 = torch.tensor(x_seq[i : i + batch_size], dtype=torch.float32)
        yb = yt_all[i : i + batch_size]
        # Deterministic random start seeded per batch
        torch.manual_seed(seed * 10000 + batch_idx)
        noise = torch.empty_like(x0).uniform_(-eps, eps)
        xb = torch.clamp(x0 + noise, x0 - eps, x0 + eps)
        xb = torch.clamp(xb, 0.0, 1.0)

        for _ in range(iters):
            xb = xb.clone().detach().requires_grad_(True)
            logits = model(xb)["logits"]
            loss = F.cross_entropy(logits, yb)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                xb = xb + alpha * xb.grad.sign()
                xb = torch.clamp(xb, x0 - eps, x0 + eps)  # epsilon-box projection
                xb = torch.clamp(xb, 0.0, 1.0)  # domain clipping to normalized bounds
            xb = xb.detach()
        adv_chunks.append(xb.cpu().numpy())
    return np.concatenate(adv_chunks, axis=0)


def compute_logits(model: torch.nn.Module, x_seq: np.ndarray, batch_size: int = 64) -> np.ndarray:
    model.eval()
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(x_seq), batch_size):
            xb = torch.tensor(x_seq[i : i + batch_size], dtype=torch.float32)
            logits_list.append(model(xb)["logits"].cpu())
    return torch.cat(logits_list, dim=0).numpy()


def evaluate_seed(
    infra,
    cfg: dict,
    model: torch.nn.Module,
    x_test_seq: np.ndarray,
    y_test: np.ndarray,
    test_ids: np.ndarray,
    folder: Path,
    checkpoint_sha: str,
    seed: int,
) -> list[dict]:
    """Evaluate trained checkpoint under Clean, FGSM and PGD on full test partition."""
    model.eval()
    before_hash = state_hash(model)
    eval_batch = cfg["resources"].get("evaluation_batch_size", 64)

    # Diagnostic probe on 2 examples
    probe_x = x_test_seq[:2]
    probe_y = y_test[:2]
    fg0 = fgsm_attack(model, probe_x, probe_y, eps=0.0, batch_size=2)
    pg0 = pgd_attack(model, probe_x, probe_y, seed=seed, eps=0.0, alpha=0.025, iters=7, batch_size=2)
    np.testing.assert_array_equal(fg0, probe_x)
    np.testing.assert_array_equal(pg0, probe_x)

    xt = torch.tensor(probe_x, requires_grad=True)
    loss = F.cross_entropy(model(xt)["logits"], torch.tensor(probe_y, dtype=torch.long))
    grad = torch.autograd.grad(loss, xt)[0]
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    assert state_hash(model) == before_hash

    eval_dir = folder / "evaluations"
    eval_dir.mkdir(exist_ok=True)
    results = {}

    for name in ["Clean", "FGSM", "PGD"]:
        dest = eval_dir / f"{name}.npz"
        meta_path = dest.with_suffix(".json")
        if dest.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if sha(dest) == meta["sha256"] and meta["checkpoint_sha256"] == checkpoint_sha:
                with np.load(dest) as saved:
                    np.testing.assert_array_equal(saved["row_ids"], test_ids)
                    np.testing.assert_array_equal(saved["labels"], y_test)
                    results[name] = {"logits": saved["logits"].copy(), "meta": meta}
                continue

        tick = time.perf_counter()
        if name == "Clean":
            attacked = x_test_seq
            bounds = np.zeros(len(x_test_seq), dtype=np.float32)
        elif name == "FGSM":
            attacked = fgsm_attack(model, x_test_seq, y_test, eps=cfg["evaluation"]["fgsm"]["epsilon"], batch_size=eval_batch)
            bounds = np.max(np.abs(attacked - x_test_seq), axis=(1, 2))
        else:  # PGD
            attacked = pgd_attack(
                model,
                x_test_seq,
                y_test,
                seed=seed,
                eps=cfg["evaluation"]["pgd"]["epsilon"],
                alpha=cfg["evaluation"]["pgd"]["alpha"],
                iters=cfg["evaluation"]["pgd"]["iterations"],
                batch_size=eval_batch,
            )
            bounds = np.max(np.abs(attacked - x_test_seq), axis=(1, 2))

        if not np.isfinite(attacked).all() or bounds.max() > 0.10001:
            raise ValueError(f"{name}: finite input or Linf bound violation (max {bounds.max()})")
        # Domain bounds check:
        if attacked.min() < -1e-6 or attacked.max() > 1.0 + 1e-6:
            raise ValueError(f"{name}: domain clipping violation (min {attacked.min()}, max {attacked.max()})")

        logits = compute_logits(model, attacked, batch_size=eval_batch)
        if not np.isfinite(logits).all() or state_hash(model) != before_hash or model.training:
            raise ValueError(f"{name}: evaluation altered model parameters or mode, or non-finite logits")

        duration = time.perf_counter() - tick
        np.savez_compressed(dest, labels=y_test, row_ids=test_ids, logits=logits, linf=bounds)
        meta = {
            "condition": name,
            "rows": len(y_test),
            "seconds": duration,
            "sha256": sha(dest),
            "checkpoint_sha256": checkpoint_sha,
            "max_linf": float(bounds.max()),
            "min_val": float(attacked.min()),
            "max_val": float(attacked.max()),
            "model_mode": "eval",
            "parameters_unchanged": True,
        }
        atomic_json(meta_path, meta)
        results[name] = {"logits": logits, "meta": meta}
        print(f"EVALUATED {name}: {meta['seconds']:.2f}s; max Linf {meta['max_linf']:.8f}; domain [{meta['min_val']:.4f}, {meta['max_val']:.4f}]", flush=True)

    audit_mod = importlib.import_module("pave_audit")
    clean_preds = results["Clean"]["logits"].argmax(axis=1)
    clean_correct = clean_preds == y_test
    denominator = int(clean_correct.sum())

    metrics, class_rows, predictions, matrices = [], [], [], {}
    for name, result in results.items():
        pred = result["logits"].argmax(axis=1)
        numerator = int((clean_correct & (pred != y_test)).sum())
        row = {
            "seed": seed,
            "condition": name,
            "rows": len(y_test),
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, labels=np.arange(27), average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, pred)),
            "asr_numerator": numerator,
            "asr_denominator": denominator,
            "asr_percent": 100.0 * numerator / denominator if denominator > 0 else None,
            "upstream_pave_audit_asr_percent": float(audit_mod._asr(y_test, clean_preds, pred)),
            "prediction_change_all_percent": float((pred != clean_preds).mean()) * 100.0,
            "all_row_error_percent": float((pred != y_test).mean()) * 100.0,
            "checkpoint_sha256": checkpoint_sha,
        }
        for k in ["accuracy", "macro_f1", "weighted_f1", "mcc"]:
            if not np.isfinite(row[k]):
                raise ValueError(f"Non-finite metric: {k}")
        metrics.append(row)

        report = classification_report(y_test, pred, labels=np.arange(27), output_dict=True, zero_division=0)
        for c in range(27):
            class_rows.append({
                "seed": seed,
                "condition": name,
                "class_id": c,
                "class_name": cfg["dataset"]["class_names"][c],
                **report[str(c)],
            })
        matrices[name] = confusion_matrix(y_test, pred, labels=np.arange(27))
        for r, y_true, p, cp in zip(test_ids, y_test, pred, clean_preds):
            predictions.append({
                "condition": name,
                "row_id": int(r),
                "label": int(y_true),
                "prediction": int(p),
                "clean_prediction": int(cp),
                "checkpoint_sha256": checkpoint_sha,
            })

    pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
    atomic_json(folder / "metrics.json", metrics)
    pd.DataFrame(class_rows).to_csv(folder / "per_class.csv", index=False)
    pd.DataFrame(predictions).to_csv(folder / "predictions.csv", index=False)
    np.savez_compressed(folder / "confusion_matrices.npz", **matrices)

    atomic_json(
        folder / "verification.json",
        {
            "epsilon_zero_fgsm_identity": True,
            "epsilon_zero_pgd_identity": True,
            "input_gradient_finite_nonzero": True,
            "input_gradient_max_abs": float(grad.abs().max()),
            "evaluation_weights_unchanged": state_hash(model) == before_hash,
            "evaluation_mode": "eval",
            "same_examples_all_conditions": True,
            "checkpoint_sha256": checkpoint_sha,
            "test_rows_evaluated": len(y_test),
        },
    )
    return metrics


def verify_seed(folder: Path, cfg: dict) -> pd.DataFrame:
    """Independently audit and verify all saved artifacts and outputs for a completed seed."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "checkpoint.pt",
        "metrics.csv",
        "metrics.json",
        "per_class.csv",
        "predictions.csv",
        "split_ids.json",
        "split_overlap_audit.json",
        "preprocessing.npz",
        "preprocessing.json",
        "confusion_matrices.npz",
        "verification.json",
        "runtime.json",
        "training_complete.json",
    }
    if not required.issubset(manifest):
        raise ValueError(f"Missing required artifacts in {folder}: {required - set(manifest)}")

    for name, digest in manifest.items():
        p = (folder / name).resolve()
        if not p.is_file() or sha(p) != digest:
            raise ValueError(f"Corrupted or modified artifact: {p}")

    training = json.loads((folder / "training_complete.json").read_text(encoding="utf-8"))
    if (training["epochs"], training["train_rows"], training["optimizer_steps"], training["configuration_hash"]) != (
        cfg["training"]["epochs"],
        cfg["partition"]["train_rows"],
        cfg["training"]["optimizer_steps_total"],
        config_hash(cfg),
    ):
        raise ValueError("Training budget mismatch in training_complete.json")

    epoch_files = sorted((folder / "epochs").glob("epoch-*.pt"))
    if len(epoch_files) != cfg["training"]["epochs"]:
        raise ValueError(f"Expected {cfg['training']['epochs']} epoch checkpoints, found {len(epoch_files)}")

    final_training_state = torch.load(epoch_files[-1], map_location="cpu", weights_only=True)
    steps = {int(v["step"].item()) for v in final_training_state["optimizer"]["state"].values() if "step" in v}
    if final_training_state["epoch"] != cfg["training"]["epochs"] or steps != {cfg["training"]["optimizer_steps_total"]}:
        raise ValueError(f"Saved optimizer step count {steps} does not match expected {cfg['training']['optimizer_steps_total']}")

    checks = json.loads((folder / "verification.json").read_text(encoding="utf-8"))
    if not checks.get("checkpoint_reload_logits_identical") or not checks.get("evaluation_weights_unchanged"):
        raise ValueError("Checkpoint verification failed in verification.json")

    splits = json.loads((folder / "split_ids.json").read_text(encoding="utf-8"))
    if len(splits["train"]) != 20997 or len(splits["validation"]) != 4500 or len(splits["test"]) != 4500:
        raise ValueError("Split counts mismatch")
    if set(splits["train"]) & set(splits["validation"]) or set(splits["train"]) & set(splits["test"]):
        raise ValueError("Partition row ID collision")

    predictions = pd.read_csv(folder / "predictions.csv")
    if len(predictions) != 4500 * 3 or predictions.duplicated(["condition", "row_id"]).any():
        raise ValueError("Prediction table row count or uniqueness violation")

    summaries = pd.read_csv(folder / "metrics.csv")
    expected_seed = int(folder.name.split("-")[-1])
    if len(summaries) != 3 or set(summaries.seed) != {expected_seed} or set(summaries.condition) != {"Clean", "FGSM", "PGD"}:
        raise ValueError("Metrics summary table invalid")

    class_rows = pd.read_csv(folder / "per_class.csv")
    if len(class_rows) != 81:
        raise ValueError(f"Expected 81 per-class rows, found {len(class_rows)}")

    for name in ["Clean", "FGSM", "PGD"]:
        grp = predictions[predictions.condition == name]
        if grp.row_id.tolist() != splits["test"] or set(grp.checkpoint_sha256) != {manifest["checkpoint.pt"]}:
            raise ValueError(f"{name}: test row IDs or checkpoint hash mismatch in predictions")
        with np.load(folder / "evaluations" / f"{name}.npz") as saved:
            np.testing.assert_array_equal(saved["row_ids"], grp.row_id)
            np.testing.assert_array_equal(saved["labels"], grp.label)
            np.testing.assert_array_equal(saved["logits"].argmax(axis=1), grp.prediction)
            assert np.isfinite(saved["logits"]).all() and saved["linf"].max() <= 0.10001
        row = summaries[summaries.condition == name].iloc[0]
        recomputed = [
            accuracy_score(grp.label, grp.prediction),
            f1_score(grp.label, grp.prediction, labels=np.arange(27), average="macro", zero_division=0),
            f1_score(grp.label, grp.prediction, average="weighted", zero_division=0),
            matthews_corrcoef(grp.label, grp.prediction),
        ]
        np.testing.assert_allclose(row[["accuracy", "macro_f1", "weighted_f1", "mcc"]].to_numpy(dtype=float), recomputed, atol=1e-12)
        with np.load(folder / "confusion_matrices.npz") as saved:
            np.testing.assert_array_equal(saved[name], confusion_matrix(grp.label, grp.prediction, labels=np.arange(27)))
        assert class_rows[class_rows.condition == name].support.sum() == 4500

    return summaries


def run_seed(infra, cfg: dict, data_bundle: dict, seed: int, root: Path, resume_allowed: bool) -> None:
    """Train and evaluate one full seed."""
    folder = root / f"seed-{seed}"
    if (folder / "manifest.json").exists():
        verify_seed(folder, cfg)
        print(f"REUSED verified completed seed {seed}", flush=True)
        return

    if folder.exists() and not resume_allowed:
        raise ValueError(f"Existing incomplete seed {seed}; use --resume to continue")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "epochs").mkdir(exist_ok=True)

    started = time.perf_counter()
    with (folder / "raw.log").open("a", encoding="utf-8") as stream, contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)), MemoryMonitor() as memory:
        print(f"START seed {seed}, pid={os.getpid()}, 25 epochs, {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Build synthetic sequences
        tick_data = time.perf_counter()
        seq_len = cfg["sequence"]["length"]  # 50
        noise_std = cfg["sequence"]["train_noise_std"]  # 0.01

        # Training sequences: tile and add Gaussian noise with training seed RNG
        rng = np.random.default_rng(seed)
        X_tr_norm = data_bundle["X_tr_norm"]
        X_tr_seq = np.repeat(X_tr_norm[:, None, :], seq_len, axis=1) + rng.normal(0, noise_std, size=(len(X_tr_norm), seq_len, 39)).astype(np.float32)
        y_tr = data_bundle["y_tr"]

        # Test evaluation sequences: tile without noise
        X_te_norm = data_bundle["X_te_norm"]
        X_te_seq = np.repeat(X_te_norm[:, None, :], seq_len, axis=1).astype(np.float32)
        y_te = data_bundle["y_te"]
        test_ids = data_bundle["test_idx"]
        data_seconds = time.perf_counter() - tick_data

        # Instantiate model
        model = infra.StandardTransformerIDS(**cfg["architecture"]).to("cpu")
        assert sum(p.numel() for p in model.parameters()) == 351612

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
            betas=tuple(cfg["training"]["betas"]),
            eps=cfg["training"]["eps"],
        )

        epoch_files = sorted((folder / "epochs").glob("epoch-*.pt"))
        resume = torch.load(epoch_files[-1], map_location="cpu", weights_only=True) if epoch_files else None
        start_epoch = 0
        previous_seconds = 0.0
        if resume:
            if resume["configuration_hash"] != config_hash(cfg) or resume["seed"] != seed:
                raise ValueError("Checkpoint configuration mismatch")
            model.load_state_dict(resume["state_dict"], strict=True)
            opt.load_state_dict(resume["optimizer"])
            torch.set_rng_state(resume["torch_rng"])
            start_epoch = resume["epoch"]
            previous_seconds = resume["training_seconds"]
            print(f"RESUME seed {seed} from completed epoch {start_epoch}; cumulative train={previous_seconds:.2f}s", flush=True)

        training_tick = time.perf_counter()
        epoch_tick = training_tick
        total_epochs = cfg["training"]["epochs"]
        bs = cfg["training"]["batch_size"]
        Xt = torch.tensor(X_tr_seq, dtype=torch.float32)
        yt = torch.tensor(y_tr, dtype=torch.long)

        for ep in range(start_epoch, total_epochs):
            model.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), bs):
                idx = perm[i : i + bs]
                xb, yb = Xt[idx], yt[idx]
                opt.zero_grad()
                out = model(xb)["logits"]
                loss = F.cross_entropy(out, yb)
                loss.backward()
                opt.step()

            now = time.perf_counter()
            elapsed = previous_seconds + now - training_tick
            epoch_num = ep + 1
            atomic_save(
                folder / "epochs" / f"epoch-{epoch_num:02d}.pt",
                {
                    "state_dict": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "epoch": epoch_num,
                    "seed": seed,
                    "training_seconds": elapsed,
                    "configuration_hash": config_hash(cfg),
                },
            )
            atomic_json(
                folder / "status.json",
                {
                    "state": "training",
                    "seed": seed,
                    "epoch_completed": epoch_num,
                    "epochs_required": total_epochs,
                    "train_rows": len(X_tr_seq),
                    "training_seconds": elapsed,
                    "last_epoch_seconds": now - epoch_tick,
                    "rss_bytes": psutil.Process().memory_info().rss,
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            print(
                f"EPOCH {epoch_num}/{total_epochs} seed={seed}: {now - epoch_tick:.2f}s; cumulative train={elapsed:.2f}s; RSS={psutil.Process().memory_info().rss / 2**30:.2f} GiB",
                flush=True,
            )
            epoch_tick = time.perf_counter()

        training_seconds = previous_seconds + time.perf_counter() - training_tick
        final_epoch_state = torch.load(folder / f"epochs/epoch-{total_epochs:02d}.pt", map_location="cpu", weights_only=True)
        if final_epoch_state["epoch"] != total_epochs:
            raise ValueError("Final epoch checkpoint missing")

        # Save final epoch 25 as checkpoint.pt (declared before execution)
        if not (folder / "checkpoint.pt").exists():
            atomic_save(
                folder / "checkpoint.pt",
                {
                    "state_dict": model.state_dict(),
                    "epoch": total_epochs,
                    "architecture": cfg["architecture"],
                    "configuration_hash": config_hash(cfg),
                    "claim": CLAIM,
                    "seed": seed,
                },
            )

        # Checkpoint reload test
        eval_bs = cfg["resources"].get("evaluation_batch_size", 64)
        model.eval()
        before_logits = compute_logits(model, X_te_seq[:128], batch_size=eval_bs)
        saved_chk = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved_chk["state_dict"], strict=True)
        model.eval()
        after_logits = compute_logits(model, X_te_seq[:128], batch_size=eval_bs)
        np.testing.assert_array_equal(before_logits, after_logits)

        atomic_json(
            folder / "training_complete.json",
            {
                "epochs": total_epochs,
                "train_rows": len(X_tr_seq),
                "samples_processed": len(X_tr_seq) * total_epochs,
                "optimizer_steps": cfg["training"]["optimizer_steps_total"],
                "batch_size": bs,
                "checkpoint_rule": "final epoch 25, no validation/evaluation-based selection",
                "configuration_hash": config_hash(cfg),
                "training_seconds": training_seconds,
            },
        )

        chk_sha = sha(folder / "checkpoint.pt")
        metrics = evaluate_seed(infra, cfg, model, X_te_seq, y_te, test_ids, folder, chk_sha, seed)

        verif = json.loads((folder / "verification.json").read_text(encoding="utf-8"))
        verif["checkpoint_reload_logits_identical"] = True
        atomic_json(folder / "verification.json", verif)

        eval_runtimes = {n: json.loads((folder / "evaluations" / f"{n}.json").read_text(encoding="utf-8"))["seconds"] for n in ["Clean", "FGSM", "PGD"]}
        atomic_json(
            folder / "runtime.json",
            {
                "data_seconds_this_invocation": data_seconds,
                "training_seconds_accumulated": training_seconds,
                "evaluation_seconds": eval_runtimes,
                "invocation_seconds_after_imports": time.perf_counter() - started,
                "peak_rss_sampled_bytes_this_invocation": memory.peak,
                "minimum_system_available_bytes_this_invocation": memory.min_available,
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        print(f"SEED {seed} EVALUATION METRICS:", json.dumps(metrics), flush=True)

    # Ensure shared split and preprocessing artifacts are present in seed folder for self-contained audit
    for shared_name in ["split_ids.json", "split_overlap_audit.json", "preprocessing.npz", "preprocessing.json"]:
        src = root / shared_name
        dst = folder / shared_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # Build manifest
    manifest = {
        p.relative_to(folder).as_posix(): sha(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name not in {"status.json", "manifest.json"} and not p.name.endswith(".tmp") and "epochs" not in p.parts
    }
    atomic_json(folder / "manifest.json", manifest)
    verify_seed(folder, cfg)
    atomic_json(folder / "status.json", {"state": "complete", "seed": seed, "full_budget_and_outputs_verified": True})
    print(f"COMPLETED AND VERIFIED SEED {seed}", flush=True)


def run_smoke(root: Path, cfg: dict):
    """Run a small smoke run on training/validation data in its own directory."""
    print("=" * 60)
    print("RUNNING SMOKE TEST ON TRAINING/VALIDATION DATA")
    print("=" * 60)
    smoke_dir = root / "smoke_checks"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    upstream = ROOT / "upstream/PAVE_artifact"
    csv_path = upstream / "data/balanced_CICIoT.csv"
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    labels = le.fit_transform(df["Label"])
    numeric = df.drop(columns="Label").apply(pd.to_numeric, errors="coerce").fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # Select small balanced smoke subset: 10 per class train (270 rows), 5 per class val (135 rows)
    train_smoke_idx, val_smoke_idx = [], []
    for c in range(27):
        c_idx = np.where(labels == c)[0]
        train_smoke_idx.extend(c_idx[:10])
        val_smoke_idx.extend(c_idx[10:15])
    train_smoke_idx = np.array(train_smoke_idx)
    val_smoke_idx = np.array(val_smoke_idx)

    # Check shapes and alignment
    assert len(train_smoke_idx) == 270 and len(val_smoke_idx) == 135
    assert not (set(train_smoke_idx) & set(val_smoke_idx))

    # Preprocessing on train only
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(numeric.iloc[train_smoke_idx].values).astype(np.float32)
    X_va = scaler.transform(numeric.iloc[val_smoke_idx].values).astype(np.float32)
    assert X_tr.min() >= -1e-6 and X_tr.max() <= 1.0 + 1e-6

    # Model and single epoch training
    infra = import_reference(smoke_dir, cfg)
    model = infra.StandardTransformerIDS(**cfg["architecture"]).to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    rng = np.random.default_rng(1)
    X_tr_seq = np.repeat(X_tr[:, None, :], 50, axis=1) + rng.normal(0, 0.01, size=(len(X_tr), 50, 39)).astype(np.float32)
    X_va_seq = np.repeat(X_va[:, None, :], 50, axis=1).astype(np.float32)
    y_tr = labels[train_smoke_idx]
    y_va = labels[val_smoke_idx]

    model.train()
    xb = torch.tensor(X_tr_seq, dtype=torch.float32)
    yb = torch.tensor(y_tr, dtype=torch.long)
    opt.zero_grad()
    loss = F.cross_entropy(model(xb)["logits"], yb)
    loss.backward()
    opt.step()
    print("Smoke 1-step optimization: PASSED, loss =", float(loss.item()))

    # Checkpoint save / reload
    chk_path = smoke_dir / "smoke_chk.pt"
    atomic_save(chk_path, {"state_dict": model.state_dict()})
    before_eval = state_hash(model)
    loaded = torch.load(chk_path, map_location="cpu", weights_only=True)
    model.load_state_dict(loaded["state_dict"])
    assert state_hash(model) == before_eval
    print("Smoke checkpoint save/reload: PASSED")

    # Evaluation checks
    model.eval()
    assert not model.training

    # Epsilon zero identity
    fg0 = fgsm_attack(model, X_va_seq[:10], y_va[:10], eps=0.0, batch_size=10)
    pg0 = pgd_attack(model, X_va_seq[:10], y_va[:10], seed=1, eps=0.0, alpha=0.025, iters=3, batch_size=10)
    np.testing.assert_array_equal(fg0, X_va_seq[:10])
    np.testing.assert_array_equal(pg0, X_va_seq[:10])
    print("Smoke epsilon-zero identity: PASSED")

    # Perturbation bounds & domain clipping
    fg = fgsm_attack(model, X_va_seq[:10], y_va[:10], eps=0.10, batch_size=10)
    pg = pgd_attack(model, X_va_seq[:10], y_va[:10], seed=1, eps=0.10, alpha=0.025, iters=3, batch_size=10)
    assert np.max(np.abs(fg - X_va_seq[:10])) <= 0.10001
    assert np.max(np.abs(pg - X_va_seq[:10])) <= 0.10001
    assert fg.min() >= -1e-6 and fg.max() <= 1.0 + 1e-6
    assert pg.min() >= -1e-6 and pg.max() <= 1.0 + 1e-6
    print("Smoke perturbation bounds & domain clipping: PASSED")

    # Weights unchanged
    assert state_hash(model) == before_eval
    print("Smoke model weights unchanged during attacks: PASSED")

    atomic_json(smoke_dir / "smoke_checks_passed.json", {"status": "all smoke checks passed", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("ALL SMOKE CHECKS PASSED SUCCESSFULLY!\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = read_config(args.config)
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        run_smoke(root, cfg)
        return

    seeds = args.seeds or cfg["seeds"]
    if len(set(seeds)) != len(seeds) or any(s not in cfg["seeds"] for s in seeds):
        parser.error("Use only the frozen predetermined replication seeds")

    frozen_config = root / "configuration.json"
    if frozen_config.exists():
        existing_cfg = json.loads(frozen_config.read_text(encoding="utf-8"))
        if existing_cfg != cfg:
            parser.error("Existing run configuration differs")
    else:
        atomic_json(frozen_config, cfg)
        atomic_json(
            root / "environment.json",
            {
                "python": sys.version,
                "executable": sys.executable,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
                "platform": platform.platform(),
                "cpu": platform.processor(),
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(),
                "ram_bytes": psutil.virtual_memory().total,
                "cuda_available": torch.cuda.is_available(),
                "git_revision": git("rev-parse", "HEAD"),
                "git_status": git("status", "--porcelain"),
                "executor_sha256": sha(Path(__file__)),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        shutil.copy2(__file__, root / "executor_snapshot.py")

    if args.verify_only:
        for seed in seeds:
            verify_seed(root / f"seed-{seed}", cfg)
        print("All requested seed bundles verified successfully")
        return

    lock = root / "run.lock"
    if lock.exists():
        old = json.loads(lock.read_text(encoding="utf-8"))
        if psutil.pid_exists(old["pid"]) and abs(psutil.Process(old["pid"]).create_time() - old["create_time"]) < 1:
            parser.error(f"Run already active as PID {old['pid']}")
        lock.rename(root / f"stale-lock-{time.time_ns()}.json")

    with lock.open("x") as stream:
        json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time()}, stream)

    try:
        torch.set_num_threads(cfg["resources"]["torch_threads"])
        torch.use_deterministic_algorithms(True)
        infra = import_reference(root, cfg)
        data_bundle = prepare_data(infra, cfg, root)

        for seed in seeds:
            run_seed(infra, cfg, data_bundle, seed, root, args.resume)
            verify_seed(root / f"seed-{seed}", cfg)

        atomic_json(root / "requested_run_complete.json", {"seeds_completed_this_request": seeds, "claim": CLAIM})
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
