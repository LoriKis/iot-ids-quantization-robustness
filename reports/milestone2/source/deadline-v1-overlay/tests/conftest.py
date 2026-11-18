"""Shared fixtures: temp OUT, synthetic split + CSV, synthetic timing measurements. No torch, no research.*."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

OVERLAY = Path(__file__).resolve().parents[1]
if str(OVERLAY) not in sys.path:
    sys.path.insert(0, str(OVERLAY))

V4_ROOT_DEFAULT = OVERLAY.parent / "IOT-network-traffic-DL-M2-VM-v4"
CONFIG_PATH = OVERLAY / "configs" / "m2_deadline_v1.json"
N_CLASSES = 27


def v4_root():
    """Local verified V4 package (read as text/JSON only) or None."""
    root = Path(os.environ.get("M2_DEADLINE_V4_ROOT", str(V4_ROOT_DEFAULT)))
    return root if (root / "MANIFEST.sha256").is_file() else None


@pytest.fixture
def v4():
    root = v4_root()
    if root is None:
        pytest.skip("V4 package not reachable on this machine (V4-dependent check skipped)")
    return root


@pytest.fixture
def config():
    from m2_deadline.common import load_config

    return load_config(CONFIG_PATH)


@pytest.fixture
def out(tmp_path):
    """Temp OUT folder (separate from the overlay tree)."""
    path = tmp_path / "artifacts" / "m2-deadline-v1"
    path.mkdir(parents=True)
    return path


def canonical_ids_hash(ids):
    """Independent re-computation of the canonical-JSON id hash used by research.common.digest."""
    payload = json.dumps([int(i) for i in sorted(ids)], sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_split_and_csv(tmp_path, test_counts=None, train_per_class=6, val_per_class=3, seed=7):
    """Synthetic split document + CSV with SHUFFLED rows and string ``Label`` values.

    ``test_counts``: list of 27 ints (rows per class in the test partition); default 15 for every class.
    Returns ``(split_dict, csv_path)``. Row ids are positional indices into the shuffled CSV.
    """
    test_counts = list(test_counts) if test_counts is not None else [15] * N_CLASSES
    names = ["class_%02d" % i for i in range(N_CLASSES)]
    rows = []  # (class, partition)
    for c in range(N_CLASSES):
        rows += [(c, "train")] * train_per_class + [(c, "validation")] * val_per_class + [(c, "test")] * test_counts[c]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    feats = rng.normal(size=(len(rows), 2))

    csv_path = Path(tmp_path) / "dataset.csv"
    with open(str(csv_path), "w", encoding="utf-8", newline="\n") as f:
        f.write("f0,f1,Label\n")
        for (c, _), (a, b) in zip(rows, feats):
            f.write("%.6f,%.6f,%s\n" % (a, b, names[c]))
    csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    parts = {"train": [], "validation": [], "test": []}
    for pos, (_, part) in enumerate(rows):
        parts[part].append(pos)
    labels = np.array([c for c, _ in rows])
    per_class = {
        p: np.bincount(labels[parts[p]], minlength=N_CLASSES).tolist() for p in parts
    }
    split = {
        "schema_version": 1,
        "dataset": {
            "path": "dataset.csv",
            "sha256": csv_sha,
            "rows": len(rows),
            "features": 2,
            "classes": N_CLASSES,
            "feature_names": ["f0", "f1"],
            "class_names": names,
        },
        "split_seed": 20270,
        "per_class_counts": per_class,
        "id_sha256": {p: canonical_ids_hash(parts[p]) for p in parts},
        "row_ids": {p: list(parts[p]) for p in parts},
    }
    return split, csv_path


@pytest.fixture
def synthetic_split(tmp_path):
    return make_split_and_csv(tmp_path)


def make_measurements(
    step_ord=0.05,
    step_adv=0.4,
    p90_factor=1.1,
    checkpoint_write_s=0.5,
    ptq_s=5.0,
    load_arrays_s=4.0,
    val_fp32=6.0,
    val_int8=9.0,
    clean=0.01,
    fgsm=0.02,
    pgd5=0.1,
    efficiency_s=300.0,
    n_train=22412,
    n_val=3821,
    n_test=3764,
):
    """Synthetic D5.3 measurements (+ ``data``). All numbers are made up; no model is involved."""
    attack = {"clean": clean, "fgsm": fgsm, "pgd5": pgd5}
    return {
        "data": {"n_train": n_train, "n_val": n_val, "n_test": n_test},
        "train_step_seconds": {
            "ordinary": {"mean": step_ord, "p90": step_ord * p90_factor, "step_cv": 0.1},
            "adversarial": {"mean": step_adv, "p90": step_adv * p90_factor, "step_cv": 0.1},
        },
        "checkpoint_write_s": checkpoint_write_s,
        "ptq_s": ptq_s,
        "load_arrays_s": load_arrays_s,
        "val_eval_s": {"fp32": val_fp32, "int8": val_int8, "n_rows": n_val},
        "attack_seconds_per_example": {
            "fp32_direct": dict(attack),
            "bpda_int8": {k: v * 2.0 for k, v in attack.items()},
        },
        "efficiency_s": efficiency_s,
    }


@pytest.fixture
def measurements():
    return make_measurements()


@pytest.fixture
def measurements_factory():
    return make_measurements


def fake_efficiency_cfg():
    """Shape of V4 ``configs/m2b/efficiency_v2.json`` fields that the protocol reads."""
    return {
        "fixed_inputs": {"count": 256},
        "batch1": {"warmup": 20, "timed": 200},
        "batch256": {"warmup": 5, "timed": 30},
        "repetitions": {"full": 3, "pilot": 2},
    }


@pytest.fixture
def efficiency_cfg():
    return copy.deepcopy(fake_efficiency_cfg())
