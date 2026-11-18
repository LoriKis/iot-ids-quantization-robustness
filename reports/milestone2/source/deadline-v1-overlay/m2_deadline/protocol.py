"""Evaluation protocol: test labels without features, the G1 attack config, the frozen protocol file.

Pure module: standard library + numpy at import time; pandas is imported inside
:func:`read_test_labels`. Python 3.9 compatible. Never imports torch or research.*.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import sampling
from .common import (
    DataIntegrityError,
    GuardError,
    N_CLASSES,
    PATHS,
    ProtocolChangedError,
    RUN_ID,
    atomic_write_bytes,
    atomic_write_text,
    canonical_json,
    eps_str,
    out_path,
    read_json,
    sha256_bytes,
    sha256_file,
)

SUBSET_RULE = (
    "sampling.stratified_subset v1: sorted IDs per class, seed=sha256(name|split_seed|class)[:8], "
    "min(k,available)"
)
MEMORY_METRIC = (
    "peak_rss_bytes = Linux VmHWM of a fresh child process after the timed runs; "
    "peak_rss_delta_bytes = VmHWM minus RSS right after model load (the baseline includes the loaded weights "
    "and the torch footprint, so the delta is inference-time working memory); "
    "peak_rss_sampled_delta_bytes = 10 ms sampler cross-check; weight memory = state_dict_bytes"
)
INT8_LABEL_TEXT = "APPROXIMATE INT8 GRADIENT (BPDA)"
INT8_DESCRIPTION = (
    "forward pass is the real INT8 model (asserted bit-equal logits); backward pass goes through each of the "
    "23 INT8 Linear layers using their dequantized weights and ignores activation quantization; softmax, "
    "LayerNorm and residuals differentiate normally; candidates are scored by the real INT8 model. Not a "
    "straight-through estimator on fake-quant and not an FP32 shadow. Robustness of INT8 models may be "
    "over-estimated if this gradient is weak."
)
FP32_DESCRIPTION = "white-box gradient of the FP32 model, scored on the same model"
NOT_RUN = [
    "transfer_fp32_to_int8",
    "transfer_fp32_to_int8_targetsel",
    "surrogate_fakequant",
    "surrogate_fakequant_targetsel",
    "pgd20",
    "pgd50x5",
    "direct_int8_grad",
]
PGD5_STATUS = "deadline_v1"


# ---------------------------------------------------------------- test labels (DESIGN 6.2a)
def read_test_labels(split: Any, dataset_cfg: Optional[Dict[str, Any]], dataset_csv: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(test_ids, y_test)`` from the split document and the dataset CSV label column only.

    No features, scaler or model are touched and ``m2_split.load_arrays`` is never called. Every failure
    raises ``DataIntegrityError`` (exit 2). ``split`` is the parsed ``split_m2.json`` or its path.
    """
    if not isinstance(split, dict):
        split = read_json(split)
    try:
        ds = split["dataset"]
        class_names = list(ds["class_names"])
        expected_sha = ds["sha256"]
        expected_rows = int(ds["rows"])
        raw_test_ids = split["row_ids"]["test"]
        expected_ids_hash = split["id_sha256"]["test"]
        expected_counts = [int(v) for v in split["per_class_counts"]["test"]]
    except (KeyError, TypeError) as exc:
        raise DataIntegrityError("split document lacks a required field: %s" % exc)

    csv_path = Path(str(dataset_csv))
    if not csv_path.is_file():
        raise DataIntegrityError("dataset CSV not found: %s" % csv_path)
    actual_sha = sha256_file(csv_path)
    if actual_sha != expected_sha:
        raise DataIntegrityError("dataset sha256 mismatch: %s != %s" % (actual_sha, expected_sha))

    import pandas as pd  # lazy: keeps the module importable without pandas

    column = (dataset_cfg or {}).get("label_column", "Label")
    df = pd.read_csv(str(csv_path))
    if len(df) != expected_rows:
        raise DataIntegrityError("dataset has %d rows, split expects %d" % (len(df), expected_rows))
    if column not in df.columns:
        raise DataIntegrityError("label column %r not in dataset" % column)

    if len(class_names) != N_CLASSES or len(set(class_names)) != N_CLASSES:
        raise DataIntegrityError("split class_names must hold %d unique names" % N_CLASSES)
    mapping = {name: i for i, name in enumerate(class_names)}
    mapped = df[column].map(mapping)
    if mapped.isna().any():
        unknown = sorted(set(str(v) for v in df[column][mapped.isna()].unique()))
        raise DataIntegrityError("unknown labels in dataset: %s" % unknown[:5])
    y_all = mapped.to_numpy(dtype=np.int64)

    test_ids = np.asarray(raw_test_ids, dtype=np.int64)
    if len(np.unique(test_ids)) != len(test_ids):
        raise DataIntegrityError("duplicate test row ids in split")
    if len(test_ids) == 0 or test_ids.min() < 0 or test_ids.max() >= len(df):
        raise DataIntegrityError("test row ids outside 0..%d" % (len(df) - 1))
    y_test = y_all[test_ids]

    actual_hash = sampling.ids_sha256(test_ids)
    if actual_hash != expected_ids_hash:
        raise DataIntegrityError("test id hash mismatch: %s != %s" % (actual_hash, expected_ids_hash))

    counts = np.bincount(y_test, minlength=N_CLASSES).tolist()
    if counts != expected_counts:
        raise DataIntegrityError("test per-class counts differ from the split document")
    return test_ids, y_test


# ---------------------------------------------------------------- G1: pgd5 attack config
def selector_calls(kind: str, steps: int, restarts: int) -> Dict[str, int]:
    """Re-derivation of ``m2_attacks.expected_selector_calls`` (checked against V4 source in tests)."""
    nr, ns = (1, 1) if kind == "fgsm" else (int(restarts), int(steps))
    candidates = 1 + nr * (1 + ns)
    return {"selector_calls": 1 + candidates, "candidates": candidates}


def build_pgd5_cfg(attacks_cfg: Dict[str, Any], pgd5: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deep copy of a V4 attacks config with ``pgd5`` added (the input is not modified)."""
    spec = dict(pgd5) if pgd5 is not None else {
        "kind": "pgd", "steps": 5, "alpha_fraction": 0.25, "restarts": 1, "random_start": True,
    }
    cfg = copy.deepcopy(attacks_cfg)
    cfg["attacks"]["pgd5"] = {
        "kind": spec["kind"],
        "steps": int(spec["steps"]),
        "alpha_fraction": float(spec["alpha_fraction"]),
        "restarts": int(spec["restarts"]),
        "random_start": bool(spec["random_start"]),
        "status": PGD5_STATUS,
    }
    cfg["expected_selector_calls_per_batch"]["pgd5"] = selector_calls(spec["kind"], spec["steps"], spec["restarts"])
    return cfg


# ---------------------------------------------------------------- subset files and the protocol
def ids_csv_bytes(ids: Any, labels: Any) -> bytes:
    lines = ["example_id,y_true"] + ["%d,%d" % (int(i), int(y)) for i, y in zip(ids, labels)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_ids_csv(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    rows = data.decode("utf-8").splitlines()
    if not rows or rows[0] != "example_id,y_true":
        raise DataIntegrityError("subset ids CSV has an unexpected header")
    pairs = [r.split(",") for r in rows[1:] if r]
    return (np.array([int(p[0]) for p in pairs], dtype=np.int64), np.array([int(p[1]) for p in pairs], dtype=np.int64))


def build_protocol(
    *,
    config: Dict[str, Any],
    config_sha256: str,
    lock: Dict[str, Any],
    lock_sha256: str,
    split: Dict[str, Any],
    split_m2_sha256: str,
    scaler_sha256: str,
    efficiency_cfg: Dict[str, Any],
    subset: Dict[str, Any],
    subset_ids_file_sha256: str,
    counts_sha256: str,
) -> Dict[str, Any]:
    """Protocol dictionary. Contains no timestamp, so it is a pure function of its inputs."""
    rob = config["robustness"]
    pgd = rob["attacks"]["pgd5"]
    nonzero = [e for e in rob["epsilons"] if float(e) != 0.0]
    alpha_at_eps = {eps_str(e): float(pgd["alpha_fraction"]) * float(e) for e in nonzero}
    eff = efficiency_cfg
    return {
        "schema": 1,
        "kind": "m2_deadline_protocol",
        "run_id": config["run_id"],
        "frozen_before": "training start and any test-set model inference",
        "config_sha256": config_sha256,
        "lock_sha256": lock_sha256,
        "epochs": int(lock["epochs"]),
        "seed": int(config["seed"]),
        "checkpoint_rule": config["training"]["checkpoint_rule"],
        "test_partition_id_sha256": split["id_sha256"]["test"],
        "split": {
            "split_m2_sha256": split_m2_sha256,
            "scaler_sha256": scaler_sha256,
            "n_train": len(split["row_ids"]["train"]),
            "n_val": len(split["row_ids"]["validation"]),
            "n_test": len(split["row_ids"]["test"]),
        },
        "full_test": {
            "metrics": ["accuracy", "macro_f1"],
            "macro_f1_labels": "labels=range(27), zero_division=0 (same for full test and subset)",
            "batch_size": int(rob["batch_size"]),
            "batch_size_reason": "equal to the attack batch size so eps-0 subset predictions are exactly comparable",
        },
        "robustness": {
            "per_class_target": int(lock["per_class_target"]),
            "n_actual": int(subset["n_actual"]),
            "subset_ids_file": Path(PATHS["subset_ids"]).name,
            "subset_ids_file_sha256": subset_ids_file_sha256,
            "subset_ids_sha256": subset["ids_sha256"],
            "subset_label_sha256": sampling.label_sha256(subset["labels"]),
            "counts_file": Path(PATHS["subset_counts"]).name,
            "counts_sha256": counts_sha256,
            "subset_rule": SUBSET_RULE,
            "epsilons": [float(e) for e in rob["epsilons"]],
            "eps0_computed_once_shared_by_both_curves": True,
            "attacks": {
                "fgsm": {"steps": 1, "restarts": 1, "random_start": False},
                "pgd5": {
                    "steps": int(pgd["steps"]),
                    "restarts": int(pgd["restarts"]),
                    "random_start": bool(pgd["random_start"]),
                    "alpha_fraction": float(pgd["alpha_fraction"]),
                    "alpha_at_eps": alpha_at_eps,
                },
            },
            "batch_size": int(rob["batch_size"]),
            "seed_rule": "derived_seed('atk-eval', 1, model_id, source, epsilon, batch_index)",
            "methods": {
                "fp32": {"source": rob["methods"]["fp32"], "description": FP32_DESCRIPTION},
                "int8": {
                    "source": rob["methods"]["int8"],
                    "label": INT8_LABEL_TEXT,
                    "description": INT8_DESCRIPTION,
                },
            },
            "not_run": list(NOT_RUN),
        },
        "efficiency": {
            "threads": int(config["efficiency"]["threads"]),
            "rows": int(eff["fixed_inputs"]["count"]),
            "rows_partition": config["efficiency"]["rows_partition"],
            "batch1": {"warmup": int(eff["batch1"]["warmup"]), "timed": int(eff["batch1"]["timed"])},
            "batch256": {"warmup": int(eff["batch256"]["warmup"]), "timed": int(eff["batch256"]["timed"])},
            "repetitions": int(eff["repetitions"]["full"]),
            "interleaved": "V1,V2,V3,V4 x repetitions, fresh child process each",
            "memory_metric": MEMORY_METRIC,
        },
        "eps0_consistency": {
            "behaviour": "record_only",
            "never_fails_run": True,
            "fields": ["eps0_vs_full_test_mismatches", "eps0_max_abs_logit_diff"],
            "curves_use": "attack-path eps-0 predictions",
        },
        "frozen_not_chosen_by_performance": True,
    }


def build_freeze(
    *,
    config: Dict[str, Any],
    config_sha256: str,
    lock: Dict[str, Any],
    lock_sha256: str,
    split: Dict[str, Any],
    split_m2_sha256: str,
    scaler_sha256: str,
    efficiency_cfg: Dict[str, Any],
    test_ids: Any,
    y_test: Any,
) -> Dict[str, Any]:
    """Subset, counts file and protocol, all as bytes in memory. Writes nothing."""
    k = int(lock["per_class_target"])
    split_seed = int(split["split_seed"])
    seed_name = config["robustness"]["subset_seed_name"]
    subset = sampling.stratified_subset(test_ids, y_test, k, split_seed, N_CLASSES, seed_name)
    counts_doc = sampling.counts_document(subset, k, split_seed, seed_name)
    counts_bytes = canonical_json(counts_doc).encode("utf-8")
    csv_bytes = ids_csv_bytes(subset["ids"], subset["labels"])
    protocol = build_protocol(
        config=config,
        config_sha256=config_sha256,
        lock=lock,
        lock_sha256=lock_sha256,
        split=split,
        split_m2_sha256=split_m2_sha256,
        scaler_sha256=scaler_sha256,
        efficiency_cfg=efficiency_cfg,
        subset=subset,
        subset_ids_file_sha256=sha256_bytes(csv_bytes),
        counts_sha256=sha256_bytes(counts_bytes),
    )
    protocol_bytes = canonical_json(protocol).encode("utf-8")
    return {
        "protocol": protocol,
        "protocol_bytes": protocol_bytes,
        "protocol_sha256": sha256_bytes(protocol_bytes),
        "subset": subset,
        "counts_doc": counts_doc,
        "counts_bytes": counts_bytes,
        "ids_csv_bytes": csv_bytes,
    }


def write_protocol(out: Any, freeze: Dict[str, Any], frozen_utc: str) -> Dict[str, Any]:
    """Write protocol files. Idempotent: identical existing files are only validated.

    Differing existing content raises ``GuardError`` before anything is written or changed.
    Returns ``{"written": bool, "protocol_sha256": str}``.
    """
    targets = [
        ("subset_ids", freeze["ids_csv_bytes"]),
        ("subset_counts", freeze["counts_bytes"]),
        ("protocol", freeze["protocol_bytes"]),
    ]
    sha_line = ("%s  %s\n" % (freeze["protocol_sha256"], Path(PATHS["protocol"]).name)).encode("utf-8")
    targets.append(("protocol_sha256", sha_line))
    pending = []
    for key, data in targets:
        path = out_path(out, key)
        if path.exists():
            if path.read_bytes() != data:
                raise GuardError("existing %s differs from the content this run would freeze; "
                                 "leaving it untouched" % path)
        else:
            pending.append((path, data))
    sidecar = out_path(out, "protocol_frozen")
    if pending or not sidecar.exists():
        for path, data in pending:
            atomic_write_bytes(path, data)
        if not sidecar.exists():
            atomic_write_text(
                sidecar,
                canonical_json({"frozen_utc": frozen_utc, "protocol_sha256": freeze["protocol_sha256"]}),
            )
    return {"written": bool(pending), "protocol_sha256": freeze["protocol_sha256"]}


def _first_token(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    return text.split()[0] if text else ""


def assert_frozen(out: Any) -> str:
    """Re-verify the frozen protocol; return its sha256. Raises ``ProtocolChangedError`` (exit 3).

    Recomputes the protocol file hash and compares it with ``protocol.json.sha256``,
    ``run_identity.json`` and every stage marker that records a protocol hash; re-checks the subset ids and
    counts files against the hashes stored inside the protocol.
    """
    proto_path = out_path(out, "protocol")
    sha_path = out_path(out, "protocol_sha256")
    for p in (proto_path, sha_path):
        if not p.is_file():
            raise ProtocolChangedError("frozen protocol file missing: %s" % p)
    actual = sha256_file(proto_path)
    recorded = _first_token(sha_path)
    if actual != recorded:
        raise ProtocolChangedError("protocol.json hash %s differs from %s (%s)" % (actual, sha_path.name, recorded))

    identity = out_path(out, "run_identity")
    if identity.is_file():
        stored = read_json(identity).get("protocol_sha256")
        if stored is not None and stored != actual:
            raise ProtocolChangedError("protocol.json hash differs from run_identity.json")
    markers = Path(str(out)) / PATHS["markers_dir"]
    if markers.is_dir():
        for marker in sorted(markers.glob("*.ok")):
            try:
                value = json.loads(marker.read_text(encoding="utf-8")).get("protocol_sha256")
            except ValueError:
                raise ProtocolChangedError("unreadable marker %s" % marker.name)
            if value is not None and value != actual:
                raise ProtocolChangedError("protocol.json hash differs from marker %s" % marker.name)

    try:
        proto = json.loads(proto_path.read_text(encoding="utf-8"))
        rob = proto["robustness"]
    except (ValueError, KeyError) as exc:
        raise ProtocolChangedError("protocol.json is not readable: %s" % exc)
    ids_path = out_path(out, "subset_ids")
    counts_path = out_path(out, "subset_counts")
    for p in (ids_path, counts_path):
        if not p.is_file():
            raise ProtocolChangedError("frozen file missing: %s" % p)
    if sha256_file(ids_path) != rob["subset_ids_file_sha256"]:
        raise ProtocolChangedError("%s changed after freeze" % ids_path.name)
    if sha256_file(counts_path) != rob["counts_sha256"]:
        raise ProtocolChangedError("%s changed after freeze" % counts_path.name)
    ids, labels = parse_ids_csv(ids_path.read_bytes())
    if sampling.ids_sha256(ids) != rob["subset_ids_sha256"] or sampling.label_sha256(labels) != rob["subset_label_sha256"]:
        raise ProtocolChangedError("%s content differs from the frozen subset hashes" % ids_path.name)
    return actual
