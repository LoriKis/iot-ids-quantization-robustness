"""Shared helpers and the on-disk contract of the M2 deadline overlay.

Pure module: standard library only (no numpy, torch or research.*). Python 3.9 compatible.

The tables in this file (``PATHS``, ``JSON_REQUIRED_KEYS``, ``CSV_COLUMNS``) are the single
description of the output layout under ``OUT``. ``stages.py`` writes these files,
``report.py`` reads them and the tests build synthetic OUT folders from the same tables.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

RUN_ID = "m2-deadline-v1"
KIND = "deadline-real"

# ---------------------------------------------------------------- exit codes
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_GUARD = 3
EXIT_INFEASIBLE = 4
EXIT_CONTENTION = 5
EXIT_INTERRUPTED = 130


class DeadlineError(Exception):
    """Base class. ``exit_code`` is what a CLI returns for this error."""

    exit_code = EXIT_ERROR


class ConfigError(DeadlineError):
    exit_code = EXIT_USAGE


class DataIntegrityError(DeadlineError):
    exit_code = EXIT_USAGE


class GuardError(DeadlineError):
    exit_code = EXIT_GUARD


class ProtocolChangedError(GuardError):
    exit_code = EXIT_GUARD


class InfeasibleError(DeadlineError):
    exit_code = EXIT_INFEASIBLE


class ContentionError(DeadlineError):
    exit_code = EXIT_CONTENTION


def exit_code_for(exc: BaseException) -> int:
    if isinstance(exc, DeadlineError):
        return exc.exit_code
    if isinstance(exc, KeyboardInterrupt):
        return EXIT_INTERRUPTED
    return EXIT_ERROR


# ---------------------------------------------------------------- hashing, time, JSON
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Any) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    """Filesystem-safe UTC stamp for log names and archive folders."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def canonical_json(obj: Any) -> str:
    """File form: sorted keys, 2-space indent, LF, trailing newline, no NaN."""
    return json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n"


def canonical_compact(obj: Any) -> str:
    """Hash form, identical to ``research.common.canonical``."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj: Any) -> str:
    """Same value as ``research.common.digest``."""
    return sha256_bytes(canonical_compact(obj).encode())


def atomic_write_bytes(path: Any, data: bytes) -> None:
    path = Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Any, text: str) -> None:
    """Writes UTF-8 with LF line endings regardless of platform."""
    atomic_write_bytes(path, text.replace("\r\n", "\n").encode("utf-8"))


def atomic_write_json(path: Any, obj: Any) -> str:
    """Atomic canonical JSON write. Returns the sha256 of the bytes written."""
    data = canonical_json(obj).encode("utf-8")
    atomic_write_bytes(path, data)
    return sha256_bytes(data)


def read_json(path: Any) -> Any:
    with open(str(path), "r", encoding="utf-8") as stream:
        return json.load(stream)


def setup_logging(out: Any, command: str) -> Tuple[logging.Logger, Path]:
    """Log to ``OUT/logs/<command>-<utc>.log`` and stderr. Returns (logger, log path)."""
    logs = Path(str(out)) / PATHS["logs_dir"]
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / ("%s-%s.log" % (command, utc_stamp()))
    logger = logging.getLogger("m2_deadline." + command)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path


# ---------------------------------------------------------------- model, stage and unit tables
MODEL_IDS = ("V1", "V2", "V3", "V4")
DISPLAY_NAMES = {
    "V1": "FP32-ordinary",
    "V2": "INT8-PTQ-ordinary",
    "V3": "FP32-adversarial",
    "V4": "INT8-PTQ-adversarial",
}
PARENTS = {"V1": None, "V2": "V1", "V3": None, "V4": "V3"}
PRECISION = {"V1": "FP32", "V2": "INT8", "V3": "FP32", "V4": "INT8"}
ADVERSARIAL = {"V1": False, "V2": False, "V3": True, "V4": True}
FP32_MODELS = ("V1", "V3")
INT8_MODELS = ("V2", "V4")
INT8_LABEL = "approx. BPDA gradient"

STAGES = (
    "preflight",
    "freeze",
    "train_ordinary",
    "train_adversarial",
    "ptq",
    "val_check",
    "test_clean",
    "robustness",
    "efficiency",
    "report",
)
# Stages after which protocol.json must exist and be unchanged.
PROTOCOL_STAGES = STAGES[1:]
# component of ``candidates[i].components_s`` that belongs to each stage (None: no own component)
STAGE_COMPONENT = {
    "preflight": None,
    "freeze": None,
    "train_ordinary": "train_ordinary",
    "train_adversarial": "train_adversarial",
    "ptq": "ptq",
    "val_check": "validation",
    "test_clean": "test",
    "robustness": "robustness",
    "efficiency": "efficiency",
    "report": "report",
}
COMPONENT_NAMES = (
    "train_ordinary",
    "train_adversarial",
    "ptq",
    "validation",
    "test",
    "robustness",
    "efficiency",
    "report",
    "overhead",
)

ATTACK_NAMES = ("fgsm", "pgd5")
CLEAN_ATTACK = "clean"
EPSILONS = (0.0, 0.05, 0.1)
N_CLASSES = 27


def eps_str(e: Any) -> str:
    """The single epsilon string form: ``repr(float(e))`` -> '0.0', '0.05', '0.1'."""
    return repr(float(e))


def unit_id(attack: str, epsilon: Any) -> str:
    """Directory/key name of a robustness unit. The shared eps-0 unit is ``clean_eps0.0``."""
    return "%s_eps%s" % (attack, eps_str(epsilon))


CLEAN_UNIT = unit_id(CLEAN_ATTACK, 0.0)
# (attack, epsilon) of the 5 units per model: clean once, then FGSM and PGD at the two non-zero eps.
ATTACK_UNITS = (
    (CLEAN_ATTACK, 0.0),
    ("fgsm", 0.05),
    ("fgsm", 0.1),
    ("pgd5", 0.05),
    ("pgd5", 0.1),
)

# ---------------------------------------------------------------- OUT-relative path table (DESIGN D3)
PATHS: Dict[str, str] = {
    "run_identity": "run_identity.json",
    "markers_dir": "stage_markers",
    "logs_dir": "logs",
    "timing_dir": "timing",
    "timing_estimate": "timing/timing_estimate.json",
    "budget_lock": "timing/budget_lock.json",
    "probe_models_dir": "timing/probe_models",
    "locks_archive_dir": "timing/locks_archive",
    "protocol_dir": "protocol",
    "protocol": "protocol/protocol.json",
    "protocol_sha256": "protocol/protocol.json.sha256",
    "protocol_frozen": "protocol/protocol.frozen.json",
    "subset_ids": "protocol/robustness_subset_ids.csv",
    "subset_counts": "protocol/robustness_subset_counts.json",
    "models_dir": "models",
    "calibration_ids": "calibration/calibration_ids.json",
    "calibration_x": "calibration/calibration_x.pt",
    "val_check": "validation/val_check.json",
    "clean_results": "test/clean_results.json",
    "predictions_full_test": "test/predictions_full_test.csv",
    # written by stage test_clean (the only stage that loads test arrays): ids, x, y of the frozen subset rows
    "subset_rows": "test/robustness_subset_rows.npz",
    "robustness_dir": "robustness",
    "predictions_subset": "robustness/predictions_subset.csv",
    "robustness_results": "robustness/robustness_results.json",
    "efficiency_dir": "efficiency",
    "efficiency": "efficiency/efficiency.json",
    "fixed_rows": "efficiency/fixed_rows.json",
    "fixed_rows_npy": "efficiency/fixed_rows.npy",
    "results_dir": "results",
    "model_comparison_csv": "results/model_comparison.csv",
    "model_comparison_md": "results/model_comparison.md",
    "robustness_curves_csv": "results/robustness_curves.csv",
    "pp_differences": "results/pp_differences.json",
    "report_dir": "report",
    "report_md": "report/m2_deadline_report.md",
    "fgsm_figure": "report/figures/fgsm_curves.png",
    "pgd5_figure": "report/figures/pgd5_curves.png",
}


def out_path(out: Any, key: str) -> Path:
    return Path(str(out)) / PATHS[key]


def marker_path(out: Any, stage: str) -> Path:
    return Path(str(out)) / PATHS["markers_dir"] / (stage + ".ok")


def model_dir(out: Any, model_id: str) -> Path:
    return Path(str(out)) / PATHS["models_dir"] / model_id


def checkpoint_path(out: Any, model_id: str) -> Path:
    """FP32 parents (V1, V3): checkpoint.pt. INT8 copies (V2, V4): state.pt."""
    name = "checkpoint.pt" if model_id in FP32_MODELS else "state.pt"
    return model_dir(out, model_id) / name


def deadline_meta_path(out: Any, model_id: str) -> Path:
    return model_dir(out, model_id) / "deadline_meta.json"


def unit_dir(out: Any, model_id: str, attack: str, epsilon: Any) -> Path:
    return Path(str(out)) / PATHS["robustness_dir"] / model_id / unit_id(attack, epsilon)


# ---------------------------------------------------------------- required keys of result files
JSON_REQUIRED_KEYS: Dict[str, Tuple[str, ...]] = {
    "timing_estimate": (
        "schema", "kind", "run_id", "created_utc", "host", "config_sha256", "v4_manifest_sha256",
        "inputs", "data", "contention", "measurements", "prior_evidence", "candidates", "decision",
        "no_performance_inspected",
    ),
    "budget_lock": (
        "schema", "kind", "run_id", "locked_utc", "epochs", "per_class_target", "planned_hours",
        "cap_hours", "hours_remaining_at_timing", "estimate_sha256", "config_sha256",
        "selection_basis", "performance_inspected",
    ),
    "run_identity": (
        "run_id", "kind", "config_sha256", "lock_sha256", "epochs", "protocol_sha256",
        "v4_manifest_sha256", "source_sha256", "created_utc",
    ),
    "protocol": (
        "schema", "kind", "run_id", "frozen_before", "config_sha256", "lock_sha256", "epochs", "seed",
        "checkpoint_rule", "test_partition_id_sha256", "split", "full_test", "robustness",
        "efficiency", "eps0_consistency", "frozen_not_chosen_by_performance",
    ),
    "protocol_frozen": ("frozen_utc", "protocol_sha256"),
    "subset_counts": (
        "per_class_target", "n_actual", "counts", "classes_present", "classes_missing",
        "subset_ids_sha256", "split_seed", "seed_name",
    ),
    "marker": (
        "stage", "run_id", "config_sha256", "lock_sha256", "protocol_sha256", "epochs",
        "started_utc", "finished_utc", "seconds", "outputs",
    ),
    "deadline_meta": (
        "run_id", "kind", "config_sha256", "model_id", "epochs", "adversarial", "checkpoint_sha256",
        "created_utc",
    ),
    "val_check": ("schema", "kind", "run_id", "n_val", "models", "gates", "informational", "passed"),
    "clean_results": ("schema", "kind", "run_id", "n_test", "labels", "batch_size", "models"),
    "robustness_results": (
        "schema", "kind", "run_id", "n", "per_class_target", "subset_ids_sha256", "epsilons",
        "units", "eps0_consistency",
    ),
    # V4 ``run_efficiency`` record (research/m2_efficiency.py); the report reads ``measurements``
    "efficiency": (
        "any_failed", "interleaved", "repetitions", "threads", "variants", "measurements",
        "summary", "peak_memory_metric",
    ),
    "pp_differences": ("schema", "kind", "run_id", "differences"),
    "unit_done": ("model_id", "attack", "epsilon", "unit_id", "ids_sha256", "valid", "files"),
}
# keys inside one model entry of clean_results["models"]
CLEAN_MODEL_KEYS = ("accuracy", "macro_f1", "n")
# keys inside one unit entry of robustness_results["units"][model_id][unit_id]
UNIT_RESULT_KEYS = ("n", "accuracy", "macro_f1", "robust_accuracy")
# keys inside robustness_results["eps0_consistency"][model_id] (record-only, written by the robustness stage)
EPS0_CONSISTENCY_KEYS = ("eps0_vs_full_test_mismatches", "eps0_max_abs_logit_diff", "n_compared")
# efficiency.json["measurements"] is a list of {"variant": "V1".., "result": {...}} (one per child process).
# Keys of ``result`` that the report reads (V4 ``child_measure`` layout; batch1/batch256 come from ``summarize``).
EFFICIENCY_RESULT_KEYS = (
    "state_dict_bytes", "peak_wset_bytes", "peak_wset_minus_baseline_bytes",
    "sampled_peak_minus_baseline_bytes", "batch1", "batch256",
)
EFFICIENCY_LATENCY_KEYS = ("median_ms", "p95_ms")  # inside result["batch1"] and result["batch256"]
# keys of protocol.json that the report reads (all written by protocol.build_protocol)
REPORT_PROTOCOL_KEYS = (
    "epochs", "seed", "checkpoint_rule", "split", "full_test", "robustness", "efficiency",
)

CSV_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "subset_ids": ("example_id", "y_true"),
    "predictions_full_test": (
        "example_id", "y_true", "in_robust_subset", "V1_pred", "V2_pred", "V3_pred", "V4_pred",
    ),
    "predictions_subset": ("model_id", "attack", "epsilon", "example_id", "y_true", "y_pred"),
    "model_comparison_csv": (
        "model_id", "display_name", "parent", "precision", "epochs", "n_test",
        "clean_accuracy_full_test", "macro_f1_full_test", "state_dict_bytes", "state_dict_mb",
        "latency_b1_median_ms", "latency_b1_p95_ms", "latency_b256_median_ms",
        "latency_b256_p95_ms", "latency_b256_per_example_median_ms", "peak_rss_bytes",
        "peak_rss_delta_bytes", "peak_rss_sampled_delta_bytes",
    ),
    "robustness_curves_csv": (
        "model_id", "display_name", "attack", "epsilon", "n", "accuracy", "macro_f1",
        "robust_accuracy", "unit_id", "eps0_shared",
    ),
}
# per-example arrays stored in each unit's arrays.npz
UNIT_ARRAY_KEYS = ("ids", "y", "pred_clean", "pred_adv", "robust_correct", "linf")


def check_required_keys(doc: Any, contract_key: str, where: str = "") -> None:
    """Raise ``DataIntegrityError`` if ``doc`` lacks a key required by the contract."""
    if not isinstance(doc, dict):
        raise DataIntegrityError("%s: expected a JSON object" % (where or contract_key))
    missing = [k for k in JSON_REQUIRED_KEYS[contract_key] if k not in doc]
    if missing:
        raise DataIntegrityError("%s: missing keys %s" % (where or contract_key, missing))


# ---------------------------------------------------------------- strict config loader
def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


_STR, _INT, _NUM, _BOOL = "str", "int", "num", "bool"

# exact key sets and leaf types (DESIGN D4)
_CONFIG_SCHEMA: Dict[str, Any] = {
    "schema_version": _INT,
    "run_id": _STR,
    "claim": _STR,
    "seed": _INT,
    "split_dir": _STR,
    "siblings": {k: _STR for k in ("training", "quantization", "attacks", "efficiency", "data")},
    "training": {
        "epoch_candidates": [_INT],
        "threads": _INT,
        "keep_last": _INT,
        "batch_size_expected": _INT,
        "checkpoint_rule": _STR,
    },
    "budget": {
        "hard_cap_hours": _NUM,
        "reserve_hours": _NUM,
        "safety_multiplier": _NUM,
        "per_class_candidates": [_INT],
        "n_classes": _INT,
        "report_allowance_seconds": _NUM,
        "stage_overhead_seconds": _NUM,
        "n_stage_processes": _INT,
        "max_probe_minutes": _NUM,
    },
    "sources": {"split_seed": _STR, "calibration_per_class": _STR},
    "timing_probe": {
        "train_steps_ordinary": _INT,
        "train_steps_adversarial": _INT,
        "warmup_steps_dropped": _INT,
        "p90_weight": _NUM,
        "max_step_cv": _NUM,
        "attack_probe_per_class": _INT,
        "contention_cpu_percent": _NUM,
    },
    "robustness": {
        "split_partition": _STR,
        "per_class_candidates": [_INT],
        "subset_seed_name": _STR,
        "epsilons": [_NUM],
        "batch_size": _INT,
        "attacks": {
            "fgsm": {"source_cfg_name": _STR},
            "pgd5": {
                "kind": _STR,
                "steps": _INT,
                "alpha_fraction": _NUM,
                "restarts": _INT,
                "random_start": _BOOL,
            },
        },
        "methods": {"fp32": _STR, "int8": _STR},
    },
    "efficiency": {"threads": _INT, "rows_partition": _STR},
    "val_check": {"blocking": [_STR], "informational": [_STR]},
    "report": {"score_pp_tolerance": _NUM, "ratio_tolerance": _NUM, "qat_artifact": _STR},
}


def _check_node(node: Any, schema: Any, path: str, errors: List[str]) -> None:
    if isinstance(schema, dict):
        if not isinstance(node, dict):
            errors.append("%s: expected object" % path)
            return
        missing = sorted(set(schema) - set(node))
        extra = sorted(set(node) - set(schema))
        if missing:
            errors.append("%s: missing keys %s" % (path, missing))
        if extra:
            errors.append("%s: unknown keys %s" % (path, extra))
        for key in schema:
            if key in node:
                _check_node(node[key], schema[key], "%s.%s" % (path, key) if path else key, errors)
    elif isinstance(schema, list):
        if not isinstance(node, list) or not node:
            errors.append("%s: expected non-empty list" % path)
            return
        for i, item in enumerate(node):
            _check_node(item, schema[0], "%s[%d]" % (path, i), errors)
    elif schema == _STR:
        if not isinstance(node, str) or not node:
            errors.append("%s: expected non-empty string" % path)
        elif "..." in node:
            errors.append("%s: placeholder text '...' is not allowed" % path)
    elif schema == _INT:
        if not _is_int(node):
            errors.append("%s: expected integer" % path)
    elif schema == _NUM:
        if not _is_num(node):
            errors.append("%s: expected finite number" % path)
    elif schema == _BOOL:
        if not isinstance(node, bool):
            errors.append("%s: expected boolean" % path)


def validate_config(cfg: Any) -> Dict[str, Any]:
    """Strict check of the overlay config (exact keys, types, frozen design values)."""
    errors: List[str] = []
    _check_node(cfg, _CONFIG_SCHEMA, "", errors)
    if errors:
        raise ConfigError("invalid config: " + "; ".join(errors))

    def need(cond: bool, message: str) -> None:
        if not cond:
            errors.append(message)

    need(cfg["schema_version"] == 1, "schema_version must be 1")
    need(cfg["run_id"] == RUN_ID, "run_id must be %r" % RUN_ID)
    need(cfg["seed"] == 1, "seed must be 1")
    need(cfg["training"]["epoch_candidates"] == [25, 10, 5], "training.epoch_candidates must be [25, 10, 5]")
    need(cfg["training"]["threads"] == 2, "training.threads must be 2")
    need(cfg["training"]["batch_size_expected"] == 512, "training.batch_size_expected must be 512")
    need(cfg["budget"]["hard_cap_hours"] == 9.0, "budget.hard_cap_hours must be 9.0")
    need(cfg["budget"]["reserve_hours"] == 3.0, "budget.reserve_hours must be 3.0")
    need(cfg["budget"]["safety_multiplier"] >= 1.0, "budget.safety_multiplier must be >= 1")
    need(cfg["budget"]["per_class_candidates"] == [10, 5], "budget.per_class_candidates must be [10, 5]")
    need(cfg["budget"]["n_classes"] == N_CLASSES, "budget.n_classes must be 27")
    need(cfg["budget"]["n_stage_processes"] == 1, "budget.n_stage_processes must be 1")
    need(cfg["budget"]["max_probe_minutes"] > 0, "budget.max_probe_minutes must be > 0")
    probe = cfg["timing_probe"]
    need(probe["warmup_steps_dropped"] >= 0, "timing_probe.warmup_steps_dropped must be >= 0")
    need(
        probe["train_steps_ordinary"] > probe["warmup_steps_dropped"]
        and probe["train_steps_adversarial"] > probe["warmup_steps_dropped"],
        "timing_probe step counts must exceed warmup_steps_dropped",
    )
    need(0 < probe["p90_weight"] <= 1, "timing_probe.p90_weight must be in (0, 1]")
    need(probe["max_step_cv"] > 0, "timing_probe.max_step_cv must be > 0")
    need(probe["attack_probe_per_class"] >= 1, "timing_probe.attack_probe_per_class must be >= 1")
    rob = cfg["robustness"]
    need(rob["split_partition"] == "test", "robustness.split_partition must be 'test'")
    need(rob["per_class_candidates"] == [10, 5], "robustness.per_class_candidates must be [10, 5]")
    need(
        [float(e) for e in rob["epsilons"]] == [0.0, 0.05, 0.1],
        "robustness.epsilons must be [0.0, 0.05, 0.1]",
    )
    need(rob["batch_size"] == 64, "robustness.batch_size must be 64")
    pgd = rob["attacks"]["pgd5"]
    need(pgd["kind"] == "pgd" and pgd["steps"] == 5 and pgd["restarts"] == 1, "pgd5 must be 5 steps x 1 restart")
    need(pgd["random_start"] is True, "pgd5.random_start must be true")
    need(pgd["alpha_fraction"] == 0.25, "pgd5.alpha_fraction must be 0.25")
    need(rob["attacks"]["fgsm"]["source_cfg_name"] == "fgsm", "fgsm.source_cfg_name must be 'fgsm'")
    need(
        rob["methods"] == {"fp32": "fp32_direct", "int8": "bpda_int8"},
        "robustness.methods must be fp32_direct / bpda_int8",
    )
    need(cfg["efficiency"]["threads"] == 1, "efficiency.threads must be 1")
    need(cfg["efficiency"]["rows_partition"] == "validation", "efficiency.rows_partition must be 'validation'")
    need(cfg["report"]["score_pp_tolerance"] > 0, "report.score_pp_tolerance must be > 0")
    need(0 < cfg["report"]["ratio_tolerance"] < 1, "report.ratio_tolerance must be in (0, 1)")
    if errors:
        raise ConfigError("invalid config: " + "; ".join(errors))
    return cfg


def load_config(path: Any) -> Dict[str, Any]:
    """Read and strictly validate the overlay config. Raises ``ConfigError`` (exit 2)."""
    try:
        with open(str(path), "r", encoding="utf-8") as stream:
            cfg = json.load(stream)
    except OSError as exc:
        raise ConfigError("cannot read config %s: %s" % (path, exc))
    except ValueError as exc:
        raise ConfigError("config %s is not valid JSON: %s" % (path, exc))
    return validate_config(cfg)


def is_finite_number(value: Any) -> bool:
    return _is_num(value)
