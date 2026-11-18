"""Numpy metrics, epsilon strings, percentage-point differences and curve assembly.

Pure module: standard library + numpy. Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .common import (
    ATTACK_NAMES,
    CLEAN_ATTACK,
    CLEAN_UNIT,
    DataIntegrityError,
    EPSILONS,
    N_CLASSES,
    UNIT_RESULT_KEYS,
    eps_str,
    unit_id,
)

__all__ = [
    "accuracy",
    "macro_f1",
    "eps_str",
    "pp_diff",
    "eps0_consistency",
    "build_curves",
    "clean_columns",
]


def _pair(y: Any, pred: Any):
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    if len(y) != len(pred):
        raise ValueError("y and pred differ in length (%d vs %d)" % (len(y), len(pred)))
    if len(y) == 0:
        raise ValueError("empty input")
    return y, pred


def accuracy(y: Any, pred: Any) -> float:
    y, pred = _pair(y, pred)
    return float(np.mean(y == pred))


def macro_f1(y: Any, pred: Any, labels: Optional[Iterable[int]] = None, zero_division: float = 0.0) -> float:
    """Unweighted mean of per-class F1 over ``labels`` (default range(27)).

    Same convention as ``sklearn.metrics.f1_score(average="macro", labels=labels, zero_division=0)``:
    a class absent from ``y`` and ``pred`` contributes ``zero_division``; a class absent from ``y``
    but predicted contributes 0 and is counted in the average.
    """
    y, pred = _pair(y, pred)
    label_list = list(range(N_CLASSES)) if labels is None else [int(c) for c in labels]
    if not label_list:
        raise ValueError("labels must not be empty")
    scores = []
    for c in label_list:
        tp = int(np.sum((y == c) & (pred == c)))
        fp = int(np.sum((y != c) & (pred == c)))
        fn = int(np.sum((y == c) & (pred != c)))
        denom = 2 * tp + fp + fn
        scores.append(float(zero_division) if denom == 0 else 2.0 * tp / denom)
    return float(np.mean(scores))


def pp_diff(a: float, b: float) -> float:
    """Score difference ``b - a`` in percentage points, rounded to 2 decimals (scores are fractions)."""
    return round((float(b) - float(a)) * 100.0, 2) + 0.0


def eps0_consistency(
    subset_ids: Any,
    subset_pred: Any,
    full_ids: Any,
    full_pred: Any,
    subset_logits: Any = None,
    full_logits: Any = None,
) -> Dict[str, Any]:
    """Record-only comparison of eps-0 subset predictions with full-test predictions (same ids).

    Never raises, even if every prediction differs or the inputs are malformed.
    """
    try:
        subset_ids = np.asarray(subset_ids, dtype=np.int64).reshape(-1)
        subset_pred = np.asarray(subset_pred, dtype=np.int64).reshape(-1)
        full_ids = np.asarray(full_ids, dtype=np.int64).reshape(-1)
        full_pred = np.asarray(full_pred, dtype=np.int64).reshape(-1)
        position = {int(i): n for n, i in enumerate(full_ids)}
        rows = [position.get(int(i), -1) for i in subset_ids]
        found = np.array([r >= 0 for r in rows], dtype=bool)
        idx = np.array([r for r in rows if r >= 0], dtype=np.int64)
        mismatches = int(np.sum(subset_pred[found] != full_pred[idx]))
        max_diff = None
        if subset_logits is not None and full_logits is not None:
            a = np.asarray(subset_logits, dtype=np.float64)[found]
            b = np.asarray(full_logits, dtype=np.float64)[idx]
            max_diff = float(np.max(np.abs(a - b))) if len(idx) else 0.0
        return {
            "behaviour": "record_only",
            "n_compared": int(found.sum()),
            "n_missing_ids": int((~found).sum()),
            "eps0_vs_full_test_mismatches": mismatches,
            "eps0_max_abs_logit_diff": max_diff,
        }
    except Exception as exc:  # record-only: a malformed input is recorded, never raised
        return {
            "behaviour": "record_only",
            "n_compared": 0,
            "n_missing_ids": None,
            "eps0_vs_full_test_mismatches": None,
            "eps0_max_abs_logit_diff": None,
            "error": "%s: %s" % (type(exc).__name__, exc),
        }


def build_curves(
    units: Dict[str, Dict[str, Dict[str, Any]]],
    model_ids: Sequence[str],
    display_names: Dict[str, str],
    epsilons: Sequence[float] = EPSILONS,
    attacks: Sequence[str] = ATTACK_NAMES,
) -> List[Dict[str, Any]]:
    """Rows of ``robustness_curves.csv`` (subset metrics only).

    ``units[model_id][unit_id]`` holds ``n``, ``accuracy``, ``macro_f1``, ``robust_accuracy``. The eps-0
    point of both curves comes from the single ``clean_eps0.0`` unit (same ``unit_id``,
    ``eps0_shared=True``). A missing unit raises ``DataIntegrityError``.
    """
    rows: List[Dict[str, Any]] = []
    for model_id in model_ids:
        if model_id not in units:
            raise DataIntegrityError("no robustness units for model %s" % model_id)
        for attack in attacks:
            for eps in epsilons:
                shared = float(eps) == 0.0
                uid = CLEAN_UNIT if shared else unit_id(attack, eps)
                if uid not in units[model_id]:
                    raise DataIntegrityError("missing robustness unit %s/%s" % (model_id, uid))
                u = units[model_id][uid]
                missing = [k for k in UNIT_RESULT_KEYS if k not in u]
                if missing:
                    raise DataIntegrityError("unit %s/%s lacks keys %s" % (model_id, uid, missing))
                rows.append(
                    {
                        "model_id": model_id,
                        "display_name": display_names[model_id],
                        "attack": attack,
                        "epsilon": eps_str(eps),
                        "n": int(u["n"]),
                        "accuracy": float(u["accuracy"]),
                        "macro_f1": float(u["macro_f1"]),
                        "robust_accuracy": float(u["robust_accuracy"]),
                        "unit_id": uid,
                        "eps0_shared": shared,
                    }
                )
    return rows


def clean_columns(full_test: Dict[str, Any], subset_clean: Dict[str, Any]) -> Dict[str, Any]:
    """Full-test clean and subset clean results as separate, explicitly named fields."""
    return {
        "clean_accuracy_full_test": float(full_test["accuracy"]),
        "macro_f1_full_test": float(full_test["macro_f1"]),
        "n_full_test": int(full_test["n"]),
        "clean_accuracy_robustness_subset": float(subset_clean["accuracy"]),
        "macro_f1_robustness_subset": float(subset_clean["macro_f1"]),
        "n_robustness_subset": int(subset_clean["n"]),
    }
