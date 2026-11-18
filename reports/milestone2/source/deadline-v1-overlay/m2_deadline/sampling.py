"""Deterministic stratified subset rule (DESIGN 6.1) and ID/label hashes.

Pure module: standard library + numpy. Python 3.9 compatible.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable

import numpy as np

from .common import DataIntegrityError, N_CLASSES

DEFAULT_SEED_NAME = "m2-deadline-robust-subset"


def ids_sha256(ids: Iterable[Any]) -> str:
    """Canonical-JSON SHA-256 of the sorted int list.

    Same value as ``research.common.digest([int(i) for i in sorted(ids)])``, which is what
    ``m2_split.id_hash`` returns and what ``evaluate_unit`` records as ``example_id_sha256``.
    """
    payload = json.dumps(
        [int(i) for i in sorted(int(x) for x in ids)],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def label_sha256(y: Any) -> str:
    """``sha256`` of the labels as little-endian int64 bytes (matches ``_label_hash``)."""
    return hashlib.sha256(np.asarray(y, dtype="<i8").tobytes()).hexdigest()


def class_seed(seed_name: str, split_seed: int, class_index: int) -> int:
    digest = hashlib.sha256(("%s|%d|%d" % (seed_name, int(split_seed), int(class_index))).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def stratified_subset(
    ids: Any,
    labels: Any,
    k: int,
    split_seed: int,
    n_classes: int = N_CLASSES,
    seed_name: str = DEFAULT_SEED_NAME,
) -> Dict[str, Any]:
    """Pick ``min(k, available)`` rows per class, independent of input row order.

    Returns ``ids`` (sorted, int64), ``labels`` (aligned), ``counts`` (class -> available/selected),
    ``n_actual``, ``ids_sha256``, ``classes_present`` and ``classes_missing``.
    """
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(ids) != len(labels):
        raise DataIntegrityError("ids and labels differ in length (%d vs %d)" % (len(ids), len(labels)))
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or int(k) < 1:
        raise DataIntegrityError("k must be a positive integer, got %r" % (k,))
    if len(np.unique(ids)) != len(ids):
        raise DataIntegrityError("duplicate example ids")
    if len(labels) and (labels.min() < 0 or labels.max() >= n_classes):
        raise DataIntegrityError("label outside 0..%d" % (n_classes - 1))

    order = np.argsort(ids, kind="stable")
    ids_sorted = ids[order]
    labels_sorted = labels[order]

    selected = []
    counts: Dict[int, Dict[str, int]] = {}
    for c in range(n_classes):
        class_ids = ids_sorted[labels_sorted == c]
        n_c = int(len(class_ids))
        take = min(int(k), n_c)
        counts[c] = {"available": n_c, "selected": take}
        if take == 0:
            continue
        rng = np.random.default_rng(class_seed(seed_name, split_seed, c))
        pos = rng.choice(n_c, take, replace=False)
        selected.append(np.sort(class_ids[pos]))
    chosen = np.sort(np.concatenate(selected)) if selected else np.zeros(0, dtype=np.int64)

    index = np.searchsorted(ids_sorted, chosen)
    return {
        "ids": chosen.astype(np.int64),
        "labels": labels_sorted[index].astype(np.int64),
        "counts": counts,
        "n_actual": int(len(chosen)),
        "ids_sha256": ids_sha256(chosen),
        "classes_present": [c for c in range(n_classes) if counts[c]["selected"] > 0],
        "classes_missing": [c for c in range(n_classes) if counts[c]["available"] == 0],
    }


def counts_document(subset: Dict[str, Any], k: int, split_seed: int, seed_name: str = DEFAULT_SEED_NAME) -> Dict[str, Any]:
    """JSON-ready ``robustness_subset_counts.json`` content (class keys as strings)."""
    return {
        "per_class_target": int(k),
        "n_actual": int(subset["n_actual"]),
        "counts": {str(c): dict(v) for c, v in sorted(subset["counts"].items())},
        "classes_present": list(subset["classes_present"]),
        "classes_missing": list(subset["classes_missing"]),
        "subset_ids_sha256": subset["ids_sha256"],
        "split_seed": int(split_seed),
        "seed_name": seed_name,
    }
