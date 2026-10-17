"""Fixed duplicate-group split, training-only scaling and isolated sequences."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .common import DATA_SHA, UPSTREAM, atomic_json, digest, sha256


def partitions(y, groups, split_seed):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    remaining, test = next(outer.split(np.zeros(len(y)), y, groups))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed + 1)
    a, b = next(inner.split(np.zeros(len(remaining)), y[remaining], groups[remaining]))
    result = {"train": remaining[a], "validation": remaining[b], "test": test}
    for name, ids in result.items():
        if set(np.unique(y[ids])) != set(np.unique(y)):
            raise ValueError(f"Split {name} lost a class; cannot run this protocol")
    for i, name in enumerate(result):
        for other in list(result)[i + 1:]:
            assert not set(groups[result[name]]) & set(groups[result[other]])
    return result


def stratified_cap(ids, y, per_class, seed):
    if per_class is None:
        return np.sort(ids)
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([
        rng.choice(ids[y[ids] == c], min(per_class, sum(y[ids] == c)), replace=False)
        for c in np.unique(y)]))


def build_data(config, folder):
    path = UPSTREAM / "data/balanced_CICIoT.csv"
    data_hash = sha256(path)
    if data_hash != DATA_SHA:
        raise ValueError("Processed CICIoT hash differs from the pinned data")
    df = pd.read_csv(path)
    features = df.drop(columns="Label")
    # Reject an unexpected schema instead of silently discarding identifiers.
    if features.shape[1] != 39 or not all(pd.api.types.is_numeric_dtype(t) for t in features.dtypes):
        raise ValueError("Expected 39 numeric CICIoT features")
    raw = features.to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite raw features; an explicit cleaning decision is required")
    labels = sorted(df.Label.unique().tolist())
    if len(labels) != 27 or "Benign_Final" not in labels:
        raise ValueError("Wrong dataset or label schema")
    y = np.array([labels.index(label) for label in df.Label], dtype=np.int64)
    # SHA-256 of canonical float64 feature bytes, independent of label. This
    # keeps conflicting labels and identical features in the same partition.
    import hashlib
    groups = np.array([hashlib.sha256(row.astype("<f8").tobytes()).hexdigest() for row in raw])
    splits = partitions(y, groups, config["split_seed"])
    ids = {name: stratified_cap(ix, y, config["per_class"].get(name), config["split_seed"] + j + 10)
           for j, (name, ix) in enumerate(splits.items())}
    scaler = StandardScaler().fit(raw[ids["train"]])
    z = scaler.transform(raw).astype(np.float32)
    preprocessing = {"kind": "StandardScaler", "fit_row_ids": ids["train"].tolist(),
                     "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                     "feature_order": features.columns.tolist(), "labels": labels}
    manifest = {
        "dataset": "PAVE balanced CICIoT2023 multiclass", "dataset_sha256": data_hash,
        "data_origin": "https://github.com/robotumbel/PAVE_artifact/blob/"
                       "e73911bbdb241f3fb6c5de346139010c4aadefd4/data/balanced_CICIoT.csv",
        "rows": len(df), "features": features.columns.tolist(), "labels": labels,
        "label_counts": {str(k): int(v) for k, v in df.Label.value_counts().items()},
        "duplicate_feature_rows": int(len(df) - len(np.unique(groups))),
        "duplicate_full_rows": int(df.duplicated().sum()),
        "conflicting_label_groups": int(pd.DataFrame({"g": groups, "y": y}).groupby("g").y.nunique().gt(1).sum()),
        "provenance_identifiers": [],
        "grouping": "SHA256 exact numeric feature row; device/capture/flow IDs unavailable",
        "split_seed": config["split_seed"], "full_split_row_ids": {k: v.tolist() for k, v in splits.items()},
        "selected_row_ids": {k: v.tolist() for k, v in ids.items()},
        "selected_group_hashes": {k: groups[v].tolist() for k, v in ids.items()},
        "sequence": {"length": config["sequence_length"], "representation": "one row repeated; no temporal provenance",
                     "train_noise_std": config["train_noise_std"], "eval_noise_std": 0},
        "preprocessing_sha256": digest(preprocessing),
    }
    manifest["split_sha256"] = digest(manifest["full_split_row_ids"])
    manifest["selected_sha256"] = digest(manifest["selected_row_ids"])
    manifest["manifest_sha256"] = digest(manifest)
    folder = Path(folder)
    for filename, value in (("manifest.json", manifest), ("preprocessing.json", preprocessing)):
        p = folder / filename
        if p.exists():
            if json.loads(p.read_text()) != value:
                raise ValueError(f"Incompatible persisted data: {p}")
        else:
            atomic_json(p, value)
    return {
        "x": {k: torch.from_numpy(z[v].copy()) for k, v in ids.items()},
        "y": {k: torch.from_numpy(y[v].copy()) for k, v in ids.items()},
        "ids": ids, "labels": labels, "manifest": manifest, "preprocessing": preprocessing,
        "lower": torch.from_numpy(z[ids["train"]].min(0)),
        "upper": torch.from_numpy(z[ids["train"]].max(0)),
    }


def sequences(x, length, noise_std=0, generator=None):
    # Construct only from rows already belonging to one partition. A sample
    # never contains another flow, let alone another partition's flow.
    result = x[:, None, :].repeat(1, length, 1)
    if noise_std:
        result += torch.randn(result.shape, generator=generator) * noise_std
    return result
