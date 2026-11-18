"""Synthetic tables and configs for Milestone 2A split tests (no real data, no model)."""
import copy
import json

import numpy as np

from research.common import ROOT

REAL_CONFIG = ROOT / "configs/m2/data.json"
SOURCE = {"source_digest": "0" * 64, "revision": "test"}


def real_config():
    return json.loads(REAL_CONFIG.read_text(encoding="utf-8"))


def synthetic_config(n_classes=4, n_features=3, seeds=(11, 12, 13), minima=(10, 3, 3),
                     band=((0.6, 0.8), (0.08, 0.22), (0.08, 0.22)), proposals=4000, patience=1200):
    cfg = copy.deepcopy(real_config())
    cfg["dataset"].update(
        rows=0, features=n_features, classes=n_classes,
        feature_names=[f"f{i}" for i in range(n_features)],
        class_names=[f"c{i}" for i in range(n_classes)])
    cfg["split"].update(
        seeds=list(seeds), proposals=proposals, patience=patience,
        hard_minima=dict(zip(("train", "validation", "test"), minima)),
        acceptance_band={p: list(b) for p, b in zip(("train", "validation", "test"), band)})
    return cfg


def synthetic_table(n_classes=4, n_features=3, groups_per_class=40, n_conflicts=5, seed=0):
    """Rows = distinct vectors repeated 1-3 times, a few large groups, a few label conflicts."""
    rng = np.random.default_rng(seed)
    raw, labels = [], []
    for c in range(n_classes):
        for g in range(groups_per_class):
            size = 15 if g == 0 else int(rng.integers(1, 4))
            vector = rng.random(n_features)
            raw.extend([vector] * size)
            labels.extend([c] * size)
    vectors = np.unique(np.array(raw), axis=0)
    for k in range(n_conflicts):
        vec = vectors[k * 7 % len(vectors)]
        original = labels[int(np.flatnonzero((np.array(raw) == vec).all(1))[0])]
        raw.append(vec)
        labels.append((original + 1) % n_classes)
    order = rng.permutation(len(raw))
    return np.array(raw, dtype=np.float64)[order], np.array(labels, dtype=np.int64)[order]


def finish(cfg, raw):
    cfg["dataset"]["rows"] = len(raw)
    return cfg


# ---- tiny-model helpers shared by the quantization and attack tests (no real data) ----
TINY_ARCH = {"input_dim": 39, "d_model": 16, "n_heads": 4, "n_layers": 1, "d_ff": 32, "n_classes": 27,
             "max_seq_len": 50, "dropout": 0.1}
TINY_ISLANDS = 11  # embedding, 4 attention, 2 feed-forward, 2 classification head, 2 anomaly scorer


def tiny_model(seed=5, arch=None):
    import torch

    from research.common import make_model
    torch.manual_seed(seed)
    return make_model(arch or TINY_ARCH).eval()


def split_arrays(per_class=40, seed=0, n_features=39, n_classes=27):
    """Synthetic {'train','validation'} arrays shaped like m2_split.load_arrays output."""
    rng = np.random.default_rng(seed)
    out = {}
    for i, name in enumerate(("train", "validation")):
        y = np.repeat(np.arange(n_classes), per_class)
        out[name] = {"ids": np.arange(len(y)) + i * 100000, "x": rng.random((len(y), n_features)).astype("float32"), "y": y}
    return out
