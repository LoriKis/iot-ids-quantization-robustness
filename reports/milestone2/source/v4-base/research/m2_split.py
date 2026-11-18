"""Milestone 2A data protocol: group-disjoint split, train-only scaler, partition guard.

One deterministic split is built from labels and group sizes only (never model
output). Every variant and seed loads the same frozen split and scaler through
``load_arrays``, which re-verifies every hash and refuses test features unless
the caller states ``purpose="range_audit"``. No function here accepts a model.

    py -B -m research.m2_split --config configs\\m2\\data.json
    py -B -m research.m2_split --config configs\\m2\\data.json --audit-only
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .common import ROOT, atomic_json, digest, sha256

log = logging.getLogger("m2_split")

PARTITIONS = ("train", "validation", "test")
SCHEMA_VERSION = 1
ALGORITHM = "lexicographic-VJ-local-search"
ALGORITHM_VERSION = 1
ALGORITHM_VERSIONS = (1, 2)
FRAMING = ("This is a new group-disjoint evaluation partition within a previously examined dataset, "
           "not a completely untouched external benchmark. The dataset was seen by Milestone 1; the "
           "earlier test partition was evaluated; identical feature vectors with conflicting labels "
           "remain in the data.")
ARTIFACTS = ("split_attempts.json", "group_manifest.json", "conflicting_labels.json",
             "scaler_minmax.json", "range_audit.json", "split_report.md", "split_m2.json")
WEAK_GROUPS = 3
V2_WEAK_GROUPS = 10
TOLERANCE = 1e-12


class ConfigError(ValueError):
    """A frozen configuration file is malformed."""


class DataIntegrityError(RuntimeError):
    """The dataset, split manifest or scaler fails a hash or invariant check."""


class SplitInfeasibleError(RuntimeError):
    """No listed seed produced a split meeting the hard minima and acceptance band."""

    def __init__(self, message, attempts=None, rung_lower_bounds=None):
        super().__init__(message)
        self.attempts = attempts or []
        self.rung_lower_bounds = rung_lower_bounds


class PartitionGuardError(RuntimeError):
    """A caller asked for partition data it is not allowed to see."""


@dataclass(frozen=True)
class FreezeBase:
    """Base of the verified evaluation freeze (research.m2b_state.EvalFreeze subclasses it).

    load_arrays(purpose='final_test') accepts only an instance whose file still hashes to `sha256`.
    """
    path: Path
    sha256: str
    data: dict


# --------------------------------------------------------------------- config
def _exact_keys(obj, keys, where):
    if not isinstance(obj, dict):
        raise ConfigError(f"{where} must be an object")
    missing, unknown = set(keys) - set(obj), set(obj) - set(keys)
    if missing or unknown:
        raise ConfigError(f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_config(cfg):
    """Raise ConfigError unless ``cfg`` is a well-formed ``configs/m2/data.json``."""
    _exact_keys(cfg, ("schema_version", "frozen_date_local", "claim", "dataset", "split",
                      "grouping", "scaler", "range_audit"), "config")
    if cfg["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("frozen_date_local", "claim"):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ConfigError(f"{key} must be a non-empty string")
    d = cfg["dataset"]
    _exact_keys(d, ("path", "sha256", "rows", "features", "classes", "feature_names",
                    "class_names", "class_names_source"), "dataset")
    if not isinstance(d["path"], str) or not isinstance(d["class_names_source"], str):
        raise ConfigError("dataset.path and dataset.class_names_source must be strings")
    if not (isinstance(d["sha256"], str) and len(d["sha256"]) == 64
            and all(ch in "0123456789abcdef" for ch in d["sha256"])):
        raise ConfigError("dataset.sha256 must be 64 lowercase hex characters")
    for key in ("rows", "features", "classes"):
        if not _is_int(d[key]) or d[key] <= 0:
            raise ConfigError(f"dataset.{key} must be a positive integer")
    for names, count in (("feature_names", "features"), ("class_names", "classes")):
        value = d[names]
        if (not isinstance(value, list) or not all(isinstance(v, str) for v in value)
                or len(set(value)) != len(value) or len(value) != d[count]):
            raise ConfigError(f"dataset.{names} must be {d[count]} unique strings")
    s = cfg["split"]
    version = s.get("algorithm_version") if isinstance(s, dict) else None
    split_keys = ("algorithm", "algorithm_version", "seeds", "targets", "hard_minima",
                  "acceptance_band", "feature_variability", "proposals", "patience", "output_dir")
    _exact_keys(s, split_keys + (("group_diversity",) if version == 2 else ()), "split")
    _exact_keys(s["feature_variability"], ("min_distinct_train_values", "at_risk_max_minority_groups"),
                "split.feature_variability")
    fv = s["feature_variability"]
    if fv["min_distinct_train_values"] != 2:
        raise ConfigError("split.feature_variability.min_distinct_train_values must be 2")
    if not _is_int(fv["at_risk_max_minority_groups"]) or fv["at_risk_max_minority_groups"] < 1:
        raise ConfigError("split.feature_variability.at_risk_max_minority_groups must be a positive integer")
    if s["algorithm"] != ALGORITHM or s["algorithm_version"] not in ALGORITHM_VERSIONS \
            or isinstance(s["algorithm_version"], bool):
        raise ConfigError(f"split algorithm must be {ALGORITHM} with version in {list(ALGORITHM_VERSIONS)}")
    if (not isinstance(s["seeds"], list) or not s["seeds"] or not all(_is_int(v) for v in s["seeds"])
            or len(set(s["seeds"])) != len(s["seeds"])):
        raise ConfigError("split.seeds must be a non-empty list of unique integers")
    for key in ("targets", "hard_minima", "acceptance_band"):
        _exact_keys(s[key], PARTITIONS, f"split.{key}")
    if not all(_is_num(s["targets"][p]) and 0 < s["targets"][p] < 1 for p in PARTITIONS):
        raise ConfigError("split.targets must be fractions in (0, 1)")
    if abs(sum(s["targets"].values()) - 1.0) > 1e-9:
        raise ConfigError("split.targets must sum to 1")
    if not all(_is_int(s["hard_minima"][p]) and s["hard_minima"][p] >= 1 for p in PARTITIONS):
        raise ConfigError("split.hard_minima must be positive integers")
    for p in PARTITIONS:
        band = s["acceptance_band"][p]
        if (not isinstance(band, list) or len(band) != 2 or not all(_is_num(v) for v in band)
                or not 0 <= band[0] < band[1] <= 1):
            raise ConfigError(f"split.acceptance_band.{p} must be [lo, hi] with 0 <= lo < hi <= 1")
        if not band[0] <= s["targets"][p] <= band[1]:
            raise ConfigError(f"split.targets.{p} lies outside its acceptance band")
    for key in ("proposals", "patience"):
        if not _is_int(s[key]) or s[key] <= 0:
            raise ConfigError(f"split.{key} must be a positive integer")
    if not isinstance(s["output_dir"], str) or not s["output_dir"]:
        raise ConfigError("split.output_dir must be a non-empty string")
    if s["algorithm_version"] == 2:
        _validate_group_diversity(s["group_diversity"])
    g = cfg["grouping"]
    _exact_keys(g, ("rule", "hash", "dtype", "label_excluded"), "grouping")
    if g["hash"] != "sha256" or g["dtype"] != "<f8" or g["label_excluded"] is not True \
            or not isinstance(g["rule"], str):
        raise ConfigError("grouping must be sha256 over '<f8' feature bytes with the label excluded")
    sc = cfg["scaler"]
    _exact_keys(sc, ("kind", "feature_range", "clip", "fit_partition"), "scaler")
    if (sc["kind"] != "MinMaxScaler" or sc["feature_range"] != [0, 1] or sc["clip"] is not False
            or sc["fit_partition"] != "train"):
        raise ConfigError("scaler must be MinMaxScaler(feature_range=(0, 1), clip=False) fitted on train")
    audit = cfg["range_audit"]
    _exact_keys(audit, ("epsilon_grid",), "range_audit")
    grid = audit["epsilon_grid"]
    if (not isinstance(grid, list) or not grid or not all(_is_num(v) and 0 <= v <= 1 for v in grid)
            or grid[0] != 0 or any(b <= a for a, b in zip(grid, grid[1:]))):
        raise ConfigError("range_audit.epsilon_grid must be strictly increasing from 0, within [0, 1]")
    return cfg


def _validate_group_diversity(gd):
    _exact_keys(gd, ("unit", "weight_group_share", "absolute_floor_groups", "ladder", "order"),
                "split.group_diversity")
    if not isinstance(gd["unit"], str) or not isinstance(gd["order"], str):
        raise ConfigError("split.group_diversity.unit and .order must be strings")
    if not _is_num(gd["weight_group_share"]) or gd["weight_group_share"] < 0:
        raise ConfigError("split.group_diversity.weight_group_share must be a non-negative number")
    _exact_keys(gd["absolute_floor_groups"], PARTITIONS, "split.group_diversity.absolute_floor_groups")
    if not all(_is_int(gd["absolute_floor_groups"][p]) and gd["absolute_floor_groups"][p] >= 1
               for p in PARTITIONS):
        raise ConfigError("split.group_diversity.absolute_floor_groups must be positive integers")
    ladder = gd["ladder"]
    if not isinstance(ladder, list) or not ladder:
        raise ConfigError("split.group_diversity.ladder must be a non-empty list")
    for k, rung in enumerate(ladder, start=1):
        _exact_keys(rung, ("rung", "floor_fraction"), f"split.group_diversity.ladder[{k - 1}]")
        if rung["rung"] != k:
            raise ConfigError("split.group_diversity.ladder rungs must be numbered 1..n in order")
        _exact_keys(rung["floor_fraction"], PARTITIONS, f"ladder rung {k} floor_fraction")
        if not all(_is_num(rung["floor_fraction"][p]) and 0 < rung["floor_fraction"][p] <= 1
                   for p in PARTITIONS):
            raise ConfigError(f"ladder rung {k} floor_fraction values must be in (0, 1]")


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config not found: {path}")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    validate_config(cfg)
    source = ROOT / cfg["dataset"]["class_names_source"]
    if not source.is_file():
        raise ConfigError(f"class_names_source not found: {source}")
    expected = json.loads(source.read_text(encoding="utf-8"))["dataset"]
    if expected["class_names"] != cfg["dataset"]["class_names"]:
        raise ConfigError("dataset.class_names differ from the Experiment C configuration")
    if expected["feature_names"] != cfg["dataset"]["feature_names"]:
        raise ConfigError("dataset.feature_names differ from the Experiment C configuration")
    return cfg


# ----------------------------------------------------------------------- data
def group_hashes(raw):
    """SHA-256 of each row as little-endian float64 bytes (identical to ``data.build_data``)."""
    return np.array([hashlib.sha256(row.astype("<f8").tobytes()).hexdigest() for row in raw])


def read_table(dataset):
    """Load and verify the processed CSV. ``dataset`` is the config's ``dataset`` block."""
    path = ROOT / dataset["path"]
    if not path.is_file():
        raise DataIntegrityError(f"Dataset not found: {path}")
    got = sha256(path)
    if got != dataset["sha256"]:
        raise DataIntegrityError(f"Dataset SHA-256 {got} differs from the frozen {dataset['sha256']}")
    df = pd.read_csv(path)
    if "Label" not in df.columns:
        raise DataIntegrityError("Dataset has no Label column")
    features = df.drop(columns="Label")
    if features.columns.tolist() != dataset["feature_names"]:
        raise DataIntegrityError("Feature columns differ from the frozen feature names")
    if features.shape != (dataset["rows"], dataset["features"]):
        raise DataIntegrityError(f"Dataset shape {features.shape} differs from the frozen shape")
    if not all(pd.api.types.is_numeric_dtype(t) for t in features.dtypes):
        raise DataIntegrityError("Non-numeric feature column")
    raw = features.to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise DataIntegrityError("Non-finite raw features; no silent cleaning is applied")
    labels = sorted(df.Label.unique().tolist())
    if labels != dataset["class_names"]:
        raise DataIntegrityError("Label names differ from the frozen class mapping")
    index = {name: i for i, name in enumerate(dataset["class_names"])}
    y = np.array([index[v] for v in df.Label], dtype=np.int64)
    return np.ascontiguousarray(raw), y


def _check_table(raw, y, n_classes):
    if raw.ndim != 2 or y.shape != (raw.shape[0],):
        raise DataIntegrityError("Table shape mismatch")
    if y.min() < 0 or y.max() >= n_classes:
        raise DataIntegrityError("Label id outside the class mapping")


# ------------------------------------------------------------ split algorithm
class _Objective:
    """V = total shortfall below hard minima (rows) plus the number of at-risk features that have
    no varying value in train; J = smooth deviation from the targets."""

    def __init__(self, class_totals, targets, minima, risk=None, div=None):
        # ``div`` (algorithm v2): {"G": distinct groups per class, "floors": 3 x classes,
        # "weight": share-term weight}; None keeps the v1 objective unchanged.
        self.risk = risk
        self.div = div
        self.totals = class_totals.astype(np.float64)
        self.n = float(class_totals.sum())
        self.t = np.asarray(targets, dtype=np.float64)
        self.m = np.asarray(minima, dtype=np.int64)[:, None]
        self.k = len(class_totals)

    def __call__(self, N, train_minority=None, g=None):
        v = int(np.maximum(0, self.m - N).sum())
        if self.risk is not None:
            v += int((train_minority == 0).sum())
        if self.div is not None:
            v += self.v_div(g)
        frac = N.sum(1) / self.n
        j = 4.0 * float(((frac - self.t) ** 2).sum())
        j += float((((N / self.totals) - self.t[:, None]) ** 2).sum()) / self.k
        if self.div is not None:
            j += self.div["weight"] * float(
                np.mean((g / self.div["G"][None, :] - self.t[:, None]) ** 2))
        return v, j

    def v_div(self, g):
        """Total shortfall of distinct groups below the per-class, per-partition floors."""
        return int(np.maximum(0, self.div["floors"] - g).sum())


def _group_table(y, groups, n_classes):
    uniq, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(uniq), n_classes), dtype=np.int64)
    np.add.at(counts, (inverse, y), 1)
    return uniq, inverse, counts


def _partition_counts(counts, assign):
    return np.stack([counts[assign == p].sum(0) for p in range(len(PARTITIONS))])


def _train_minority(risk, assign):
    return risk[assign == 0].sum(0) if risk is not None else None


def _greedy(counts, order, objective, presence=None):
    N = np.zeros((len(PARTITIONS), counts.shape[1]), dtype=np.int64)
    G = np.zeros_like(N) if presence is not None else None
    assign = np.zeros(len(counts), dtype=np.int64)
    risk = objective.risk
    seen = np.zeros(risk.shape[1], dtype=np.int64) if risk is not None else None
    for g in order:
        best = None
        for p in range(len(PARTITIONS)):
            trial = N.copy()
            trial[p] += counts[g]
            trial_seen = seen + risk[g] if (risk is not None and p == 0) else seen
            trial_g = None
            if G is not None:
                trial_g = G.copy()
                trial_g[p] += presence[g]
            key = (*objective(trial, trial_seen, trial_g), p)
            if best is None or key < best:
                best = key
        assign[g] = best[2]
        N[best[2]] += counts[g]
        if G is not None:
            G[best[2]] += presence[g]
        if risk is not None and best[2] == 0:
            seen = seen + risk[g]
    return (assign, N) if G is None else (assign, N, G)


def _search(counts, sizes, uniq, seed, split_cfg, risk=None, presence=None, div=None):
    """Greedy start, then lexicographic (V, J) local search with a seeded generator.

    ``presence`` (counts > 0) and ``div`` (see ``_Objective``) switch on the algorithm-v2 group floors.
    """
    objective = _Objective(counts.sum(0), [split_cfg["targets"][p] for p in PARTITIONS],
                           [split_cfg["hard_minima"][p] for p in PARTITIONS], risk, div)
    # Largest groups first; ties broken by group hash (``uniq`` is sorted, argsort is stable).
    order = np.argsort(-sizes, kind="stable")
    if presence is None:
        assign, N = _greedy(counts, order, objective)
        G = None
    else:
        assign, N, G = _greedy(counts, order, objective, presence)
    seen = _train_minority(risk, assign)
    start_v, start_j = objective(N, seen, G)
    cur_v, cur_j = start_v, start_j
    rng = np.random.default_rng(seed)
    n_groups, rejections, used = len(counts), 0, 0
    for used in range(1, split_cfg["proposals"] + 1):
        if rng.random() < 0.5:
            g = int(rng.integers(n_groups))
            q = int((assign[g] + rng.integers(1, len(PARTITIONS))) % len(PARTITIONS))
            changes = ((int(assign[g]), g, -1), (q, g, 1))
            moved = ((g, q),)
        else:
            g, h = (int(v) for v in rng.integers(n_groups, size=2))
            if assign[g] == assign[h]:
                rejections += 1
                if rejections >= split_cfg["patience"]:
                    break
                continue
            pg, ph = int(assign[g]), int(assign[h])
            changes = ((pg, g, -1), (ph, g, 1), (ph, h, -1), (pg, h, 1))
            moved = ((g, ph), (h, pg))
        trial = N.copy()
        trial_seen = seen.copy() if risk is not None else None
        trial_g = G.copy() if G is not None else None
        for part, grp, sign in changes:
            trial[part] += sign * counts[grp]
            if trial_g is not None:
                trial_g[part] += sign * presence[grp]
            if risk is not None and part == 0:
                trial_seen += sign * risk[grp]
        v, j = objective(trial, trial_seen, trial_g)
        if v < cur_v or (v == cur_v and j < cur_j - TOLERANCE):
            for grp, part in moved:
                assign[grp] = part
            N, seen, G, cur_v, cur_j, rejections = trial, trial_seen, trial_g, v, j, 0
        else:
            rejections += 1
            if rejections >= split_cfg["patience"]:
                break
    return {"assign": assign, "N": N, "G": G, "V": cur_v, "J": cur_j, "start_V": start_v,
            "start_J": start_j, "proposals_used": used}


def _binding_class(N, counts, split_cfg, class_names):
    minima = np.array([split_cfg["hard_minima"][p] for p in PARTITIONS])[:, None]
    shortfall = np.maximum(0, minima - N)
    if shortfall.sum() == 0:
        return None
    c = int(shortfall.sum(0).argmax())
    holders = np.flatnonzero(counts[:, c])
    sizes = np.sort(counts[holders, c])[::-1]
    return {"class_id": c, "class_name": class_names[c], "shortfall_by_partition":
            {p: int(shortfall[i, c]) for i, p in enumerate(PARTITIONS)},
            "class_rows": int(counts[:, c].sum()), "groups_holding_class": int(len(holders)),
            "largest_group_rows_of_class": [int(v) for v in sizes[:5]]}


def _acceptance_reason(N, v, split_cfg, v_div=0, unmet=(), rung=None):
    if v_div:
        return f"group floors unmet at rung {rung}: {', '.join(unmet)} (V_div={v_div})"
    if v:
        return f"hard minima or feature variability unmet (V={v}: row shortfall plus constant train features)"
    if (N.min(axis=1) <= 0).any():
        return "a class is missing from a partition"
    frac = N.sum(1) / N.sum()
    for i, p in enumerate(PARTITIONS):
        lo, hi = split_cfg["acceptance_band"][p]
        if not lo <= frac[i] <= hi:
            return f"{p} fraction {frac[i]:.4f} outside acceptance band [{lo}, {hi}]"
    return None


def _group_floors(G, rung, gd):
    """F_p(c) = min(max(abs_floor_p, ceil(frac_p * G_c)), floor(G_c / 3)) as a 3 x classes array."""
    out = np.zeros((len(PARTITIONS), len(G)), dtype=np.int64)
    for i, p in enumerate(PARTITIONS):
        want = np.ceil(np.round(rung["floor_fraction"][p] * G, 9)).astype(np.int64)
        out[i] = np.minimum(np.maximum(gd["absolute_floor_groups"][p], want), G // 3)
    return out


def _rung_lower_bound(counts, floors, split_cfg):
    """Structure-only bound: a partition meeting F_p(c) holds at least the F_p(c) smallest class-c
    group row counts of class c. Different classes are different rows, so the class sums add."""
    n = float(counts.sum())
    rows = np.zeros(floors.shape, dtype=np.int64)
    for c in range(counts.shape[1]):
        col = np.sort(counts[:, c][counts[:, c] > 0])
        for i in range(len(PARTITIONS)):
            rows[i, c] = col[:floors[i, c]].sum()
    total = rows.sum(1)
    fractions = {p: float(total[i] / n) for i, p in enumerate(PARTITIONS)}
    over = [p for p in PARTITIONS if fractions[p] > split_cfg["acceptance_band"][p][1]]
    return rows, {"min_rows": {p: int(total[i]) for i, p in enumerate(PARTITIONS)},
                  "min_fraction": fractions, "band_ceiling": {p: split_cfg["acceptance_band"][p][1]
                                                              for p in PARTITIONS},
                  "dropped": bool(over), "exceeds_ceiling_in": over}


def _binding_div(g, floors, class_names):
    shortfall = np.maximum(0, floors - g)
    if shortfall.sum() == 0:
        return None
    c = int(shortfall.sum(0).argmax())
    return {"class_id": c, "class_name": class_names[c],
            "group_shortfall_by_partition": {p: int(shortfall[i, c]) for i, p in enumerate(PARTITIONS)},
            "floors": {p: int(floors[i, c]) for i, p in enumerate(PARTITIONS)}}


def _variability_risk(raw, groups, split_cfg):
    """Pin features that only a handful of groups make non-constant.

    For each feature, the modal value over groups is the "constant" value. A feature is at risk
    when at most ``at_risk_max_minority_groups`` groups differ from it. The returned (groups x
    at-risk features) 0/1 matrix marks those minority groups; the objective charges one unit of V
    per at-risk feature without a minority group in train. Uses feature values only.
    """
    if raw is None:
        return None, []
    uniq, first = np.unique(groups, return_index=True)
    vectors = raw[first]
    cap = split_cfg["feature_variability"]["at_risk_max_minority_groups"]
    columns, features = [], []
    for f in range(vectors.shape[1]):
        values, counts = np.unique(vectors[:, f], return_counts=True)
        if len(values) < 2:
            raise DataIntegrityError(f"Feature {f} is constant over the whole dataset; it cannot be scaled")
        minority = vectors[:, f] != values[counts.argmax()]
        if minority.sum() <= cap:
            columns.append(minority.astype(np.int64))
            features.append(f)
    return (np.stack(columns, axis=1) if columns else None), features


def make_split(y, groups, cfg, raw=None):
    """Return the split for the FIRST accepted (rung, seed) pair, else raise.

    Algorithm v1 has one implicit rung. Algorithm v2 walks the pre-registered ladder of group-diversity
    floors in order: ``for rung: for seed: first accepted wins``; rungs whose structure-only row lower
    bound cannot fit the acceptance-band ceiling are dropped first. Uses labels, group sizes and (when
    ``raw`` is given) feature constancy only; never model output. Every attempt is recorded.
    """
    split_cfg, dataset = cfg["split"], cfg["dataset"]
    n_classes = dataset["classes"]
    names = dataset["class_names"]
    y, groups = np.asarray(y), np.asarray(groups)
    if y.shape != groups.shape:
        raise DataIntegrityError("labels and group ids differ in length")
    uniq, inverse, counts = _group_table(y, groups, n_classes)
    if (counts.sum(0) == 0).any():
        raise DataIntegrityError("A class has no rows; the task would change")
    sizes = counts.sum(1)
    risk, at_risk = _variability_risk(raw, groups, split_cfg)
    v2 = split_cfg["algorithm_version"] == 2
    attempts, last, bounds, plan = [], None, [], [(None, None)]
    presence = G_c = gd = None
    if v2:
        gd = split_cfg["group_diversity"]
        presence = (counts > 0).astype(np.int64)
        G_c = presence.sum(0)
        plan, last_bound = [], None
        for rung in gd["ladder"]:
            floors = _group_floors(G_c, rung, gd)
            rows, bound = _rung_lower_bound(counts, floors, split_cfg)
            bounds.append({"rung": rung["rung"], "floor_fraction": rung["floor_fraction"],
                           "floors": {p: floors[i].tolist() for i, p in enumerate(PARTITIONS)}, **bound})
            last_bound = (rung["rung"], rows, bound)
            if not bound["dropped"]:
                plan.append((rung, floors))
        if not plan:
            r, rows, bound = last_bound
            p = bound["exceeds_ceiling_in"][0]
            c = int(rows[PARTITIONS.index(p)].argmax())
            raise SplitInfeasibleError(
                f"Every rung's structure-only row lower bound exceeds a band ceiling. Binding class at "
                f"rung {r}: {names[c]} (class-{c} lower bound {int(rows[PARTITIONS.index(p), c])} rows in "
                f"{p}; partition bound fraction {bound['min_fraction'][p]:.4f} > "
                f"{bound['band_ceiling'][p]}). Rung table: {bounds}. Band and rungs were not changed.",
                attempts, bounds)
    for rung, floors in plan:
        div = None if rung is None else {"G": G_c.astype(np.float64), "floors": floors,
                                         "weight": gd["weight_group_share"]}
        for seed in split_cfg["seeds"]:
            if v2:
                run = _search(counts, sizes, uniq, seed, split_cfg, risk, presence, div)
            else:
                run = _search(counts, sizes, uniq, seed, split_cfg, risk)
            N = run["N"]
            v_div, below = 0, []
            if div is not None:
                v_div = int(np.maximum(0, floors - run["G"]).sum())
                below = [names[c] for c in np.flatnonzero((run["G"] < floors).any(0))]
            reason = _acceptance_reason(N, run["V"], split_cfg, v_div, below,
                                        None if rung is None else rung["rung"])
            constant = []
            if reason is None and raw is not None:
                # Feature-only structural check (no labels, no model): a feature constant on the
                # training rows cannot be MinMax-scaled, so such a seed is rejected like a band miss.
                train_rows = raw[run["assign"][inverse] == 0]
                constant = np.flatnonzero(train_rows.max(0) == train_rows.min(0)).tolist()
                if constant:
                    reason = f"constant training feature(s) {constant}"
            attempt = {
                "seed": int(seed), "V": run["V"], "J": run["J"], "start_V": run["start_V"],
                "start_J": run["start_J"], "proposals_used": run["proposals_used"],
                "fractions": {p: float(N[i].sum() / N.sum()) for i, p in enumerate(PARTITIONS)},
                "rows": {p: int(N[i].sum()) for i, p in enumerate(PARTITIONS)},
                "min_class_rows": {p: int(N[i].min()) for i, p in enumerate(PARTITIONS)},
                "constant_training_features": constant, "at_risk_features": at_risk,
                "accepted": reason is None, "reason": reason}
            if div is not None:
                attempt = {"rung": rung["rung"], **attempt,
                           "floors": {p: floors[i].tolist() for i, p in enumerate(PARTITIONS)},
                           "g_by_partition": {p: run["G"][i].tolist() for i, p in enumerate(PARTITIONS)},
                           "V_div": v_div, "classes_below_floor": below}
            attempts.append(attempt)
            log.info("rung %s seed %s: V=%s fractions=%s accepted=%s",
                     None if rung is None else rung["rung"], seed, run["V"],
                     attempt["fractions"], reason is None)
            last = (run, floors)
            if reason is None:
                row_part = run["assign"][inverse]
                ids = {p: np.flatnonzero(row_part == i).astype(np.int64) for i, p in enumerate(PARTITIONS)}
                out = {"seed": int(seed), "attempts": attempts, "ids": ids, "at_risk_features": at_risk}
                if v2:
                    out.update(rung=rung["rung"], floors=floors, rung_lower_bounds=bounds)
                return out
    run, floors = last
    binding = _binding_class(run["N"], counts, split_cfg, names)
    if v2:
        binding = {"hard_minima": binding, "group_floors": _binding_div(run["G"], floors, names),
                   "last_rung": attempts[-1]["rung"]}
    raise SplitInfeasibleError(
        f"No (rung, seed) in seeds {split_cfg['seeds']} met the hard minima and acceptance band. "
        f"Binding class: {binding}. Last reason: {attempts[-1]['reason']}. "
        "Band and minima were not relaxed; a documented protocol revision is required.",
        attempts, bounds or None)


# ----------------------------------------------------------------- invariants
def id_hash(ids):
    return digest([int(i) for i in sorted(ids)])


def check_partitions(ids, y, groups, n_rows, n_classes):
    """Disjoint cover of all rows, no shared group, every class in every partition."""
    seen = np.concatenate([np.asarray(ids[p], dtype=np.int64) for p in PARTITIONS])
    if len(seen) != n_rows or len(np.unique(seen)) != n_rows or seen.min() < 0 or seen.max() >= n_rows:
        raise DataIntegrityError("Partitions are not a disjoint cover of all row IDs")
    group_sets = {p: set(groups[np.asarray(ids[p], dtype=np.int64)].tolist()) for p in PARTITIONS}
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            shared = group_sets[a] & group_sets[b]
            if shared:
                raise DataIntegrityError(f"{len(shared)} identical feature vectors cross {a}/{b}")
    for p in PARTITIONS:
        if set(np.unique(y[np.asarray(ids[p], dtype=np.int64)]).tolist()) != set(range(n_classes)):
            raise DataIntegrityError(f"Partition {p} does not contain every class")


def fit_scaler(raw, train_ids, scaler_cfg):
    """Fit on the training rows only. Both options are pinned explicitly."""
    if scaler_cfg["feature_range"] != [0, 1] or scaler_cfg["clip"] is not False:
        raise ConfigError("scaler must be feature_range (0, 1) with clip False")
    scaler = MinMaxScaler(feature_range=(0, 1), clip=False).fit(raw[np.asarray(train_ids)])
    constant = np.flatnonzero(scaler.data_max_ == scaler.data_min_)
    if len(constant):
        raise DataIntegrityError(f"Constant training features {constant.tolist()}")
    return scaler


def scaler_document(scaler, feature_names, train_ids):
    return {
        "schema_version": SCHEMA_VERSION, "kind": "sklearn.preprocessing.MinMaxScaler",
        "feature_range": [0, 1], "clip": False, "fitted_on": "training rows only",
        "feature_names": list(feature_names), "n_samples_seen": int(scaler.n_samples_seen_),
        "train_ids_sha256": id_hash(train_ids),
        "data_min": scaler.data_min_.tolist(), "data_max": scaler.data_max_.tolist(),
        "data_range": scaler.data_range_.tolist(), "scale": scaler.scale_.tolist(),
        "min": scaler.min_.tolist(),
    }


def scaler_from_document(doc):
    """Rebuild the fitted sklearn scaler from its saved document; no refit."""
    if doc.get("clip") is not False or doc.get("feature_range") != [0, 1]:
        raise DataIntegrityError("Saved scaler is not MinMaxScaler(feature_range=(0, 1), clip=False)")
    scaler = MinMaxScaler(feature_range=(0, 1), clip=False)
    scaler.data_min_ = np.array(doc["data_min"], dtype=np.float64)
    scaler.data_max_ = np.array(doc["data_max"], dtype=np.float64)
    scaler.data_range_ = np.array(doc["data_range"], dtype=np.float64)
    scaler.scale_ = np.array(doc["scale"], dtype=np.float64)
    scaler.min_ = np.array(doc["min"], dtype=np.float64)
    scaler.n_samples_seen_ = int(doc["n_samples_seen"])
    scaler.n_features_in_ = len(scaler.data_min_)
    return scaler


def _verify_scaler(doc, raw, train_ids, expected_train_hash):
    if doc.get("train_ids_sha256") != expected_train_hash or doc["train_ids_sha256"] != id_hash(train_ids):
        raise DataIntegrityError("Scaler was not fitted on the frozen training IDs")
    rows = raw[np.asarray(train_ids)]
    data_min, data_max = rows.min(0), rows.max(0)
    if not (np.array_equal(np.array(doc["data_min"]), data_min)
            and np.array_equal(np.array(doc["data_max"]), data_max)):
        raise DataIntegrityError("Scaler min/max differ from the training rows")
    if (data_max == data_min).any():
        raise DataIntegrityError("Constant training feature")
    scale = 1.0 / (data_max - data_min)
    if not (np.array_equal(np.array(doc["scale"]), scale)
            and np.array_equal(np.array(doc["min"]), 0 - data_min * scale)):
        raise DataIntegrityError("Scaler scale/min are inconsistent with its min/max")
    if int(doc["n_samples_seen"]) != len(train_ids):
        raise DataIntegrityError("Scaler sample count differs from the training partition")


# --------------------------------------------------------------- split document
def _per_class_counts(ids, y, n_classes):
    return {p: np.bincount(y[np.asarray(ids[p], dtype=np.int64)], minlength=n_classes).tolist()
            for p in PARTITIONS}


def _group_counts(ids, groups):
    return {p: int(len(set(groups[np.asarray(ids[p], dtype=np.int64)].tolist()))) for p in PARTITIONS}


def split_document(cfg, result, y, groups, source, config_sha256):
    ids, n_classes = result["ids"], cfg["dataset"]["classes"]
    n_rows = len(y)
    body = {
        "schema_version": SCHEMA_VERSION, "claim": FRAMING,
        "dataset": copy.deepcopy(cfg["dataset"]),
        "split_seed": result["seed"], "seeds_attempted": list(cfg["split"]["seeds"]),
        "algorithm": {"name": ALGORITHM, "version": cfg["split"]["algorithm_version"],
                      "proposals": cfg["split"]["proposals"], "patience": cfg["split"]["patience"]},
        "targets": cfg["split"]["targets"], "hard_minima": cfg["split"]["hard_minima"],
        "feature_variability": {
            **cfg["split"]["feature_variability"],
            "at_risk_features": [cfg["dataset"]["feature_names"][f] for f in result["at_risk_features"]]},
        "acceptance_band": cfg["split"]["acceptance_band"], "grouping": cfg["grouping"],
        "scaler": cfg["scaler"], "config_sha256": config_sha256,
        "source_digest": source.get("source_digest"), "git_revision": source.get("revision"),
        "partition_rows": {p: len(ids[p]) for p in PARTITIONS},
        "fractions": {p: len(ids[p]) / n_rows for p in PARTITIONS},
        "group_counts": _group_counts(ids, groups),
        "per_class_counts": _per_class_counts(ids, y, n_classes),
        "id_sha256": {p: id_hash(ids[p]) for p in PARTITIONS},
        "row_ids": {p: [int(i) for i in sorted(ids[p])] for p in PARTITIONS},
    }
    if cfg["split"]["algorithm_version"] == 2:
        names = cfg["dataset"]["class_names"]
        floors = result["floors"]
        body["group_diversity"] = {
            **copy.deepcopy(cfg["split"]["group_diversity"]), "chosen_rung": result["rung"],
            "chosen_floors": {p: {names[c]: int(floors[i][c]) for c in range(len(names))}
                              for i, p in enumerate(PARTITIONS)}}
    return {**body, "manifest_sha256": digest(body)}


def verify_split_document(split):
    """Recompute the manifest hash and every partition hash; raise DataIntegrityError on tamper."""
    if not isinstance(split, dict) or split.get("schema_version") != SCHEMA_VERSION:
        raise DataIntegrityError("Unsupported or malformed split manifest")
    body = {k: v for k, v in split.items() if k != "manifest_sha256"}
    if digest(body) != split.get("manifest_sha256"):
        raise DataIntegrityError("Split manifest hash mismatch (file was modified)")
    n_rows = split["dataset"]["rows"]
    all_ids = []
    for p in PARTITIONS:
        ids = split["row_ids"][p]
        if not all(_is_int(i) for i in ids) or len(set(ids)) != len(ids):
            raise DataIntegrityError(f"Partition {p} IDs are not unique integers")
        if id_hash(ids) != split["id_sha256"][p]:
            raise DataIntegrityError(f"Partition {p} ID hash mismatch")
        all_ids.extend(ids)
    if len(all_ids) != n_rows or set(all_ids) != set(range(n_rows)):
        raise DataIntegrityError("Partitions do not cover each row exactly once")
    return split


# --------------------------------------------------------------------- loading
def _load_json(value):
    if isinstance(value, (str, os.PathLike)):
        return json.loads(Path(value).read_text(encoding="utf-8"))
    return value


def _check_final_test_freeze(freeze, split) -> None:
    """purpose='final_test' needs a verified EvalFreeze whose file is unchanged and whose test IDs match the split."""
    if not isinstance(freeze, FreezeBase):
        raise PartitionGuardError("purpose='final_test' needs a verified evaluation freeze (freeze=...)")
    path = Path(freeze.path)
    if not path.is_file() or sha256(path) != freeze.sha256:
        raise PartitionGuardError("the evaluation freeze file changed after it was verified")
    frozen = freeze.data.get("test_partition_id_sha256")
    if frozen is not None and frozen != _load_json(split)["id_sha256"]["test"]:
        raise PartitionGuardError("the freeze names a different test partition than this split")


def load_arrays(split, scaler, partitions=("train", "validation"), *, purpose=None, table=None, freeze=None):
    """Return verified, scaled arrays ``{partition: {"ids", "x", "y"}}``.

    ``split`` and ``scaler`` are saved documents or their paths. ``test`` is only available with
    ``purpose="range_audit"`` (callers of that route take no model) or ``purpose="final_test"`` together with a
    verified evaluation freeze (``freeze=``, created only by research.m2b_state.verify_eval_freeze).
    ``table`` = (raw, y) replaces the CSV (synthetic tests); otherwise the CSV hash is checked.
    """
    partitions = tuple(partitions)
    if purpose not in (None, "range_audit", "final_test"):
        raise PartitionGuardError(f"Unknown purpose {purpose!r}")
    unknown = set(partitions) - set(PARTITIONS)
    if unknown:
        raise PartitionGuardError(f"Unknown partitions {sorted(unknown)}")
    if purpose == "final_test":
        _check_final_test_freeze(freeze, split)
    elif freeze is not None:
        raise PartitionGuardError("freeze= is only accepted with purpose='final_test'")
    if "test" in partitions and purpose not in ("range_audit", "final_test"):
        raise PartitionGuardError("The test partition is available only with purpose='range_audit' or a verified final_test freeze")
    split, scaler_doc = _load_json(split), _load_json(scaler)
    verify_split_document(split)
    dataset = split["dataset"]
    raw, y = table if table is not None else read_table(dataset)
    _check_table(raw, y, dataset["classes"])
    if raw.shape[0] != dataset["rows"]:
        raise DataIntegrityError("Table row count differs from the split manifest")
    ids = {p: np.array(split["row_ids"][p], dtype=np.int64) for p in PARTITIONS}
    groups = group_hashes(raw)
    check_partitions(ids, y, groups, dataset["rows"], dataset["classes"])
    if _per_class_counts(ids, y, dataset["classes"]) != split["per_class_counts"]:
        raise DataIntegrityError("Per-class counts differ from the split manifest")
    _verify_scaler(scaler_doc, raw, ids["train"], split["id_sha256"]["train"])
    transform = scaler_from_document(scaler_doc)
    out = {}
    for p in partitions:
        x = transform.transform(raw[ids[p]]).astype(np.float32)
        out[p] = {"ids": ids[p].copy(), "x": x, "y": y[ids[p]].copy()}
    return out


# ---------------------------------------------------------------------- audits
def range_audit(arrays, epsilon_grid):
    """Features-only audit: values outside [0, 1] and the E2 feasible set under the grid."""
    result = {}
    for p, a in arrays.items():
        x = a["x"]
        outside = (x < 0) | (x > 1)
        rows = np.flatnonzero(outside.any(1))
        entry = {
            "rows": int(len(x)), "rows_with_value_outside_unit_range": int(len(rows)),
            "coordinates_outside_unit_range": int(outside.sum()),
            "row_ids_outside_unit_range": [int(a["ids"][i]) for i in rows],
            "max_excursion_above_1": float(max(0.0, float(x.max()) - 1.0)),
            "max_excursion_below_0": float(max(0.0, -float(x.min()))),
            "per_epsilon": {},
        }
        for eps in epsilon_grid:
            e = np.float32(eps)
            lower = np.maximum(x - e, np.minimum(np.float32(0), x))
            upper = np.minimum(x + e, np.maximum(np.float32(1), x))
            entry["per_epsilon"][repr(float(eps))] = {
                "empty_intervals": int((lower > upper).sum()),
                "one_sided_out_of_range_coordinates": int(outside.sum()),
            }
        result[p] = entry
    return {"schema_version": SCHEMA_VERSION, "claim": FRAMING,
            "policy": "clean inputs are never clipped, dropped or skipped; the feasible interval is "
                      "[max(x-eps, min(0, x)), min(x+eps, max(1, x))] and always contains x",
            "epsilon_grid": [float(v) for v in epsilon_grid], "partitions": result,
            "feature_space_only": "features and labels only; no model was constructed or called"}


def group_manifest(y, groups, ids, class_names):
    part_of = np.empty(len(y), dtype=np.int64)
    for i, p in enumerate(PARTITIONS):
        part_of[np.asarray(ids[p], dtype=np.int64)] = i
    uniq, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    counts = np.zeros((len(uniq), len(class_names)), dtype=np.int64)
    np.add.at(counts, (inverse, y), 1)
    table = {}
    for k, g in enumerate(uniq):
        table[str(g)] = {"partition": PARTITIONS[part_of[first[k]]], "rows": int(counts[k].sum()),
                         "label_counts": {class_names[c]: int(n) for c, n in enumerate(counts[k]) if n}}
    return {"schema_version": SCHEMA_VERSION, "groups": table}


def conflict_document(manifest, ids, class_names):
    groups = manifest["groups"]
    conflicts = [{"group": g, "partition": v["partition"], "rows": v["rows"],
                  "label_counts": v["label_counts"]}
                 for g, v in groups.items() if len(v["label_counts"]) > 1]
    per_partition = {}
    for p in PARTITIONS:
        members = [v for v in groups.values() if v["partition"] == p]
        n_rows = sum(v["rows"] for v in members)
        majority = sum(max(v["label_counts"].values()) for v in members)
        in_conflict = [v for v in members if len(v["label_counts"]) > 1]
        per_partition[p] = {
            "rows": n_rows, "groups": len(members), "conflict_groups": len(in_conflict),
            "conflict_rows": sum(v["rows"] for v in in_conflict),
            "unavoidable_error_rows": n_rows - majority,
            "bayes_style_accuracy_ceiling": majority / n_rows}
    return {"schema_version": SCHEMA_VERSION, "claim": FRAMING,
            "policy": "reported only; labels are neither changed nor removed",
            "conflict_groups": len(conflicts), "conflict_rows": sum(c["rows"] for c in conflicts),
            "per_partition": per_partition,
            "ceiling_note": "Rows of a group beyond its majority label are unavoidable errors for any "
                            "classifier that sees only the features.",
            "groups": sorted(conflicts, key=lambda c: c["group"])}


def _per_class_group_counts(ids, y, groups, n_classes):
    out = {}
    for p in PARTITIONS:
        rows = np.asarray(ids[p], dtype=np.int64)
        out[p] = [len(set(groups[rows[y[rows] == c]].tolist())) for c in range(n_classes)]
    return out


def report_markdown(cfg, split, conflicts, audit, attempts, groups_per_class, bounds=None):
    names = cfg["dataset"]["class_names"]
    v2 = split["algorithm"]["version"] == 2
    weak_groups = V2_WEAK_GROUPS if v2 else WEAK_GROUPS
    n = sum(split["partition_rows"].values())
    lines = ["# Milestone 2B split report (split revision 2)" if v2 else "# Milestone 2A split report", "",
             FRAMING, "",
             "Generated by `research.m2_split`. Labels and group sizes only; no model output was used "
             "and no model was run on the test partition.", "",
             "## Frozen split", "",
             f"- Split seed: {split['split_seed']} (listed seeds: {split['seeds_attempted']})",
             f"- Dataset SHA-256: `{cfg['dataset']['sha256']}` ({n} rows)",
             f"- Manifest SHA-256: `{split['manifest_sha256']}`",
             f"- Config SHA-256: `{split['config_sha256']}`", "",
             "| Partition | Rows | Fraction | Target | Distinct feature vectors | ID SHA-256 |",
             "|---|---|---|---|---|---|"]
    for p in PARTITIONS:
        lines.append(f"| {p} | {split['partition_rows'][p]} | {split['fractions'][p]:.4f} | "
                     f"{split['targets'][p]:.2f} | {split['group_counts'][p]} | `{split['id_sha256'][p]}` |")
    fv = split["feature_variability"]
    lines += ["", f"Feature variability (protocol revision 1): every feature must take at least "
              f"{fv['min_distinct_train_values']} distinct values among the training rows. Features whose "
              f"modal value is shared by all but at most {fv['at_risk_max_minority_groups']} groups were "
              "pinned in the split objective (one unit of V each while train lacks a differing group): "
              + (", ".join(fv["at_risk_features"]) or "none") + "."]
    lines += ["", "The achieved fractions differ from 70/15/15 because a few large groups of identical "
              "feature vectors must stay in one partition. Acceptance band: " +
              ", ".join(f"{p} {split['acceptance_band'][p]}" for p in PARTITIONS) +
              "; hard minima (rows per class): " +
              ", ".join(f"{p} {split['hard_minima'][p]}" for p in PARTITIONS) + ".", "",
              "## Seed attempts", "", "| Seed | V | Fractions (train/val/test) | Accepted | Reason |",
              "|---|---|---|---|---|"]
    for a in attempts:
        f = a["fractions"]
        lines.append(f"| {a['seed']} | {a['V']} | {f['train']:.4f} / {f['validation']:.4f} / "
                     f"{f['test']:.4f} | {a['accepted']} | {a['reason'] or ''} |")
    lines += ["", "## Per-class rows and independent groups", "",
              "| Class | Name | Rows tr/va/te | Groups tr/va/te | Flag |", "|---|---|---|---|---|"]
    weak = []
    for c, name in enumerate(names):
        rows = [split["per_class_counts"][p][c] for p in PARTITIONS]
        grp = [groups_per_class[p][c] for p in PARTITIONS]
        flag = ""
        if grp[1] < weak_groups or grp[2] < weak_groups:
            flag = "weak per-class evidence"
            weak.append(name)
        lines.append(f"| {c} | {name} | {rows[0]} / {rows[1]} / {rows[2]} | "
                     f"{grp[0]} / {grp[1]} / {grp[2]} | {flag} |")
    lines += ["", f"Classes with fewer than {weak_groups} groups in validation or test behave like one or two "
              "independent vectors there: " + (", ".join(weak) if weak else "none") + ". "
              "The flag is informational and does not change the split.", "",
              "## Conflicting labels among identical feature vectors", "",
              f"{conflicts['conflict_groups']} groups ({conflicts['conflict_rows']} rows) contain more than one "
              "label. They are reported only; labels are not changed and rows are not removed.", "",
              "| Partition | Rows | Conflict groups | Conflict rows | Unavoidable error rows | Accuracy ceiling |",
              "|---|---|---|---|---|---|"]
    for p in PARTITIONS:
        c = conflicts["per_partition"][p]
        lines.append(f"| {p} | {c['rows']} | {c['conflict_groups']} | {c['conflict_rows']} | "
                     f"{c['unavoidable_error_rows']} | {c['bayes_style_accuracy_ceiling']:.4f} |")
    lines += ["", "## Scaler and input range", "",
              "`MinMaxScaler(feature_range=(0, 1), clip=False)` fitted on the training rows only. Held-out "
              "features may fall outside [0, 1]; they are never clipped.", "",
              "| Partition | Rows outside [0,1] | Coordinates outside | Max above 1 | Max below 0 |",
              "|---|---|---|---|---|"]
    for p in PARTITIONS:
        a = audit["partitions"][p]
        lines.append(f"| {p} | {a['rows_with_value_outside_unit_range']} | {a['coordinates_outside_unit_range']} | "
                     f"{a['max_excursion_above_1']:.4f} | {a['max_excursion_below_0']:.4f} |")
    lines += ["", "No feasible interval is empty at any epsilon in the grid "
              f"{audit['epsilon_grid']} (see `range_audit.json`).", ""]
    if v2:
        lines += _v2_section(split, bounds or [])
    return "\n".join(lines)


def _v2_section(split, bounds):
    gd = split["group_diversity"]
    lines = ["## Group-diversity criterion (algorithm version 2)", "",
             "Revision 2 changes only the allocation. The dataset, grouping rule, scaler rule, acceptance band, "
             "hard row minima and feature-variability pins are those of revision 1. Identical feature vectors "
             "stay together, including groups with conflicting labels. The rule uses labels, group sizes and "
             "feature values only; no model output chose it.", "",
             "For class c with G_c distinct feature-vector groups (a group counts toward every class among "
             "its rows' labels) the floor in partition p is "
             "`F_p(c) = min(max(absolute_floor_p, ceil(fraction_p * G_c)), floor(G_c / 3))`. "
             f"Absolute floors: {gd['absolute_floor_groups']}. The cap floor(G_c / 3) keeps each class "
             "individually satisfiable; for small G_c the train floor is the cap, not the fraction. The cap "
             "ignores groups shared by several classes, so joint satisfiability is decided by the search "
             "(V_div = 0), not assumed. The search stays lexicographic (V, J), with V = row shortfall + "
             "constant-feature risk + V_div and J = the version-1 deviation plus "
             f"{gd['weight_group_share']} * the mean squared deviation of the per-class group share from the "
             "partition targets.", "",
             f"Pre-registered ladder (order: {gd['order']}). A rung is dropped before searching iff its "
             "structure-only row lower bound (sum of the F_p(c) smallest class-c group row counts, summed "
             "over classes) exceeds a band ceiling. **Chosen rung: " + str(gd["chosen_rung"]) + "**, seed "
             + str(split["split_seed"]) + ".", "",
             "| Rung | Dropped | Lower-bound fraction tr/va/te | Band ceilings |", "|---|---|---|---|"]
    for b in bounds:
        f, c = b["min_fraction"], b["band_ceiling"]
        lines.append(f"| {b['rung']} | {b['dropped']} | {f['train']:.4f} / {f['validation']:.4f} / "
                     f"{f['test']:.4f} | {c['train']} / {c['validation']} / {c['test']} |")
    lines += ["", "## Limitations", "",
              "1. A class with a dominant feature vector still has most of its TRAIN ROWS from that single "
              "vector. The floor raises distinct vectors, not row balance. `split_comparison_v1_v2.md` "
              "reports per class the distinct vectors, the largest-group share in train and the effective "
              "vector count 1 / sum(p_g^2) over the class's train rows (reported only, not constrained).",
              "2. Validation and test hold few rows for those classes (about 20-90), and the floors count "
              f"distinct vectors, so per-class macro-F1 is noisy. Classes with fewer than {V2_WEAK_GROUPS} "
              "distinct vectors in validation or test are flagged as weak per-class evidence "
              "(revision 1 used 3).",
              "3. Identical vectors with conflicting labels stay together and cap accuracy per partition "
              "(`conflicting_labels.json`).",
              "4. Held-out coordinates outside the training range (for example the `Number` singleton rows) "
              "remain and are never clipped (`range_audit.json`).",
              "5. The dataset was seen by Milestone 1 and the old test partition was evaluated: " + FRAMING, ""]
    return lines


# --------------------------------------------------------------------- writing
def _write_new_json(path, value):
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    atomic_json(path, value)


def _write_new_text(path, text):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _refuse_existing(out):
    existing = [name for name in ARTIFACTS if (Path(out) / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite split artifacts in {out}: {existing}")


def _source_info():
    from .common import provenance
    info = provenance()
    return {"source_digest": info["source_digest"], "revision": info["revision"]}


def create_split(cfg, out_dir=None, *, table=None, source=None, config_sha256=""):
    """Build and write the seven artifacts; refuses to overwrite. Returns the split document."""
    out = Path(out_dir) if out_dir else ROOT / cfg["split"]["output_dir"]
    _refuse_existing(out)
    raw, y = table if table is not None else read_table(cfg["dataset"])
    _check_table(raw, y, cfg["dataset"]["classes"])
    groups = group_hashes(raw)
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = make_split(y, groups, cfg, raw)
    except SplitInfeasibleError as exc:
        doc = {"schema_version": SCHEMA_VERSION, "status": "infeasible", "error": str(exc),
               "attempts": exc.attempts}
        if exc.rung_lower_bounds is not None:
            doc["rung_lower_bounds"] = exc.rung_lower_bounds
        _write_new_json(out / "split_attempts.json", doc)
        raise
    ids = result["ids"]
    check_partitions(ids, y, groups, len(y), cfg["dataset"]["classes"])
    scaler = fit_scaler(raw, ids["train"], cfg["scaler"])
    scaler_doc = scaler_document(scaler, cfg["dataset"]["feature_names"], ids["train"])
    source = source or _source_info()
    split = split_document(cfg, result, y, groups, source, config_sha256)
    manifest = group_manifest(y, groups, ids, cfg["dataset"]["class_names"])
    conflicts = conflict_document(manifest, ids, cfg["dataset"]["class_names"])
    # Everything is read back through the guarded loader before the commit marker is written.
    accepted = {"schema_version": SCHEMA_VERSION, "status": "accepted", "frozen_seed": result["seed"],
                "attempts": result["attempts"]}
    if "rung_lower_bounds" in result:
        accepted["frozen_rung"] = result["rung"]
        accepted["rung_lower_bounds"] = result["rung_lower_bounds"]
    _write_new_json(out / "split_attempts.json", accepted)
    _write_new_json(out / "group_manifest.json", manifest)
    _write_new_json(out / "conflicting_labels.json", conflicts)
    _write_new_json(out / "scaler_minmax.json", scaler_doc)
    arrays = load_arrays(split, scaler_doc, PARTITIONS, purpose="range_audit", table=(raw, y))
    audit = range_audit(arrays, cfg["range_audit"]["epsilon_grid"])
    _write_new_json(out / "range_audit.json", audit)
    per_class_groups = _per_class_group_counts(ids, y, groups, cfg["dataset"]["classes"])
    _write_new_text(out / "split_report.md", report_markdown(
        cfg, split, conflicts, audit, result["attempts"], per_class_groups,
        result.get("rung_lower_bounds")))
    _write_new_json(out / "split_m2.json", split)
    log.info("wrote split artifacts to %s", out)
    return split


def verify_freeze(out, split, config_path=None):
    """Recompute every hash recorded in ``split_freeze.json`` (written by ``m2b_splitreport``)."""
    freeze = json.loads((Path(out) / "split_freeze.json").read_text(encoding="utf-8"))
    expected = {"dataset_sha256": split["dataset"]["sha256"], "manifest_sha256": split["manifest_sha256"],
                "id_sha256": split["id_sha256"], "split_seed": split["split_seed"],
                "config_sha256": split["config_sha256"],
                "rung": split.get("group_diversity", {}).get("chosen_rung")}
    for key, value in expected.items():
        if freeze.get(key) != value:
            raise DataIntegrityError(f"split_freeze.json {key} differs from the split manifest")
    if config_path is not None and sha256(config_path) != freeze["config_sha256"]:
        raise DataIntegrityError("Config file SHA-256 differs from split_freeze.json")
    names = freeze["artifact_sha256"]
    missing = [n for n in ARTIFACTS if n not in names]
    if missing:
        raise DataIntegrityError(f"split_freeze.json does not list {missing}")
    for name, want in names.items():
        path = Path(out) / name
        if not path.is_file() or sha256(path) != want:
            raise DataIntegrityError(f"{name} differs from the SHA-256 recorded in split_freeze.json")
    return {"artifacts_verified": len(names)}


def audit_existing(cfg, out_dir=None, *, table=None, config_path=None):
    """Read-only integrity, coverage and range audit of the saved split. Writes nothing."""
    out = Path(out_dir) if out_dir else ROOT / cfg["split"]["output_dir"]
    missing = [name for name in ARTIFACTS if not (out / name).is_file()]
    if missing:
        raise DataIntegrityError(f"Missing split artifacts in {out}: {missing}")
    split = json.loads((out / "split_m2.json").read_text(encoding="utf-8"))
    scaler_doc = json.loads((out / "scaler_minmax.json").read_text(encoding="utf-8"))
    arrays = load_arrays(split, scaler_doc, PARTITIONS, purpose="range_audit", table=table)
    raw, y = table if table is not None else read_table(split["dataset"])
    groups = group_hashes(raw)
    ids = {p: arrays[p]["ids"] for p in PARTITIONS}
    names = split["dataset"]["class_names"]
    recomputed = {
        "range_audit.json": range_audit(arrays, cfg["range_audit"]["epsilon_grid"]),
        "group_manifest.json": group_manifest(y, groups, ids, names)}
    recomputed["conflicting_labels.json"] = conflict_document(recomputed["group_manifest.json"], ids, names)
    for name, value in recomputed.items():
        saved = json.loads((out / name).read_text(encoding="utf-8"))
        if saved != json.loads(json.dumps(value)):
            raise DataIntegrityError(f"{name} differs from a fresh recomputation")
    for p in PARTITIONS:
        if set(np.unique(arrays[p]["y"]).tolist()) != set(range(split["dataset"]["classes"])):
            raise DataIntegrityError(f"{p} lacks a class")
    freeze = None
    if (out / "split_freeze.json").is_file():
        freeze = verify_freeze(out, split, config_path)
    return {"manifest_sha256": split["manifest_sha256"], "split_seed": split["split_seed"],
            "freeze": freeze,
            "partition_rows": split["partition_rows"], "fractions": split["fractions"],
            "classes_in_every_partition": True, "shared_groups_between_partitions": 0,
            "rows_outside_unit_range": {
                p: recomputed["range_audit.json"]["partitions"][p]["rows_with_value_outside_unit_range"]
                for p in PARTITIONS}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true",
                        help="read-only integrity, coverage and range audit of the saved split")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    cfg = load_config(args.config)
    if args.audit_only:
        print(json.dumps(audit_existing(cfg, config_path=args.config), indent=2))
        return 0
    split = create_split(cfg, config_sha256=sha256(args.config))
    print(json.dumps({k: split[k] for k in ("split_seed", "manifest_sha256", "partition_rows",
                                            "fractions", "group_counts", "id_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
