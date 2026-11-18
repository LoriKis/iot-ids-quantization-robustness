"""Split revision 2 (group-diversity floors): synthetic ~600-row table, no model, no real data."""
import copy
import json

import numpy as np
import pytest

from m2_helpers import SOURCE, real_config, synthetic_config
from research import m2_split as m2
from research import m2b_splitreport as rep
from research.common import ROOT, sha256

PARTS = m2.PARTITIONS
DATA_V2 = ROOT / "configs/m2b/data_v2.json"
N_CLASSES = 6


def _table(seed=0):
    """Six classes, each with one dominant vector (40 rows) and about 60 rows of small groups."""
    rng = np.random.default_rng(seed)
    raw, labels = [], []
    for c in range(N_CLASSES):
        vector = rng.random(3)
        raw.extend([vector] * 40)
        labels.extend([c] * 40)
        rows = 0
        while rows < 60:
            size = int(rng.integers(1, 3))
            raw.extend([rng.random(3)] * size)
            labels.extend([c] * size)
            rows += size
    order = rng.permutation(len(raw))
    return np.array(raw, dtype=np.float64)[order], np.array(labels, dtype=np.int64)[order]


def _v2_config(raw, ladder=None, minima=(10, 3, 3)):
    cfg = synthetic_config(n_classes=N_CLASSES, minima=minima)
    cfg["dataset"]["rows"] = len(raw)
    cfg["split"]["algorithm_version"] = 2
    cfg["split"]["group_diversity"] = json.loads(DATA_V2.read_text(encoding="utf-8"))["split"]["group_diversity"]
    if ladder is not None:
        cfg["split"]["group_diversity"]["ladder"] = ladder
    return cfg


@pytest.fixture(scope="module")
def table():
    return _table()


@pytest.fixture(scope="module")
def groups(table):
    return m2.group_hashes(table[0])


@pytest.fixture(scope="module")
def cfg(table):
    return _v2_config(table[0])


@pytest.fixture(scope="module")
def result(table, cfg, groups):
    return m2.make_split(table[1], groups, cfg, table[0])


def test_real_v2_config_is_valid_and_leaves_v1_config_alone():
    loaded = m2.load_config(DATA_V2)
    assert loaded["split"]["algorithm_version"] == 2
    assert loaded["split"]["seeds"] == [20270, 20271, 20272]
    assert [r["rung"] for r in loaded["split"]["group_diversity"]["ladder"]] == [1, 2, 3, 4, 5]
    assert loaded["split"]["output_dir"] == "reports/milestone2b/split_v2"
    assert real_config()["split"]["algorithm_version"] == 1 and "group_diversity" not in real_config()["split"]
    assert loaded["dataset"] == real_config()["dataset"]


def test_config_version_rules(cfg):
    m2.validate_config(copy.deepcopy(cfg))
    missing = copy.deepcopy(cfg)
    del missing["split"]["group_diversity"]
    with pytest.raises(m2.ConfigError):
        m2.validate_config(missing)
    v1_with_block = copy.deepcopy(cfg)
    v1_with_block["split"]["algorithm_version"] = 1
    with pytest.raises(m2.ConfigError):
        m2.validate_config(v1_with_block)
    extra = copy.deepcopy(cfg)
    extra["split"]["group_diversity"]["surprise"] = 1
    with pytest.raises(m2.ConfigError):
        m2.validate_config(extra)
    reordered = copy.deepcopy(cfg)
    reordered["split"]["group_diversity"]["ladder"].reverse()
    with pytest.raises(m2.ConfigError):
        m2.validate_config(reordered)


def test_zero_group_overlap_cover_and_all_classes(table, groups, result, cfg):
    raw, y = table
    ids = result["ids"]
    m2.check_partitions(ids, y, groups, len(raw), N_CLASSES)
    assert np.array_equal(np.sort(np.concatenate([ids[p] for p in PARTS])), np.arange(len(raw)))
    sets = {p: set(groups[ids[p]]) for p in PARTS}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                or sets["validation"] & sets["test"])
    for p in PARTS:
        assert set(np.unique(y[ids[p]]).tolist()) == set(range(N_CLASSES))


def test_floors_met_and_formula(table, groups, result):
    _, y = table
    ids, floors = result["ids"], result["floors"]
    G = np.array([len(set(groups[y == c])) for c in range(N_CLASSES)])
    gd = _v2_config(table[0])["split"]["group_diversity"]
    rung = gd["ladder"][result["rung"] - 1]
    for i, p in enumerate(PARTS):
        expected = np.minimum(np.maximum(gd["absolute_floor_groups"][p], np.ceil(rung["floor_fraction"][p] * G)),
                              G // 3)
        assert np.array_equal(floors[i], expected)
        for c in range(N_CLASSES):
            rows = ids[p][y[ids[p]] == c]
            assert len(set(groups[rows])) >= floors[i][c]
    accepted = result["attempts"][-1]
    assert accepted["accepted"] and accepted["V_div"] == 0 and accepted["classes_below_floor"] == []
    assert accepted["rung"] == result["rung"] and set(accepted["floors"]) == set(PARTS)


def test_no_constant_train_feature_and_rows_labels_preserved(table, result, cfg):
    raw, y = table
    scaler = m2.fit_scaler(raw, result["ids"]["train"], cfg["scaler"])
    assert (scaler.data_max_ > scaler.data_min_).all()
    rebuilt = np.full(len(y), -1)
    for p in PARTS:
        rebuilt[result["ids"][p]] = y[result["ids"][p]]
    assert np.array_equal(rebuilt, y)


def test_determinism(table, groups, cfg, result):
    again = m2.make_split(table[1], groups, cfg, table[0])
    assert again["seed"] == result["seed"] and again["rung"] == result["rung"]
    for p in PARTS:
        assert np.array_equal(again["ids"][p], result["ids"][p])


def test_group_floors_raise_distinct_train_vectors_over_row_only_split(table, groups, result):
    _, y = table
    v1_cfg = synthetic_config(n_classes=N_CLASSES)
    v1_cfg["dataset"]["rows"] = len(y)
    v1 = m2.make_split(y, groups, v1_cfg, table[0])
    for c in range(N_CLASSES):
        old = len(set(groups[v1["ids"]["train"][y[v1["ids"]["train"]] == c]]))
        new = len(set(groups[result["ids"]["train"][y[result["ids"]["train"]] == c]]))
        assert new >= result["floors"][0][c]
        assert new >= 1 and old >= 1


def test_v1_path_unchanged(table, groups, tmp_path):
    raw, y = table
    cfg = synthetic_config(n_classes=N_CLASSES)
    cfg["dataset"]["rows"] = len(raw)
    result = m2.make_split(y, groups, cfg, raw)
    assert "rung" not in result and all("rung" not in a for a in result["attempts"])
    out = tmp_path / "v1"
    split = m2.create_split(cfg, out, table=table, source=SOURCE)
    assert split["algorithm"]["version"] == 1 and "group_diversity" not in split
    assert "rung_lower_bounds" not in json.loads((out / "split_attempts.json").read_text(encoding="utf-8"))


def test_ladder_falls_to_rung_two(table, groups, cfg, monkeypatch):
    real = m2._search
    calls = []

    def reject_rung_one(counts, sizes, uniq, seed, split_cfg, risk=None, presence=None, div=None):
        run = real(counts, sizes, uniq, seed, split_cfg, risk, presence, div)
        calls.append(seed)
        if len(calls) <= len(cfg["split"]["seeds"]):
            run["V"] = 5
        return run

    monkeypatch.setattr(m2, "_search", reject_rung_one)
    result = m2.make_split(table[1], groups, cfg, table[0])
    assert result["rung"] == 2
    assert [a["rung"] for a in result["attempts"]] == [1, 1, 1, 2]
    assert [a["accepted"] for a in result["attempts"]] == [False, False, False, True]


def test_infeasible_error_names_binding_class(table, groups):
    cfg = _v2_config(table[0], minima=(10_000, 3, 3))
    with pytest.raises(m2.SplitInfeasibleError, match="Binding class") as info:
        m2.make_split(table[1], groups, cfg, table[0])
    assert "class_name" in str(info.value)
    assert len(info.value.attempts) == len(cfg["split"]["seeds"]) * 5
    assert len(info.value.rung_lower_bounds) == 5


def test_rung_lower_bound_drops_rung_exceeding_ceiling():
    counts = np.array([[5, 0], [4, 0], [3, 1], [0, 6], [0, 7]], dtype=np.int64)
    split_cfg = {"acceptance_band": {"train": [0.6, 0.8], "validation": [0.1, 0.17], "test": [0.1, 0.17]}}
    floors = np.array([[1, 1], [2, 1], [2, 1]])
    rows, bound = m2._rung_lower_bound(counts, floors, split_cfg)
    # class 0 smallest groups: 3, 4 -> validation floor 2 = 7 rows; class 1: 1 row. Total 8 / 26 rows.
    assert rows[1].tolist() == [7, 1] and bound["min_rows"]["validation"] == 8
    assert bound["dropped"] and bound["exceeds_ceiling_in"] == ["validation", "test"]


def test_written_artifacts_record_version_two_and_freeze_roundtrip(table, cfg, tmp_path):
    cfg_path = tmp_path / "data_v2.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    out = tmp_path / "split_v2"
    split = m2.create_split(cfg, out, table=table, source=SOURCE, config_sha256=sha256(cfg_path))
    assert split["algorithm"]["version"] == 2
    assert split["group_diversity"]["chosen_rung"] in (1, 2, 3, 4, 5)
    assert split["group_diversity"]["ladder"] == cfg["split"]["group_diversity"]["ladder"]
    saved = json.loads((out / "split_m2.json").read_text(encoding="utf-8"))
    assert saved["algorithm"]["version"] == 2 and saved["group_diversity"] == split["group_diversity"]
    attempts = json.loads((out / "split_attempts.json").read_text(encoding="utf-8"))
    assert attempts["frozen_rung"] == split["group_diversity"]["chosen_rung"]
    assert len(attempts["rung_lower_bounds"]) == 5
    report = (out / "split_report.md").read_text(encoding="utf-8")
    assert "Group-diversity criterion" in report and "Limitations" in report
    freeze = rep.build_freeze(out, cfg_path)
    (out / "split_freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    summary = m2.audit_existing(cfg, out, table=table, config_path=cfg_path)
    assert summary["freeze"]["artifacts_verified"] == 7
    assert freeze["rung"] == split["group_diversity"]["chosen_rung"]
    (out / "group_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(m2.DataIntegrityError):
        m2.audit_existing(cfg, out, table=table, config_path=cfg_path)


def test_comparison_on_synthetic_table(table, cfg, tmp_path, monkeypatch):
    raw, y = table
    old_cfg = synthetic_config(n_classes=N_CLASSES)
    old_cfg["dataset"]["rows"] = len(raw)
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    m2.create_split(old_cfg, old_dir, table=table, source=SOURCE)
    m2.create_split(cfg, new_dir, table=table, source=SOURCE)
    monkeypatch.setattr(m2, "read_table", lambda dataset: (raw, y))
    cmp = rep.build_comparison(old_dir, new_dir)
    assert cmp["rows_preserved"] and cmp["labels_unchanged"] and cmp["all_classes_in_every_partition"]
    assert not any(cmp["shared_groups"]["new"].values()) and not any(cmp["shared_groups"]["old"].values())
    assert cmp["labels_sha256"]["before"] == cmp["labels_sha256"]["after"]
    assert sum(t["rows"] for t in cmp["totals"]["new"].values()) == len(y)
    assert len(cmp["classes"]) == N_CLASSES
    assert "| Class |" in rep.comparison_markdown(cmp)


@pytest.mark.needs_real_data
def test_frozen_real_split_audit_only():
    out = ROOT / "reports/milestone2b/split_v2"
    if not (out / "split_freeze.json").is_file():
        pytest.skip("split_v2 not built")
    cfg = m2.load_config(DATA_V2)
    summary = m2.audit_existing(cfg, config_path=DATA_V2)
    assert summary["shared_groups_between_partitions"] == 0 and summary["classes_in_every_partition"]
    assert summary["freeze"]["artifacts_verified"] >= 9
