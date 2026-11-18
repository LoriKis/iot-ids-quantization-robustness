import copy
import json

import numpy as np
import pytest
import torch

from m2_helpers import REAL_CONFIG, SOURCE, finish, real_config, synthetic_config, synthetic_table
from research import m2_split as m2

PARTS = m2.PARTITIONS


@pytest.fixture(scope="module")
def table():
    return synthetic_table()


@pytest.fixture(scope="module")
def cfg(table):
    return finish(synthetic_config(), table[0])


@pytest.fixture(scope="module")
def groups(table):
    return m2.group_hashes(table[0])


@pytest.fixture(scope="module")
def result(table, cfg, groups):
    return m2.make_split(table[1], groups, cfg)


def test_real_config_is_valid_and_matches_experiment_c():
    loaded = m2.load_config(REAL_CONFIG)
    assert loaded["split"]["seeds"] == [20260, 20261, 20262]
    assert loaded["split"]["hard_minima"] == {"train": 200, "validation": 20, "test": 20}
    assert loaded["dataset"]["classes"] == 27 and loaded["dataset"]["rows"] == 29997
    assert loaded["scaler"]["clip"] is False and loaded["scaler"]["feature_range"] == [0, 1]


def test_determinism(table, cfg, groups, result):
    again = m2.make_split(table[1], groups, cfg)
    assert again["seed"] == result["seed"]
    for p in PARTS:
        assert np.array_equal(again["ids"][p], result["ids"][p])
        assert m2.id_hash(again["ids"][p]) == m2.id_hash(result["ids"][p])


def test_partitions_cover_rows_and_share_no_group(table, cfg, groups, result):
    raw, y = table
    ids = result["ids"]
    assert sum(len(ids[p]) for p in PARTS) == len(raw)
    assert len(np.unique(np.concatenate([ids[p] for p in PARTS]))) == len(raw)
    sets = {p: set(groups[ids[p]]) for p in PARTS}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                or sets["validation"] & sets["test"])
    m2.check_partitions(ids, y, groups, len(raw), cfg["dataset"]["classes"])


def test_every_class_in_every_partition_and_minima_met(table, cfg, result):
    _, y = table
    minima = cfg["split"]["hard_minima"]
    for p in PARTS:
        counts = np.bincount(y[result["ids"][p]], minlength=cfg["dataset"]["classes"])
        assert counts.min() >= minima[p] > 0


def test_identical_vectors_with_different_labels_stay_together(table, groups, result):
    _, y = table
    part_of = np.empty(len(y), dtype=object)
    for p in PARTS:
        part_of[result["ids"][p]] = p
    conflicting = [g for g in np.unique(groups) if len(set(y[groups == g])) > 1]
    assert len(conflicting) >= 5
    for g in conflicting:
        assert len(set(part_of[groups == g])) == 1


def test_conflict_report_on_synthetic_table_without_relabeling():
    raw = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0, 6.0]])
    y = np.array([0, 0, 1, 2, 1, 1])
    y_before, raw_before = y.copy(), raw.copy()
    groups = m2.group_hashes(raw)
    ids = {"train": np.array([0, 1, 2]), "validation": np.array([3]), "test": np.array([4, 5])}
    manifest = m2.group_manifest(y, groups, ids, ["a", "b", "c"])
    report = m2.conflict_document(manifest, ids, ["a", "b", "c"])
    assert report["conflict_groups"] == 1 and report["conflict_rows"] == 3
    group = report["groups"][0]
    assert group["partition"] == "train" and group["label_counts"] == {"a": 2, "b": 1}
    train = report["per_partition"]["train"]
    assert train["unavoidable_error_rows"] == 1
    assert train["bayes_style_accuracy_ceiling"] == pytest.approx(2 / 3)
    assert report["per_partition"]["test"]["conflict_groups"] == 0
    assert sum(g["rows"] for g in manifest["groups"].values()) == len(y)
    assert np.array_equal(y, y_before) and np.array_equal(raw, raw_before)


def test_scaler_is_train_only_and_pinned(table, cfg, groups, result):
    raw, _ = table
    ids = result["ids"]
    scaler = m2.fit_scaler(raw, ids["train"], cfg["scaler"])
    assert scaler.feature_range == (0, 1) and scaler.clip is False
    assert np.array_equal(scaler.data_min_, raw[ids["train"]].min(0))
    perturbed = raw.copy()
    perturbed[ids["validation"][0]] = 1e6
    perturbed[ids["test"][0]] = -1e6
    other = m2.fit_scaler(perturbed, ids["train"], cfg["scaler"])
    assert np.array_equal(other.data_min_, scaler.data_min_) and np.array_equal(other.data_max_, scaler.data_max_)
    above = scaler.data_max_ + 0.5 * scaler.data_range_
    assert (scaler.transform(above[None, :]) > 1.0).all()
    with pytest.raises(m2.ConfigError):
        m2.fit_scaler(raw, ids["train"], {"feature_range": [0, 1], "clip": True})


def test_constant_training_feature_is_rejected(table, cfg, result):
    raw, _ = table
    constant = raw.copy()
    constant[:, 1] = 0.5
    with pytest.raises(m2.DataIntegrityError):
        m2.fit_scaler(constant, result["ids"]["train"], cfg["scaler"])


@pytest.fixture
def written(tmp_path, table, cfg):
    out = tmp_path / "split"
    split = m2.create_split(cfg, out, table=table, source=SOURCE, config_sha256="a" * 64)
    return out, split


def test_seven_artifacts_written_and_overwrite_refused(written, table, cfg):
    out, split = written
    assert sorted(p.name for p in out.iterdir()) == sorted(m2.ARTIFACTS)
    with pytest.raises(FileExistsError):
        m2.create_split(cfg, out, table=table, source=SOURCE)
    assert m2.FRAMING in (out / "split_report.md").read_text(encoding="utf-8")


def test_conflict_artifact_matches_independent_count(written, table):
    out, _ = written
    raw, y = table
    groups = m2.group_hashes(raw)
    expected = {g for g in np.unique(groups) if len(set(y[groups == g])) > 1}
    saved = json.loads((out / "conflicting_labels.json").read_text(encoding="utf-8"))
    assert {c["group"] for c in saved["groups"]} == expected
    assert saved["conflict_rows"] == sum(int((groups == g).sum()) for g in expected)


def test_load_arrays_roundtrip_returns_scaled_train_and_validation_only(written, table):
    out, split = written
    arrays = m2.load_arrays(out / "split_m2.json", out / "scaler_minmax.json", table=table)
    assert set(arrays) == {"train", "validation"}
    x = arrays["train"]["x"]
    assert x.dtype == np.float32 and x.min() >= 0.0 and x.max() <= 1.0
    assert np.array_equal(arrays["train"]["ids"], np.array(split["row_ids"]["train"]))


def test_tamper_detection(written, table):
    out, split = written
    scaler = json.loads((out / "scaler_minmax.json").read_text(encoding="utf-8"))

    swapped = copy.deepcopy(split)
    swapped["row_ids"]["train"][0], swapped["row_ids"]["validation"][0] = (
        swapped["row_ids"]["validation"][0], swapped["row_ids"]["train"][0])
    with pytest.raises(m2.DataIntegrityError):
        m2.load_arrays(swapped, scaler, table=table)

    renamed = copy.deepcopy(split)
    renamed["split_seed"] += 1
    with pytest.raises(m2.DataIntegrityError, match="manifest hash"):
        m2.load_arrays(renamed, scaler, table=table)

    # An attacker who also recomputes the hashes is still caught by the data itself.
    forged = copy.deepcopy(split)
    forged["row_ids"]["train"] = forged["row_ids"]["train"][1:] + [forged["row_ids"]["validation"][0]]
    forged["row_ids"]["validation"] = forged["row_ids"]["validation"][1:] + [split["row_ids"]["train"][0]]
    for p in PARTS:
        forged["id_sha256"][p] = m2.id_hash(forged["row_ids"][p])
    forged["manifest_sha256"] = m2.digest({k: v for k, v in forged.items() if k != "manifest_sha256"})
    with pytest.raises(m2.DataIntegrityError):
        m2.load_arrays(forged, scaler, table=table)

    bad_scaler = copy.deepcopy(scaler)
    bad_scaler["data_max"][0] += 1.0
    with pytest.raises(m2.DataIntegrityError):
        m2.load_arrays(split, bad_scaler, table=table)

    raw, y = table
    relabeled = y.copy()
    relabeled[0] = (relabeled[0] + 1) % 4
    with pytest.raises(m2.DataIntegrityError):
        m2.load_arrays(split, scaler, table=(raw, relabeled))

    leaked = raw.copy()
    leaked[split["row_ids"]["validation"][0]] = raw[split["row_ids"]["train"][0]]
    with pytest.raises(m2.DataIntegrityError):
        m2.load_arrays(split, scaler, table=(leaked, y))


def test_no_model_call_during_audit(tmp_path, table, cfg, monkeypatch):
    def forbidden(self, *args, **kwargs):
        raise AssertionError("a Module was called while auditing partitions")

    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    out = tmp_path / "split"
    m2.create_split(cfg, out, table=table, source=SOURCE)
    summary = m2.audit_existing(cfg, out, table=table)
    assert summary["classes_in_every_partition"] and summary["shared_groups_between_partitions"] == 0
    arrays = m2.load_arrays(out / "split_m2.json", out / "scaler_minmax.json", PARTS,
                            purpose="range_audit", table=table)
    assert set(arrays) == set(PARTS)


def test_audit_detects_modified_artifact(written, table, cfg):
    out, _ = written
    path = out / "range_audit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["partitions"]["test"]["rows"] += 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(m2.DataIntegrityError):
        m2.audit_existing(cfg, out, table=table)


def test_range_audit_reports_out_of_range_without_clipping():
    arrays = {"test": {"ids": np.array([7]), "x": np.array([[2.0, -0.3, 0.5]], dtype=np.float32)}}
    audit = m2.range_audit(arrays, [0, 0.01, 0.1])
    entry = audit["partitions"]["test"]
    assert entry["rows_with_value_outside_unit_range"] == 1 and entry["row_ids_outside_unit_range"] == [7]
    assert entry["max_excursion_above_1"] == pytest.approx(1.0)
    assert entry["max_excursion_below_0"] == pytest.approx(0.3, abs=1e-6)
    assert all(v["empty_intervals"] == 0 for v in entry["per_epsilon"].values())
    assert arrays["test"]["x"][0, 0] == np.float32(2.0)  # the audit never modifies the clean input


def test_infeasible_start_is_repaired(table, cfg, groups, monkeypatch):
    raw, y = table

    def everything_in_train(counts, order, objective):
        assign = np.zeros(len(counts), dtype=np.int64)
        return assign, np.stack([counts.sum(0), np.zeros(counts.shape[1], dtype=np.int64),
                                 np.zeros(counts.shape[1], dtype=np.int64)])

    monkeypatch.setattr(m2, "_greedy", everything_in_train)
    result = m2.make_split(y, groups, cfg)
    first = result["attempts"][0]
    assert first["start_V"] > 0 and first["V"] == 0 and first["accepted"]


def test_seed_fallback_picks_first_passing_seed_and_records_all(table, cfg, groups, monkeypatch):
    _, y = table
    real = m2._search

    def reject_first_seed(counts, sizes, uniq, seed, split_cfg, risk=None):
        run = real(counts, sizes, uniq, seed, split_cfg, risk)
        if seed == cfg["split"]["seeds"][0]:
            run["V"] = 4
        return run

    monkeypatch.setattr(m2, "_search", reject_first_seed)
    result = m2.make_split(y, groups, cfg)
    assert result["seed"] == cfg["split"]["seeds"][1]
    assert [a["seed"] for a in result["attempts"]] == cfg["split"]["seeds"][:2]
    assert [a["accepted"] for a in result["attempts"]] == [False, True]
    assert "hard minima" in result["attempts"][0]["reason"]


def test_infeasible_split_raises_and_writes_only_attempts(tmp_path, table):
    raw, y = table
    cfg = finish(synthetic_config(minima=(10_000, 3, 3)), raw)
    out = tmp_path / "split"
    with pytest.raises(m2.SplitInfeasibleError, match="Binding class") as info:
        m2.create_split(cfg, out, table=table, source=SOURCE)
    assert [a["seed"] for a in info.value.attempts] == cfg["split"]["seeds"]
    assert [p.name for p in out.iterdir()] == ["split_attempts.json"]


def test_seed_with_constant_training_feature_is_rejected_and_recorded(table, cfg, groups, monkeypatch):
    # Defensive path: with pinning disabled the exact post-hoc check still rejects the seed.
    monkeypatch.setattr(m2, "_variability_risk", lambda raw, groups, split_cfg: (None, []))
    raw, y = table
    seeds = cfg["split"]["seeds"]
    train_by_seed = {}
    for seed in seeds:
        one = copy.deepcopy(cfg)
        one["split"]["seeds"] = [seed]
        train_by_seed[seed] = set(m2.make_split(y, groups, one)["ids"]["train"].tolist())
    sizes = {g: int((groups == g).sum()) for g in np.unique(groups)}
    # A singleton row outside the first seed's training partition is the only non-constant value.
    odd = next(i for i, g in enumerate(groups) if sizes[g] == 1 and i not in train_by_seed[seeds[0]])
    raw = raw.copy()
    raw[:, 2] = 0.5
    raw[odd, 2] = 0.9
    expected = next((s for s in seeds if odd in train_by_seed[s]), None)
    if expected is None:
        with pytest.raises(m2.SplitInfeasibleError, match="constant training feature") as info:
            m2.make_split(y, groups, cfg, raw)
        attempts = info.value.attempts
    else:
        result = m2.make_split(y, groups, cfg, raw)
        assert result["seed"] == expected
        attempts = result["attempts"]
    assert attempts[0]["accepted"] is False and attempts[0]["constant_training_features"] == [2]
    assert "constant training feature" in attempts[0]["reason"]


def test_constant_feature_rejection_raises_when_every_seed_fails(table, cfg, groups, monkeypatch):
    monkeypatch.setattr(m2, "_variability_risk", lambda raw, groups, split_cfg: (None, []))
    raw, y = table
    raw = raw.copy()
    raw[:, 2] = 0.5
    with pytest.raises(m2.SplitInfeasibleError, match="constant training feature") as info:
        m2.make_split(y, groups, cfg, raw)
    assert len(info.value.attempts) == len(cfg["split"]["seeds"])


def test_pinning_puts_a_rare_value_row_into_train_for_every_seed(table, cfg, groups):
    raw, y = table
    seeds = cfg["split"]["seeds"]
    sizes = {g: int((groups == g).sum()) for g in np.unique(groups)}
    singles = [i for i, g in enumerate(groups) if sizes[g] == 1]
    raw = raw.copy()
    raw[:, 2] = 0.5
    odd = singles[:2]
    raw[odd, 2] = [0.9, 0.1]
    for seed in seeds:
        one = copy.deepcopy(cfg)
        one["split"]["seeds"] = [seed]
        result = m2.make_split(y, groups, one, raw)
        assert result["at_risk_features"] == [2]
        train = set(result["ids"]["train"].tolist())
        assert train & set(odd)
        assert len(np.unique(raw[sorted(train), 2])) >= 2
        assert result["attempts"][0]["accepted"] and result["attempts"][0]["V"] == 0
        for p in ("validation", "test"):
            assert np.bincount(y[result["ids"][p]], minlength=4).min() >= cfg["split"]["hard_minima"][p]


def test_globally_constant_feature_is_a_data_integrity_error(table, cfg, groups):
    raw, y = table
    raw = raw.copy()
    raw[:, 2] = 0.5
    with pytest.raises(m2.DataIntegrityError, match="constant over the whole dataset"):
        m2.make_split(y, groups, cfg, raw)


def test_variability_pinning_is_deterministic_and_recorded(table, cfg, groups):
    raw, y = table
    sizes = {g: int((groups == g).sum()) for g in np.unique(groups)}
    raw = raw.copy()
    raw[:, 2] = 0.5
    raw[next(i for i, g in enumerate(groups) if sizes[g] == 1), 2] = 0.9
    a = m2.make_split(y, groups, cfg, raw)
    b = m2.make_split(y, groups, cfg, raw)
    assert all(np.array_equal(a["ids"][p], b["ids"][p]) for p in PARTS)
    split = m2.split_document(cfg, a, y, groups, SOURCE, "0" * 64)
    assert split["feature_variability"]["at_risk_features"] == ["f2"]


def test_band_miss_is_not_silently_widened(table):
    raw, y = table
    cfg = finish(synthetic_config(band=((0.95, 0.99), (0.08, 0.22), (0.08, 0.22))), raw)
    with pytest.raises(m2.SplitInfeasibleError):
        m2.make_split(y, m2.group_hashes(raw), cfg)


@pytest.mark.slow
def test_frozen_real_split_if_present():
    cfg = m2.load_config(REAL_CONFIG)
    out = m2.ROOT / cfg["split"]["output_dir"]
    if not (out / "split_m2.json").exists():
        pytest.skip("frozen split not created yet")
    summary = m2.audit_existing(cfg)
    assert sum(summary["partition_rows"].values()) == 29997
    assert summary["classes_in_every_partition"]
