"""Guards that need only the split: partition access, overwrite refusal, config validation."""
import copy
import json

import pytest

from m2_helpers import SOURCE, finish, real_config, synthetic_config, synthetic_table
from research import m2_split as m2


@pytest.fixture(scope="module")
def table():
    return synthetic_table()


@pytest.fixture
def frozen(tmp_path, table):
    cfg = finish(synthetic_config(), table[0])
    out = tmp_path / "split"
    m2.create_split(cfg, out, table=table, source=SOURCE)
    return out / "split_m2.json", out / "scaler_minmax.json"


def test_test_partition_is_refused_without_audit_purpose(frozen, table):
    split, scaler = frozen
    for partitions in (("test",), ("train", "test"), m2.PARTITIONS):
        with pytest.raises(m2.PartitionGuardError):
            m2.load_arrays(split, scaler, partitions, table=table)


def test_guard_runs_before_any_data_is_read(frozen):
    split, scaler = frozen
    # A table that would crash the loader proves the guard fires first.
    with pytest.raises(m2.PartitionGuardError):
        m2.load_arrays(split, scaler, ("test",), table=None)


def test_unknown_purpose_and_partition_are_refused(frozen, table):
    split, scaler = frozen
    with pytest.raises(m2.PartitionGuardError):
        m2.load_arrays(split, scaler, ("test",), purpose="evaluation", table=table)
    with pytest.raises(m2.PartitionGuardError):
        m2.load_arrays(split, scaler, ("holdout",), table=table)


def test_default_route_returns_only_train_and_validation(frozen, table):
    split, scaler = frozen
    assert set(m2.load_arrays(split, scaler, table=table)) == {"train", "validation"}
    audit = m2.load_arrays(split, scaler, ("test",), purpose="range_audit", table=table)
    assert set(audit) == {"test"}


def test_second_write_refuses_to_overwrite(frozen, table):
    split, _ = frozen
    cfg = finish(synthetic_config(), table[0])
    before = split.read_bytes()
    with pytest.raises(FileExistsError):
        m2.create_split(cfg, split.parent, table=table, source=SOURCE)
    assert split.read_bytes() == before


def test_partial_write_blocks_a_rerun(tmp_path, table):
    cfg = finish(synthetic_config(), table[0])
    out = tmp_path / "split"
    out.mkdir()
    (out / "scaler_minmax.json").write_text("{}")
    with pytest.raises(FileExistsError):
        m2.create_split(cfg, out, table=table, source=SOURCE)


def _mutated(mutate):
    cfg = copy.deepcopy(real_config())
    mutate(cfg)
    return cfg


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(extra=1),
    lambda c: c.pop("scaler"),
    lambda c: c["split"].update(unknown=1),
    lambda c: c["split"].update(seeds=[]),
    lambda c: c["split"].update(seeds=[1, 1]),
    lambda c: c["split"].update(seeds=[1.5]),
    lambda c: c["split"]["targets"].update(train=0.5),
    lambda c: c["split"]["hard_minima"].update(test=0),
    lambda c: c["split"]["acceptance_band"].update(train=[0.8, 0.7]),
    lambda c: c["split"]["acceptance_band"].update(validation=[0.2, 0.3]),
    lambda c: c["split"].update(proposals=0),
    lambda c: c["split"]["feature_variability"].update(min_distinct_train_values=1),
    lambda c: c["split"]["feature_variability"].update(at_risk_max_minority_groups=0),
    lambda c: c["split"]["feature_variability"].update(extra=1),
    lambda c: c["split"].pop("feature_variability"),
    lambda c: c["split"].update(algorithm="other"),
    lambda c: c["scaler"].update(clip=True),
    lambda c: c["scaler"].update(feature_range=[0, 2]),
    lambda c: c["scaler"].update(fit_partition="all"),
    lambda c: c["grouping"].update(label_excluded=False),
    lambda c: c["dataset"].update(sha256="abc"),
    lambda c: c["dataset"].update(rows=-1),
    lambda c: c["dataset"]["class_names"].pop(),
    lambda c: c["range_audit"].update(epsilon_grid=[0.01, 0.05]),
    lambda c: c["range_audit"].update(epsilon_grid=[0, 0.05, 0.05]),
    lambda c: c["range_audit"].update(epsilon_grid=[0, 2]),
    lambda c: c.update(schema_version=2),
])
def test_invalid_configs_raise_config_error(mutate):
    with pytest.raises(m2.ConfigError):
        m2.validate_config(_mutated(mutate))


def test_config_file_problems_raise_config_error(tmp_path):
    with pytest.raises(m2.ConfigError):
        m2.load_config(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(m2.ConfigError):
        m2.load_config(bad)
    renamed = tmp_path / "renamed.json"
    cfg = real_config()
    cfg["dataset"]["class_names"][0], cfg["dataset"]["class_names"][1] = (
        cfg["dataset"]["class_names"][1], cfg["dataset"]["class_names"][0])
    renamed.write_text(json.dumps(cfg))
    with pytest.raises(m2.ConfigError, match="Experiment C"):
        m2.load_config(renamed)


# ---------------------------------------------------------------- pilot, campaign and phase guards (FEAT-004)
from research import m2_campaign as mc  # noqa: E402
from research import m2_pilot as mp  # noqa: E402
from research.common import ROOT, sha256  # noqa: E402


def _pilot_cfg():
    return json.loads(mp.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _campaign_cfg():
    return json.loads(mc.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_frozen_pilot_and_campaign_configs_validate():
    cfg = mp.load_pilot_config()
    assert (cfg["seeds"], cfg["epochs"], cfg["batch_size"], cfg["phase"]) == ([1], 2, 512, "m2a")
    assert cfg["allow_partitions"] == ["train", "validation"] and cfg["output_dir"] == "artifacts/m2a-pilot"
    assert cfg["subsets"]["expected_rows"] == {"train": 2700, "validation": 540, "attack": 270, "calibration": 540}
    campaign = mc.load_campaign_config()
    assert campaign["seeds"] == [1, 7, 13, 29] and campaign["step0"]["blocking"] is True
    assert set(campaign["siblings"]) == {"data", "training", "quantization", "attacks", "efficiency", "pilot"}


def test_partition_guard_by_phase():
    for partitions in (("train",), ("validation",), ("train", "validation")):
        mc.guard_partitions("m2a", partitions)
        mc.guard_partitions("m2b_validation", partitions)
    for phase in ("m2a", "m2b_validation"):
        with pytest.raises(m2.PartitionGuardError):
            mc.guard_partitions(phase, ("test",))
        with pytest.raises(m2.PartitionGuardError):
            mc.guard_partitions(phase, ("train", "validation", "test"))
    mc.guard_partitions("m2b_test", m2.PARTITIONS)
    with pytest.raises(m2.ConfigError):
        mc.guard_partitions("m2c", ("train",))


def test_pilot_config_requesting_test_is_refused_before_anything_is_created(tmp_path):
    cfg = _pilot_cfg()
    cfg["allow_partitions"] = ["train", "validation", "test"]
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    with pytest.raises(m2.PartitionGuardError):
        mp.run_pilot(path, out_dir=out)
    assert not out.exists()


def test_pilot_refuses_an_existing_output_directory(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "keep.txt").write_text("precious")
    with pytest.raises(FileExistsError):
        mp.run_pilot(mp.DEFAULT_CONFIG, out_dir=out)
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"] and (out / "keep.txt").read_text() == "precious"


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(extra=1),
    lambda c: c.pop("threads"),
    lambda c: c.update(schema_version=2),
    lambda c: c.update(phase="m2b_validation"),
    lambda c: c.update(seeds=[]),
    lambda c: c.update(seeds=[1, 1]),
    lambda c: c.update(seeds=[1.5]),
    lambda c: c.update(seeds=[True]),
    lambda c: c.update(seeds=[7]),
    lambda c: c.update(epochs=0),
    lambda c: c.update(epochs=3),
    lambda c: c.update(epochs="2"),
    lambda c: c.update(batch_size=0),
    lambda c: c.update(batch_size=5000),
    lambda c: c.update(allow_partitions=["train"]),
    lambda c: c.update(allow_partitions="train"),
    lambda c: c["attack"].update(epsilon_grid=[0.01, 0.05]),
    lambda c: c["attack"].update(epsilon_grid=[0, 0.05, 0.05]),
    lambda c: c["attack"].update(epsilon_grid=[0, 0.1, 0.05]),
    lambda c: c["attack"].update(epsilon_grid=[0, 2]),
    lambda c: c["attack"].update(budgets=[]),
    lambda c: c["attack"].update(budgets=["pgd50x5"]),
    lambda c: c["attack"].update(sources=["fp32_direct"]),
    lambda c: c["attack"].update(extra=1),
    lambda c: c["subsets"].update(train_per_class=0),
    lambda c: c["subsets"].update(attack_per_class=99),
    lambda c: c["subsets"]["expected_rows"].pop("train"),
    lambda c: c["e4_plumbing"].update(per_class=0),
    lambda c: c["e4_plumbing"].update(per_class=11),
    lambda c: c["efficiency"].update(mode="full"),
    lambda c: c["threads"].update(training=0),
    lambda c: c.update(banner="Pilot results are scientific."),
])
def test_invalid_pilot_configs_raise_config_error(mutate):
    cfg = _pilot_cfg()
    mutate(cfg)
    with pytest.raises(m2.ConfigError):
        mp.validate_pilot_config(cfg)


def test_pilot_config_file_problems_raise_config_error(tmp_path):
    with pytest.raises(m2.ConfigError):
        mp.load_pilot_config(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(m2.ConfigError):
        mp.load_pilot_config(bad)


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(extra=1),
    lambda c: c.pop("matrix"),
    lambda c: c.update(schema_version=2),
    lambda c: c.update(seeds=[1, 7, 13, 13]),
    lambda c: c.update(seeds=[1, 7, 13, 30]),
    lambda c: c.update(seeds=[]),
    lambda c: c.update(epochs=26),
    lambda c: c.update(epochs=24),
    lambda c: c.update(batch_size=4097),
    lambda c: c.update(batch_size=256),
    lambda c: c.update(epsilon_grid=[0, 0.01, 0.025, 0.05]),
    lambda c: c.update(epsilon_grid=[0.01, 0.025, 0.05, 0.1]),
    lambda c: c.update(epsilon_grid=[0, 0.05, 0.025, 0.1]),
    lambda c: c.update(epsilon_grid=[0, 0.01, 0.025, 0.05, 1.5]),
    lambda c: c["siblings"]["attacks"].update(sha256="0" * 64),
    lambda c: c["siblings"]["data"].update(path="configs/m2/does-not-exist.json"),
    lambda c: c["siblings"].pop("efficiency"),
    lambda c: c["step0"].update(blocking=False),
    lambda c: c["phases"].pop("m2b_test"),
])
def test_invalid_campaign_configs_raise_config_error(mutate):
    cfg = _campaign_cfg()
    mutate(cfg)
    with pytest.raises(m2.ConfigError):
        mc.validate_campaign_config(cfg)


def test_campaign_hashes_match_the_sibling_configs_on_disk():
    cfg = _campaign_cfg()
    assert cfg["siblings"] == mc.sibling_hashes()


def test_m2b_test_needs_a_freeze_sha_and_is_refused_before_any_data_is_read(monkeypatch):
    reads = []
    monkeypatch.setattr(m2, "read_table", lambda *a, **k: reads.append("read_table"))
    monkeypatch.setattr(m2, "load_arrays", lambda *a, **k: reads.append("load_arrays"))
    for confirm in (None, "", "not-hex", "0" * 63):
        with pytest.raises(m2.PartitionGuardError, match="confirm-freeze"):
            mc.run_phase(mc.DEFAULT_CONFIG, "m2b_test", confirm)
    with pytest.raises(m2.PartitionGuardError):
        mc.main(["--phase", "m2b_test"])
    # well-formed hex that names no freeze file is refused by the M2B gate (exit 3) before any data access
    with pytest.raises(m2.PartitionGuardError, match="confirm-freeze"):
        mc.run_phase(mc.DEFAULT_CONFIG, "m2b_test", "0" * 64)
    assert reads == []
    with pytest.raises(m2.ConfigError):
        mc.run_phase(mc.DEFAULT_CONFIG, "m2c")
    with pytest.raises(m2.ConfigError):
        mc.load_campaign_config(ROOT / "configs/m2/missing.json")


def test_m2b_phases_delegate_to_the_m2b_runner(monkeypatch):
    from research import m2b_campaign
    calls = []
    monkeypatch.setattr(m2b_campaign, "main", lambda argv, **k: calls.append(list(argv)) or 0)
    assert mc.run_phase(mc.DEFAULT_CONFIG, "m2b_validation") == 0
    assert mc.run_phase(mc.DEFAULT_CONFIG, "m2b_test", "a" * 64) == 0
    assert calls[0] == ["run", "--phase", "validation"]
    assert calls[1][:4] == ["run", "--phase", "test", "--confirm-freeze"] and calls[1][4] == "a" * 64
    monkeypatch.setattr(m2b_campaign, "main", lambda argv, **k: 3)
    with pytest.raises(m2.PartitionGuardError):
        mc.run_phase(mc.DEFAULT_CONFIG, "m2b_test", "b" * 64)
