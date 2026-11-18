from __future__ import annotations

import copy
import json

import pytest

from conftest import CONFIG_PATH

from m2_deadline import common
from m2_deadline.common import ConfigError, load_config, validate_config


def raw():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_config_parses_and_has_design_values(config):
    assert config["run_id"] == "m2-deadline-v1"
    assert config["seed"] == 1
    assert config["training"]["epoch_candidates"] == [25, 10, 5]
    assert config["robustness"]["epsilons"] == [0.0, 0.05, 0.1]
    pgd = config["robustness"]["attacks"]["pgd5"]
    assert (pgd["steps"], pgd["restarts"]) == (5, 1)
    assert config["budget"]["reserve_hours"] == 3.0
    assert config["budget"]["hard_cap_hours"] == 9.0
    assert config["budget"]["n_stage_processes"] == 1
    assert config["budget"]["per_class_candidates"] == [10, 5]


def test_exact_key_set_rejects_unknown_and_missing_keys():
    extra = raw()
    extra["surprise"] = 1
    with pytest.raises(ConfigError, match="unknown keys"):
        validate_config(extra)
    missing = raw()
    del missing["budget"]["safety_multiplier"]
    with pytest.raises(ConfigError, match="missing keys"):
        validate_config(missing)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["training"].__setitem__("epoch_candidates", [25, 10]),
        lambda c: c["robustness"].__setitem__("epsilons", [0.0, 0.05, 0.10, 0.2]),
        lambda c: c["robustness"]["attacks"]["pgd5"].__setitem__("steps", 20),
        lambda c: c["robustness"]["attacks"]["pgd5"].__setitem__("restarts", 3),
        lambda c: c["budget"].__setitem__("reserve_hours", 2.0),
        lambda c: c["budget"].__setitem__("hard_cap_hours", 10.0),
        lambda c: c["budget"].__setitem__("n_stage_processes", 10),
        lambda c: c["budget"].__setitem__("per_class_candidates", [10]),
        lambda c: c.__setitem__("seed", 2),
        lambda c: c["efficiency"].__setitem__("threads", 4),
        lambda c: c["robustness"].__setitem__("split_partition", "validation"),
    ],
)
def test_frozen_design_values_are_enforced(mutate):
    cfg = raw()
    mutate(cfg)
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_type_errors_are_rejected():
    cfg = raw()
    cfg["budget"]["safety_multiplier"] = "1.25"
    with pytest.raises(ConfigError, match="finite number"):
        validate_config(cfg)
    cfg = raw()
    cfg["training"]["threads"] = True  # bool is not an int here
    with pytest.raises(ConfigError, match="integer"):
        validate_config(cfg)


def test_load_config_reports_unreadable_and_invalid_json(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(bad)


def test_no_placeholder_text_in_config():
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert "..." not in text
    assert "\r" not in text
    cfg = raw()
    cfg["claim"] = "to be written ..."
    with pytest.raises(ConfigError, match="placeholder"):
        validate_config(cfg)


def test_claim_text_is_complete(config):
    claim = config["claim"]
    for phrase in ("One seed (1)", "post-training quantization", "feature-space", "BPDA", "QAT feasibility only"):
        assert phrase in claim


def test_no_key_duplicates_a_v4_config_value(config):
    forbidden = {"rows", "reps", "repetitions", "calibration_per_class", "split_seed"}
    assert not (forbidden & set(config["efficiency"]))

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in forbidden and path != "sources":
                    raise AssertionError("duplicate of a V4 value at %s.%s" % (path, key))
                walk(value, "%s.%s" % (path, key) if path else key)

    walk(config, "")
    # the two entries under "sources" only describe where the values come from (strings)
    assert all(isinstance(v, str) for v in config["sources"].values())


def test_no_timing_key_refers_to_the_test_partition(config):
    assert "test" not in json.dumps(config["timing_probe"]).lower()
    assert config["efficiency"]["rows_partition"] == "validation"


def test_pgd5_config_agrees_with_computed_selector_table(config):
    from m2_deadline.protocol import selector_calls

    pgd = config["robustness"]["attacks"]["pgd5"]
    assert selector_calls(pgd["kind"], pgd["steps"], pgd["restarts"]) == {"selector_calls": 8, "candidates": 7}


def test_validate_config_does_not_mutate_its_input():
    cfg = raw()
    before = copy.deepcopy(cfg)
    validate_config(cfg)
    assert cfg == before


def test_v4_sibling_values_match_assumptions(v4):
    """V4 JSON is read as text/JSON only (no research.* import)."""
    eff = json.loads((v4 / "configs/m2b/efficiency_v2.json").read_text(encoding="utf-8"))
    reps = eff["repetitions"]
    assert set(reps) == {"full", "pilot"} and 1 <= reps["pilot"] <= reps["full"]
    quant = json.loads((v4 / "configs/m2/quantization.json").read_text(encoding="utf-8"))
    assert quant["ptq"]["calibration_per_class"] == 20
    for rel in load_config(CONFIG_PATH)["siblings"].values():
        assert (v4 / rel).is_file(), rel


def test_pgd5_table_matches_v4_expected_selector_calls_source(v4):
    """The re-derived formula equals V4's source text (not imported)."""
    src = (v4 / "research" / "m2_attacks.py").read_text(encoding="utf-8")
    assert "candidates = 1 + nr * (1 + ns)" in src
    assert 'return {"selector_calls": 1 + candidates, "candidates": candidates}' in src
    attacks = json.loads((v4 / "configs/m2/attacks.json").read_text(encoding="utf-8"))
    from m2_deadline.protocol import build_pgd5_cfg, selector_calls

    for name, table in attacks["expected_selector_calls_per_batch"].items():
        if name in attacks["attacks"]:
            a = attacks["attacks"][name]
            assert selector_calls(a["kind"], a["steps"], a["restarts"]) == table


def test_contract_tables_are_consistent():
    assert set(common.CSV_COLUMNS) >= {"predictions_full_test", "predictions_subset", "model_comparison_csv"}
    assert len(set(common.PATHS.values())) == len(common.PATHS)
    assert common.eps_str(0) == "0.0" and common.eps_str(0.10) == "0.1" and common.eps_str(0.05) == "0.05"
