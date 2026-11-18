"""Final-test gating tests (DESIGN 4.8, AC-11): stub executors, no model, no training, no attack."""
import copy
import json
import shutil

import numpy as np
import pytest
from m2b_stub_support import BASE_SEED1_STAGES, cli, run_dir, snapshot, stub_executors, unit_record

from research import m2_attacks as ma
from research import m2_split
from research import m2b_campaign as mc
from research import m2b_exec as ex
from research import m2b_state
from research.m2_split import FreezeBase, PartitionGuardError
from research.m2b_state import FreezeGuardError

FREEZE = "f1"


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """A completed (stub) seed-1 validation run plus a freeze named f1; copied per test because tests tamper with it."""
    base = tmp_path_factory.mktemp("frozen")
    assert cli(base, "run", "--phase", "validation", "--seeds", "1", "--stages", *BASE_SEED1_STAGES,
               executors=stub_executors()) == 0
    assert cli(base, "freeze-eval", "--seeds", "1", "--freeze-name", FREEZE) == 0
    return base


@pytest.fixture()
def run(finished, tmp_path):
    shutil.copytree(finished / "t", tmp_path / "t")
    return tmp_path


def sha(tmp_path, name=FREEZE):
    return (run_dir(tmp_path) / "freezes" / f"{name}.json.sha256").read_text(encoding="utf-8").strip()


def spies(monkeypatch):
    """Record any call that reads or hashes the dataset."""
    calls = []
    for owner, name in ((m2_split, "read_table"), (m2_split, "load_arrays"), (m2b_state, "data_component")):
        monkeypatch.setattr(owner, name, lambda *a, _n=name, **k: calls.append(_n))
    return calls


def test_validation_never_loads_the_test_partition(tmp_path, monkeypatch):
    seen = []

    def fake_load(split, scaler, partitions=("train", "validation"), **kwargs):
        seen.append((tuple(partitions), kwargs))
        return {p: {"ids": np.arange(3), "x": np.zeros((3, 39), "float32"), "y": np.zeros(3, int)} for p in partitions}
    monkeypatch.setattr(m2_split, "load_arrays", fake_load)

    def hook(unit, env):
        env.arrays()
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "train",
               executors=stub_executors(hook=hook)) == 0
    assert seen and all(parts == ("train", "validation") and not kw for parts, kw in seen)


def test_validation_env_refuses_a_loader_that_returns_test(tmp_path):
    env = ex.Env(".", tmp_path, {"epochs": 25}, arrays_loader=lambda parts: {"train": {}, "validation": {}, "test": {}})
    with pytest.raises(PartitionGuardError):
        env.arrays()


def test_validation_command_line_has_no_path_to_the_test_partition(tmp_path):
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--confirm-freeze", "a" * 64,
               executors=stub_executors()) == 2
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "final_test",
               executors=stub_executors()) == 2
    assert cli(tmp_path, "run", "--phase", "test", "--stages", "train", "--confirm-freeze", "a" * 64) == 2
    assert not run_dir(tmp_path).exists()


@pytest.mark.parametrize("confirm", [None, "", "not-hex", "A" * 64, "0" * 64])
def test_test_phase_without_a_valid_confirm_exits_3_before_any_data_access(run, monkeypatch, confirm):
    calls = spies(monkeypatch)
    before = snapshot(run_dir(run))
    args = ["run", "--phase", "test", "--seeds", "1", "--freeze-name", FREEZE]
    if confirm is not None:
        args += ["--confirm-freeze", confirm]
    assert cli(run, *args, executors=stub_executors()) == 3
    assert calls == [] and snapshot(run_dir(run)) == before


def test_the_sha_of_another_freeze_is_rejected(run, monkeypatch):
    assert cli(run, "freeze-eval", "--seeds", "1", "--freeze-name", "f2") == 0
    other = sha(run, "f2")
    assert other != sha(run, FREEZE)
    calls = spies(monkeypatch)
    assert cli(run, "run", "--phase", "test", "--seeds", "1", "--freeze-name", FREEZE, "--confirm-freeze", other,
               executors=stub_executors()) == 3
    assert calls == []


def test_missing_freeze_file_exits_3(run, monkeypatch):
    calls = spies(monkeypatch)
    assert cli(run, "run", "--phase", "test", "--seeds", "1", "--freeze-name", "nope", "--confirm-freeze", "a" * 64,
               executors=stub_executors()) == 3 and calls == []


def test_final_test_purpose_needs_a_verified_freeze(tmp_path):
    missing = tmp_path / "does-not-exist.json"          # the guard fires before any file is read
    with pytest.raises(PartitionGuardError):
        m2_split.load_arrays(missing, missing, ("test",), purpose="final_test")
    with pytest.raises(PartitionGuardError):
        m2_split.load_arrays(missing, missing, ("test",), purpose="final_test", freeze=object())
    with pytest.raises(PartitionGuardError):
        m2_split.load_arrays(missing, missing, ("test",))
    with pytest.raises(PartitionGuardError):
        m2_split.load_arrays(missing, missing, ("train",), freeze=object())
    forged_file = tmp_path / "forged.json"
    forged_file.write_text("{}", encoding="utf-8")
    forged = FreezeBase(forged_file, "0" * 64, {})       # right type, wrong hash: refused as changed after verification
    with pytest.raises(PartitionGuardError):
        m2_split.load_arrays(missing, missing, ("test",), purpose="final_test", freeze=forged)


def test_a_valid_test_run_loads_test_rows_only_with_the_verified_freeze(run, monkeypatch):
    seen = []

    def fake_load(split, scaler, partitions, **kwargs):
        seen.append((tuple(partitions), kwargs.get("purpose"), kwargs.get("freeze")))
        return {"test": {"ids": np.arange(3), "x": np.zeros((3, 39), "float32"), "y": np.zeros(3, int)}}
    monkeypatch.setattr(m2_split, "load_arrays", fake_load)

    def hook(unit, env):
        env.arrays()
    stubs = stub_executors(hook=hook)
    assert cli(run, "run", "--phase", "test", "--seeds", "1", "--freeze-name", FREEZE, "--confirm-freeze", sha(run),
               executors=stubs) == 0
    parts, purpose, freeze = seen[0]
    assert parts == ("test",) and purpose == "final_test" and isinstance(freeze, m2b_state.EvalFreeze)
    assert len(seen) == 1                                 # loaded once per process, not once per unit
    ids = set(stubs["train"].calls)
    assert len(ids) == 4 + 120 and all(".test." in u for u in ids)   # clean + attack only: no train, ptq, e4, qat
    assert unit_record(run, "s1.test.clean.V4")["status"] == "completed"


def test_a_modified_checkpoint_after_the_freeze_blocks_the_test_phase(run, monkeypatch):
    (run_dir(run) / "seeds/s1/models/V1/checkpoint.pt").write_bytes(b"tampered")
    calls = spies(monkeypatch)
    assert cli(run, "run", "--phase", "test", "--seeds", "1", "--freeze-name", FREEZE, "--confirm-freeze", sha(run),
               executors=stub_executors()) == 3
    assert calls == []


@pytest.mark.parametrize("relative", ["seeds/s1/calibration/calibration_ids.json", "seeds/s1/calibration/calibration_x.pt",
                                      "seeds/s1/models/V2/conversion.json", "seeds/s1/models/V4/state.pt",
                                      "step0/budget_decision.json"])
def test_any_frozen_input_changed_after_the_freeze_blocks_the_test_phase(run, monkeypatch, relative):
    path = run_dir(run) / relative
    path.write_bytes(path.read_bytes() + b" ")
    calls = spies(monkeypatch)
    assert cli(run, "run", "--phase", "test", "--seeds", "1", "--freeze-name", FREEZE, "--confirm-freeze", sha(run),
               executors=stub_executors()) == 3
    assert calls == []


def test_a_changed_freeze_file_does_not_match_its_recorded_sha(run, monkeypatch):
    digest = sha(run)
    path = run_dir(run) / "freezes" / f"{FREEZE}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["seeds"] = [1, 7]
    path.write_text(json.dumps(doc), encoding="utf-8")
    calls = spies(monkeypatch)
    assert cli(run, "run", "--phase", "test", "--seeds", "1", "--freeze-name", FREEZE, "--confirm-freeze", digest,
               executors=stub_executors()) == 3 and calls == []


def test_test_phase_seeds_must_be_frozen_and_stages_limited(run):
    args = ["--freeze-name", FREEZE, "--confirm-freeze", sha(run)]
    assert cli(run, "run", "--phase", "test", "--seeds", "7", *args, executors=stub_executors()) == 3
    assert cli(run, "run", "--phase", "test", "--seeds", "1", "--stages", "train", *args, executors=stub_executors()) == 2
    assert cli(run, "run", "--phase", "test", "--seeds", "1", "--stages", "qat", "--opt-in", "qat", *args,
               executors=stub_executors()) == 3                      # the freeze has qat_included false
    assert cli(run, "run", "--phase", "test", "--seeds", "1", "--archive-and-restart-affected", "x", *args,
               executors=stub_executors()) == 2                      # the test phase has no override


def test_the_test_attack_config_is_a_validated_copy(tmp_path, monkeypatch):
    partitions = []
    real = ma.validate_config
    monkeypatch.setattr(ma, "validate_config", lambda cfg: partitions.append(cfg["evaluation"]["evaluation_partition"]) or real(cfg))
    validated = ma.load_config("configs/m2/attacks.json")
    copied = ex.deep_copy_test_attack_cfg(validated)
    assert copied["evaluation"]["evaluation_partition"] == "test"
    assert validated["evaluation"]["evaluation_partition"] == "validation"       # the original is untouched
    assert partitions == ["validation"]                                          # only the unmodified config was validated
    assert copy.deepcopy(validated) == validated and copied is not validated
    with pytest.raises(Exception):
        real(copied)                                                             # a test copy never passes validation


def test_freeze_eval_preconditions(tmp_path, monkeypatch):
    assert cli(tmp_path, "freeze-eval", "--freeze-name", "x") == 2                # --seeds is required
    assert cli(tmp_path, "freeze-eval", "--seeds", "1") == 3                      # no run state at all
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "train", "calib", "paircheck", "ptq",
               "clean", executors=stub_executors()) == 0
    assert cli(tmp_path, "freeze-eval", "--seeds", "1") == 3                      # no budget decision yet
    assert not (run_dir(tmp_path) / "freezes").exists()


def test_freeze_eval_refuses_to_overwrite_and_unfinished_units(run, finished):
    assert cli(run, "freeze-eval", "--seeds", "1", "--freeze-name", FREEZE) == 2
    assert cli(run, "freeze-eval", "--seeds", "1", "7", "--freeze-name", "f9") == 3   # seed 7 has not run
    assert not (run_dir(run) / "freezes" / "f9.json").exists()
    doc = json.loads((run_dir(run) / "freezes" / f"{FREEZE}.json").read_text(encoding="utf-8"))
    assert doc["qat_included"] is False and doc["attack_budget"] == "pgd20" and set(doc["models"]["1"]) == {"V1", "V2", "V3", "V4"}
    assert doc["models"]["1"]["V1"]["calibration_x_sha256"] is None and doc["models"]["1"]["V2"]["conversion_json_sha256"]


def test_include_qat_needs_every_qat_unit_finished(run, monkeypatch):
    assert cli(run, "freeze-eval", "--seeds", "1", "--include-qat", "--freeze-name", "fq") == 3
    monkeypatch.setenv("M2_ENABLE_QAT", "yes")
    assert cli(run, "run", "--phase", "validation", "--seeds", "1", "--stages", "qat", "--opt-in", "qat",
               executors=stub_executors(blocked={"s1.val.qat.QAT-FT"})) == 0
    assert cli(run, "freeze-eval", "--seeds", "1", "--include-qat", "--freeze-name", "fq") == 0   # blocked with a reason is allowed
    doc = json.loads((run_dir(run) / "freezes/fq.json").read_text(encoding="utf-8"))
    assert doc["qat_included"] is True and "QAT-FT" not in doc["models"]["1"] and "QAT-FT" in "".join(doc["qat_blocked"])
    assert {"FP32-FT", "FP32-FT-PTQ"} <= set(doc["models"]["1"])
    stubs = stub_executors()
    assert cli(run, "run", "--phase", "test", "--seeds", "1", "--freeze-name", "fq", "--confirm-freeze", sha(run, "fq"),
               executors=stubs) == 0
    assert any(u.startswith("s1.test.attack.FP32-FT-PTQ.") for u in stubs["train"].calls)
    assert not any("QAT-FT" in u for u in stubs["train"].calls)


def test_plain_validation_without_the_opt_in_never_plans_qat(tmp_path):
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "qat", executors=stub_executors()) == 3
