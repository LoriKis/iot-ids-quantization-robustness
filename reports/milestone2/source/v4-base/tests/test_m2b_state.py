import copy
import json
import shutil
from pathlib import Path

import pytest

from research import m2b_state as st
from research.common import ROOT
from research.m2b_io import sha256_file

SOURCE_COMPONENTS = ("source_train", "source_quant", "source_attack", "source_eff")


def _package(tmp_path):
    shutil.copytree(ROOT / "research", tmp_path / "research", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path / "research"


def _touch(path: Path):
    path.write_text(path.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")


def _digests(package_dir, pave=None):
    return st.source_components(package_dir, pave)["digests"]


# ---- fingerprints ---------------------------------------------------------------------------------------------------

def test_fingerprint_stable_and_pycache_ignored(tmp_path):
    package = _package(tmp_path)
    first = _digests(package)
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "m2_training.cpython-311.pyc").write_bytes(b"junk")
    assert _digests(package) == first
    assert _digests(package) == first


def test_orchestration_edit_changes_only_orchestration_sha(tmp_path):
    package = _package(tmp_path)
    before = st.source_components(package)
    _touch(package / "m2_campaign.py")
    after = st.source_components(package)
    assert after["digests"] == before["digests"]
    assert after["orchestration_sha"] != before["orchestration_sha"]


def test_source_train_edit_changes_all_source_components(tmp_path):
    package = _package(tmp_path)
    before = _digests(package)
    _touch(package / "m2_training.py")
    after = _digests(package)
    assert all(after[c] != before[c] for c in SOURCE_COMPONENTS)


def test_m2_quant_edit_leaves_train_unchanged(tmp_path):
    package = _package(tmp_path)
    before = _digests(package)
    _touch(package / "m2_quant.py")
    after = _digests(package)
    assert after["source_train"] == before["source_train"]
    assert all(after[c] != before[c] for c in ("source_quant", "source_attack", "source_eff"))


@pytest.mark.parametrize("name", ["quantization.py", "m2_attacks.py", "attacks.py"])
def test_attack_and_quantization_edits_reach_training(tmp_path, name):
    package = _package(tmp_path)
    before = _digests(package)
    _touch(package / name)
    assert _digests(package)["source_train"] != before["source_train"]


def test_efficiency_edit_changes_only_eff(tmp_path):
    package = _package(tmp_path)
    before = _digests(package)
    _touch(package / "m2_efficiency.py")
    after = _digests(package)
    assert after["source_eff"] != before["source_eff"]
    assert all(after[c] == before[c] for c in ("source_train", "source_quant", "source_attack"))


def test_closures_exclude_orchestration_and_include_io():
    sets = st.source_sets(ROOT / "research")
    assert st.check_orchestration_not_in_closures(ROOT / "research") == []
    assert "m2b_io.py" in sets["source_train"] and "m2_training.py" in sets["source_train"]
    assert "m2_attacks.py" in sets["source_train"] and "quantization.py" in sets["source_train"]
    assert set(sets["source_train"]) <= set(sets["source_quant"]) <= set(sets["source_attack"])
    assert "m2_quant.py" not in sets["source_train"] and "m2_quant.py" in sets["source_quant"]
    assert "m2_efficiency.py" in sets["source_eff"] and "m2_efficiency.py" not in sets["source_quant"]


def test_pave_tree_id_null_without_identity_and_included_when_present(tmp_path):
    package = _package(tmp_path)
    assert st.read_pave_tree_id(tmp_path) is None
    base = _digests(package, None)
    (tmp_path / "PACKAGE_IDENTITY.json").write_text(json.dumps({"pave": {"tree_id": "t" * 40}}), encoding="utf-8")
    assert st.read_pave_tree_id(tmp_path) == "t" * 40
    assert _digests(package, "t" * 40) != base


# ---- import closure -------------------------------------------------------------------------------------------------

def _mini_package(tmp_path, files):
    package = tmp_path / "research"
    package.mkdir()
    for name, text in files.items():
        (package / f"{name}.py").write_text(text, encoding="utf-8")
    return package


def test_relative_and_nested_imports_are_resolved(tmp_path):
    package = _mini_package(tmp_path, {
        "root": "from . import a, b\nfrom .c import thing\nimport numpy\n",
        "a": "def f():\n    from .d import x\n",
        "b": "if True:\n    import research.e\n",
        "c": "from research import f\n",
        "d": "", "e": "", "f": "from . import root\n", "unused": "",
    })
    assert st.import_closure(["root"], package) == ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py", "root.py"]


def test_level_two_and_unresolved_imports_raise(tmp_path):
    package = _mini_package(tmp_path, {"root": "from .. import x\n"})
    with pytest.raises(st.ClosureResolutionError):
        st.import_closure(["root"], package)
    other = tmp_path / "other"
    other.mkdir()
    package2 = _mini_package(other, {"root": "from .missing import y\n"})
    with pytest.raises(st.ClosureResolutionError):
        st.import_closure(["root"], package2)


# ---- runbook table --------------------------------------------------------------------------------------------------

def test_runbook_table_matches_regenerated(tmp_path):
    sets = st.fingerprint_sets(ROOT / "research")
    text = f"intro\n{st.RUNBOOK_BEGIN}\nOLD\n{st.RUNBOOK_END}\noutro\n"
    updated = st.replace_runbook_table(text, st.runbook_table(sets))
    assert st.runbook_table_matches(updated, sets)
    assert not st.runbook_table_matches(updated.replace("m2_training.py", "m2_trainingX.py"), sets)
    assert "m2b_io.py" in sets["components"]["source_train"]
    assert sets["kinds_by_file"]["m2_training.py"] == sorted(st.KIND_SOURCE)
    assert sets["kinds_by_file"]["m2_quant.py"] == sorted(k for k, c in st.KIND_SOURCE.items() if c != "source_train")


# ---- configs, data and split components -----------------------------------------------------------------------------

def test_config_component_verifies_sibling_hashes(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "a.json").write_text("{}", encoding="utf-8")
    campaign = {"siblings": {"a": {"path": "configs/a.json", "sha256": sha256_file(tmp_path / "configs" / "a.json")}}}
    path = tmp_path / "configs" / "campaign.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    digest, files = st.config_component(tmp_path, path)
    assert set(files) == {"configs/campaign.json", "configs/a.json"}
    (tmp_path / "configs" / "a.json").write_text('{"changed": 1}', encoding="utf-8")
    with pytest.raises(st.ConfigIntegrityError):
        st.config_component(tmp_path, path)


@pytest.mark.needs_real_data
def test_real_fingerprints_compute_and_are_stable():
    first = st.compute_fingerprints(ROOT)
    second = st.compute_fingerprints(ROOT)
    assert first["components"] == second["components"]
    assert set(first["components"]) == set(st.ALL_COMPONENTS)
    identity = ROOT / "PACKAGE_IDENTITY.json"
    if identity.is_file():  # transfer package: the fingerprint carries the tree id recorded at packaging
        expected = json.loads(identity.read_text(encoding="utf-8"))["pave"]["tree_id"]
        assert isinstance(expected, str) and len(expected) == 40 and set(expected) <= set("0123456789abcdef")
        assert first["pave_tree_id"] == expected
        assert second["pave_tree_id"] == expected
    else:  # the development checkout has no PACKAGE_IDENTITY.json
        assert first["pave_tree_id"] is None
        assert second["pave_tree_id"] is None
    assert first["details"]["data"]["dataset_sha256"] == "0d869e334cc9a2795c96996fbde8fb04a8acac61f5e67c5d46aee2b0717a53ed"


# ---- state, mismatch, archive, completion ---------------------------------------------------------------------------

def _fp(**changes):
    comps = {c: c[:3] * 21 + "0" for c in st.ALL_COMPONENTS}
    comps.update(changes)
    return {"schema": 1, "pave_tree_id": None, "components": comps, "orchestration_sha": "o" * 64,
            "details": {"source_files": {c: {f"research/{c}.py": comps[c]} for c in SOURCE_COMPONENTS},
                        "orchestration": {"research/m2_campaign.py": "1" * 64}, "config": {}, "data": {}, "split": {},
                        "env": {}, "warn_only": {}}}


def _unit(state, out_root, unit_id, kind, fp, seed=1, status="completed", payload=b"data"):
    unit_dir = f"seeds/s{seed}/{unit_id}"
    (out_root / unit_dir).mkdir(parents=True, exist_ok=True)
    (out_root / unit_dir / "out.bin").write_bytes(payload)
    hashes, sizes = st.describe_outputs(out_root, [f"{unit_dir}/out.bin"])
    record = st.make_unit_record(unit_id, kind, "validation", seed, status, fingerprints=fp, unit_dir=unit_dir,
                                 outputs=hashes, output_sizes=sizes)
    state.write_unit(record)
    return record


def test_mismatch_raises_names_component_and_files_and_writes_nothing(tmp_path):
    state = st.CampaignState(tmp_path / "state")
    state.init("run", "real", [1], fingerprints=_fp())
    before = sorted(p.name for p in (tmp_path / "state").rglob("*"))
    with pytest.raises(st.FingerprintMismatchError) as err:
        st.require_matching(state.read_fingerprints(), _fp(source_train="x" * 64))
    assert "source_train" in str(err.value) and "train" in str(err.value)
    assert err.value.differences[0]["files"] == ["research/source_train.py"]
    assert sorted(p.name for p in (tmp_path / "state").rglob("*")) == before


def test_orchestration_difference_warns_only(tmp_path, caplog):
    recorded, current = _fp(), _fp()
    current["details"]["orchestration"] = {"research/m2_campaign.py": "2" * 64}
    import logging
    with caplog.at_level(logging.WARNING):
        changed = st.require_matching(recorded, current, logging.getLogger("t"))
    assert changed == ["research/m2_campaign.py"] and "WARN orchestration code changed" in caplog.text


def test_archive_moves_only_affected_units(tmp_path):
    out = tmp_path / "out"
    state = st.CampaignState(out / "state")
    old = _fp()
    state.init("run", "real", [1], fingerprints=old)
    train = _unit(state, out, "s1.val.train.V1", "train", old)
    attack = _unit(state, out, "s1.val.attack.V2.bpda_int8.fgsm.eps0.05", "attack", old)
    new = _fp(source_attack="z" * 64)
    moved = st.archive_and_restart_affected(state, out, "fix1", new, now="T0")
    assert state.read_unit(train["id"]) is not None and (out / train["unit_dir"]).exists()
    assert state.read_unit(attack["id"]) is None and not (out / attack["unit_dir"]).exists()
    archived = list((out / "archive").glob("fix1-T0/*"))
    assert {p.name for p in archived} >= {attack["id"] + ".json", "fingerprints.json", Path(attack["unit_dir"]).name}
    assert state.read_fingerprints()["components"]["source_attack"] == "z" * 64
    assert moved


def test_archive_tag_validation(tmp_path):
    state = st.CampaignState(tmp_path / "state")
    state.init("run", "real", [1], fingerprints=_fp())
    with pytest.raises(ValueError):
        st.archive_and_restart_affected(state, tmp_path, "bad tag!", _fp())


def test_completion_check_detects_truncation_and_change(tmp_path):
    out = tmp_path / "out"
    state = st.CampaignState(out / "state")
    state.init("run", "real", [1], fingerprints=_fp())
    record = _unit(state, out, "s1.val.train.V1", "train", _fp(), payload=b"0123456789")
    st.verify_unit_completion(record, out)
    path = out / record["unit_dir"] / "out.bin"
    path.write_bytes(b"01234")
    with pytest.raises(st.UnitCorruptionError, match="size"):
        st.verify_unit_completion(record, out)
    path.write_bytes(b"0123456780")
    with pytest.raises(st.UnitCorruptionError, match="SHA-256"):
        st.verify_unit_completion(record, out)
    path.unlink()
    with pytest.raises(st.UnitCorruptionError, match="missing"):
        st.verify_unit_completion(record, out)
    running = dict(record, status="running")
    with pytest.raises(st.UnitCorruptionError):
        st.verify_unit_completion(running, out)


def test_incomplete_unit_directory_is_archived(tmp_path):
    out = tmp_path / "out"
    state = st.CampaignState(out / "state")
    state.init("run", "real", [1], fingerprints=_fp())
    (out / "seeds/s1/u").mkdir(parents=True)
    (out / "seeds/s1/u/partial.bin").write_bytes(b"x")
    moved = st.archive_incomplete_unit(state, out, "s1.val.train.V1", "seeds/s1/u", now="T1")
    assert moved is not None and not (out / "seeds/s1/u").exists()
    assert (out / "archive" / "incomplete-T1" / "u" / "partial.bin").exists()
    # a completed unit directory is never moved
    record = _unit(state, out, "s1.val.train.V3", "train", _fp())
    assert st.archive_incomplete_unit(state, out, record["id"], record["unit_dir"]) is None
    assert (out / record["unit_dir"]).exists()


def test_adding_seeds_keeps_seed1_records_untouched(tmp_path):
    out = tmp_path / "out"
    state = st.CampaignState(out / "state")
    state.init("run", "real", [1], fingerprints=_fp())
    record = _unit(state, out, "s1.val.train.V1", "train", _fp())
    before = state.unit_path(record["id"]).read_bytes()
    st.CampaignState(out / "state").init("run", "real", [7, 13, 29], opt_ins=["qat"])
    assert state.unit_path(record["id"]).read_bytes() == before
    assert state.read_state()["seeds_requested"] == [1, 7, 13, 29] and state.read_state()["opt_ins"] == ["qat"]
    st.verify_unit_completion(state.read_unit(record["id"]), out)


def test_smoke_and_real_states_are_never_shared(tmp_path):
    state = st.CampaignState(tmp_path / "state")
    state.init("run", "smoke", [1], fingerprints=_fp())
    with pytest.raises(st.FingerprintMismatchError):
        state.init("run", "real", [1])


def test_unit_record_schema_validation(tmp_path):
    state = st.CampaignState(tmp_path / "state")
    state.init("run", "real", [1], fingerprints=_fp())
    bad = st.make_unit_record("u", "train", "validation", 1, "completed", fingerprints=_fp())
    bad["status"] = "weird"
    with pytest.raises(st.UnitCorruptionError):
        state.write_unit(bad)
    good = st.make_unit_record("u", "ptq", "validation", 1, "failed", fingerprints=_fp())
    assert set(good["fingerprints_used"]) == {"source_quant", "config", "data", "split", "env"}
    assert good["orchestration_sha"] == "o" * 64


# ---- evaluation freeze stub -----------------------------------------------------------------------------------------

def test_verify_eval_freeze_checks_hex_and_hash(tmp_path):
    freeze = tmp_path / "eval_freeze.json"
    freeze.write_text(json.dumps({"seeds": [1]}), encoding="utf-8")
    digest = sha256_file(freeze)
    assert st.verify_eval_freeze(freeze, digest).data == {"seeds": [1]}
    for bad in ("", "ABC", digest.upper(), "0" * 64):
        with pytest.raises(st.FreezeGuardError):
            st.verify_eval_freeze(freeze, bad)
    with pytest.raises(st.FreezeGuardError):
        st.verify_eval_freeze(tmp_path / "missing.json", digest)
