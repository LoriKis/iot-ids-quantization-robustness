from __future__ import annotations

import json

import pytest

from m2_deadline import markers
from m2_deadline.common import (
    GuardError,
    STAGES,
    checkpoint_path,
    marker_path,
    model_dir,
    out_path,
)

CFG = "c" * 64
LOCK = "l" * 64
PROTO = "p" * 64


def ctx(protocol=PROTO, epochs=10, config=CFG, lock=LOCK):
    return markers.context(config_sha256=config, lock_sha256=lock, protocol_sha256=protocol, epochs=epochs)


def make_output(out, name, data=b"payload"):
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------- smoke paths
@pytest.mark.parametrize(
    "path,expected",
    [
        ("output/m2b-v1-smoke/x", True),
        ("smoke/x", True),
        ("/data/run/SMOKE/out", True),
        ("output/m2b-v1/x", False),
        ("/home/smokey/x", False),
        ("/home/smoke_tests/x", False),
        ("artifacts/m2-deadline-v1", False),
    ],
)
def test_is_smoke_path_matches_components_not_substrings(path, expected):
    assert markers.is_smoke_path(path) is expected


def test_assert_not_smoke_raises_guard_error():
    with pytest.raises(GuardError, match="smoke"):
        markers.assert_not_smoke("output/m2b-v1-smoke/models")
    markers.assert_not_smoke("artifacts/m2-deadline-v1")


# ---------------------------------------------------------------- write_after_validate / verify
def test_marker_written_only_after_successful_validation(out):
    f = make_output(out, "models/V1/checkpoint.pt")
    calls = []

    def validate():
        calls.append("validated")
        assert not marker_path(out, "train_ordinary").exists()  # nothing exists while validating
        return True

    m = markers.write_after_validate(out, "train_ordinary", validate, [f], ctx(), seconds=1.5)
    assert calls == ["validated"] and marker_path(out, "train_ordinary").is_file()
    assert m["outputs"] == {"models/V1/checkpoint.pt": markers.sha256_file(f)}
    assert m["seconds"] == 1.5 and m["protocol_sha256"] == PROTO and m["epochs"] == 10
    assert json.loads(marker_path(out, "train_ordinary").read_text()) == m


def test_failed_validation_writes_no_marker_and_removes_a_stale_one(out):
    f = make_output(out, "models/V1/checkpoint.pt")
    markers.write_after_validate(out, "train_ordinary", lambda: True, [f], ctx())
    assert marker_path(out, "train_ordinary").exists()

    def boom():
        raise RuntimeError("validation failed")

    with pytest.raises(RuntimeError):
        markers.write_after_validate(out, "train_ordinary", boom, [f], ctx())
    assert not marker_path(out, "train_ordinary").exists()
    with pytest.raises(GuardError):
        markers.write_after_validate(out, "train_ordinary", lambda: False, [f], ctx())
    assert not marker_path(out, "train_ordinary").exists()


def test_missing_output_or_unknown_stage_or_missing_protocol_write_nothing(out):
    with pytest.raises(GuardError, match="missing"):
        markers.write_after_validate(out, "ptq", lambda: True, [out / "nope.pt"], ctx())
    f = make_output(out, "a.json")
    with pytest.raises(GuardError, match="unknown stage"):
        markers.write_after_validate(out, "bogus", lambda: True, [f], ctx())
    with pytest.raises(GuardError, match="protocol hash"):
        markers.write_after_validate(out, "ptq", lambda: True, [f], ctx(protocol=None))
    assert not (out / "stage_markers").exists() or not list((out / "stage_markers").iterdir())


def test_preflight_marker_has_null_protocol_hash(out):
    f = make_output(out, "logs/pre.log")
    m = markers.write_after_validate(out, "preflight", lambda: True, [f], ctx(protocol=None))
    assert m["protocol_sha256"] is None
    assert markers.verify(out, "preflight", ctx(protocol=None)) == (True, "ok")
    assert markers.verify(out, "preflight", ctx())[0] is True  # later protocol hash does not invalidate preflight


def test_verify_detects_tampering_of_a_listed_output(out):
    f = make_output(out, "models/V1/checkpoint.pt", b"abc")
    markers.write_after_validate(out, "train_ordinary", lambda: True, [f], ctx())
    assert markers.verify(out, "train_ordinary", ctx()) == (True, "ok")
    f.write_bytes(b"abd")
    ok, reason = markers.verify(out, "train_ordinary", ctx())
    assert not ok and "output changed" in reason
    f.unlink()
    ok, reason = markers.verify(out, "train_ordinary", ctx())
    assert not ok and "output missing" in reason


@pytest.mark.parametrize(
    "changed,expected",
    [
        (dict(config="d" * 64), "config_sha256"),
        (dict(lock="m" * 64), "lock_sha256"),
        (dict(protocol="q" * 64), "protocol_sha256"),
        (dict(epochs=5), "epochs"),
    ],
)
def test_verify_rejects_context_mismatch(out, changed, expected):
    f = make_output(out, "models/V1/checkpoint.pt")
    markers.write_after_validate(out, "train_ordinary", lambda: True, [f], ctx())
    ok, reason = markers.verify(out, "train_ordinary", ctx(**changed))
    assert not ok and expected in reason


def test_verify_rejects_missing_unreadable_and_wrong_stage_markers(out):
    assert markers.verify(out, "ptq", ctx()) == (False, "no marker")
    path = marker_path(out, "ptq")
    path.parent.mkdir(parents=True)
    path.write_text("{broken")
    assert markers.verify(out, "ptq", ctx())[1] == "marker unreadable"
    path.write_text(json.dumps({"stage": "freeze"}))
    assert markers.verify(out, "ptq", ctx())[1] == "marker names another stage"


def test_marker_run_id_mismatch_invalidates(out):
    f = make_output(out, "a.bin")
    markers.write_after_validate(out, "ptq", lambda: True, [f], ctx())
    other = markers.context("other-run", config_sha256=CFG, lock_sha256=LOCK, protocol_sha256=PROTO, epochs=10)
    ok, reason = markers.verify(out, "ptq", other)
    assert not ok and "run_id" in reason


# ---------------------------------------------------------------- resume planning
def fill(out, upto):
    for stage in STAGES[:upto]:
        f = make_output(out, "outputs/%s.bin" % stage, stage.encode())
        markers.write_after_validate(out, stage, lambda: True, [f], ctx(protocol=None if stage == "preflight" else PROTO))


def test_resume_plan_skips_valid_stages_and_returns_first_invalid(out):
    assert markers.first_invalid_stage(out, ctx()) == "preflight"
    fill(out, 4)
    assert markers.first_invalid_stage(out, ctx()) == STAGES[4]
    assert markers.valid_stages(out, ctx()) == list(STAGES[:4])
    (out / "outputs" / (STAGES[1] + ".bin")).write_bytes(b"tampered")
    assert markers.first_invalid_stage(out, ctx()) == STAGES[1]
    fill(out, len(STAGES))
    assert markers.first_invalid_stage(out, ctx()) is None


def test_clear_and_any_marker_exists(out):
    assert markers.any_marker_exists(out) is False
    f = make_output(out, "a.bin")
    markers.write_after_validate(out, "ptq", lambda: True, [f], ctx())
    assert markers.any_marker_exists(out) is True
    assert markers.clear(out, "ptq") is True and markers.clear(out, "ptq") is False
    assert markers.any_marker_exists(out) is False


def test_marker_writes_are_atomic_no_temp_files_left(out):
    f = make_output(out, "a.bin")
    markers.write_after_validate(out, "ptq", lambda: True, [f], ctx())
    assert [p.name for p in (out / "stage_markers").iterdir()] == ["ptq.ok"]


# ---------------------------------------------------------------- deadline_meta
def make_ckpt(out, model_id, data=b"weights"):
    path = checkpoint_path(out, model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_deadline_meta_roundtrip_and_flags(out):
    make_ckpt(out, "V3")
    meta = markers.write_deadline_meta(out, "V3", config_sha256=CFG, epochs=10)
    assert meta["adversarial"] is True and meta["model_id"] == "V3" and meta["kind"] == "deadline-real"
    assert markers.verify_deadline_meta(out, "V3", config_sha256=CFG, epochs=10)["checkpoint_sha256"] == meta["checkpoint_sha256"]
    make_ckpt(out, "V1")
    assert markers.write_deadline_meta(out, "V1", config_sha256=CFG, epochs=10)["adversarial"] is False
    make_ckpt(out, "V2")
    assert checkpoint_path(out, "V2").name == "state.pt"


@pytest.mark.parametrize("kwargs", [dict(config_sha256="d" * 64, epochs=10), dict(config_sha256=CFG, epochs=5)])
def test_deadline_meta_rejects_other_config_or_epochs(out, kwargs):
    make_ckpt(out, "V1")
    markers.write_deadline_meta(out, "V1", config_sha256=CFG, epochs=10)
    with pytest.raises(GuardError):
        markers.verify_deadline_meta(out, "V1", **kwargs)


def test_deadline_meta_rejects_changed_checkpoint_missing_meta_and_foreign_kind(out):
    make_ckpt(out, "V1")
    with pytest.raises(GuardError, match="missing"):
        markers.verify_deadline_meta(out, "V1", config_sha256=CFG, epochs=10)
    markers.write_deadline_meta(out, "V1", config_sha256=CFG, epochs=10)
    make_ckpt(out, "V1", b"other weights")
    with pytest.raises(GuardError, match="does not match"):
        markers.verify_deadline_meta(out, "V1", config_sha256=CFG, epochs=10)
    make_ckpt(out, "V1")
    meta_path = model_dir(out, "V1") / "deadline_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["kind"] = "smoke"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(GuardError, match="kind"):
        markers.verify_deadline_meta(out, "V1", config_sha256=CFG, epochs=10)


def test_write_deadline_meta_requires_the_checkpoint(out):
    with pytest.raises(GuardError, match="missing"):
        markers.write_deadline_meta(out, "V1", config_sha256=CFG, epochs=10)


# ---------------------------------------------------------------- run_identity
def fake_v4(tmp_path):
    research = tmp_path / "v4" / "research"
    research.mkdir(parents=True)
    for name in markers.SOURCE_FILES:
        (research / name).write_text("# %s\n" % name)
    return tmp_path / "v4"


def identity(v4_root, **over):
    kw = dict(
        config_sha256=CFG, lock_sha256=LOCK, epochs=10, protocol_sha256=PROTO,
        v4_manifest_sha256="v" * 64, source_sha256=markers.source_hashes(v4_root),
    )
    kw.update(over)
    return markers.build_run_identity(**kw)


def test_source_hashes_cover_the_expected_files(tmp_path):
    root = fake_v4(tmp_path)
    assert set(markers.source_hashes(root)) == set(markers.SOURCE_FILES)
    assert len(markers.SOURCE_FILES) == 7


def test_run_identity_is_written_once_and_verified(out, tmp_path):
    root = fake_v4(tmp_path)
    ident = identity(root)
    assert markers.write_run_identity(out, ident) is True
    assert markers.write_run_identity(out, identity(root)) is False  # same content, new created_utc: kept
    got = markers.verify_run_identity(
        out, config_sha256=CFG, lock_sha256=LOCK, epochs=10, v4_root=root, v4_manifest_sha256="v" * 64
    )
    assert got["kind"] == "deadline-real"
    with pytest.raises(GuardError, match="different content"):
        markers.write_run_identity(out, identity(root, epochs=5))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(epochs=5), "epochs"),
        (dict(config_sha256="d" * 64), "config_sha256"),
        (dict(lock_sha256="m" * 64), "lock_sha256"),
        (dict(v4_manifest_sha256="w" * 64), "MANIFEST"),
    ],
)
def test_run_identity_mismatches_exit_3_class(out, tmp_path, kwargs, match):
    root = fake_v4(tmp_path)
    markers.write_run_identity(out, identity(root))
    base = dict(config_sha256=CFG, lock_sha256=LOCK, epochs=10, v4_root=root, v4_manifest_sha256="v" * 64)
    base.update(kwargs)
    with pytest.raises(GuardError, match=match) as err:
        markers.verify_run_identity(out, **base)
    assert err.value.exit_code == 3


def test_run_identity_detects_changed_v4_source_and_missing_file(out, tmp_path):
    root = fake_v4(tmp_path)
    with pytest.raises(GuardError, match="missing"):
        markers.verify_run_identity(out, config_sha256=CFG, lock_sha256=LOCK, epochs=10)
    markers.write_run_identity(out, identity(root))
    (root / "research" / "m2_training.py").write_text("# edited\n")
    with pytest.raises(GuardError, match="m2_training.py"):
        markers.verify_run_identity(out, config_sha256=CFG, lock_sha256=LOCK, epochs=10, v4_root=root)


def test_contract_required_keys_are_present_in_what_markers_write(out, tmp_path):
    from m2_deadline.common import JSON_REQUIRED_KEYS

    f = make_output(out, "a.bin")
    m = markers.write_after_validate(out, "ptq", lambda: True, [f], ctx())
    assert set(JSON_REQUIRED_KEYS["marker"]) <= set(m)
    make_ckpt(out, "V1")
    assert set(JSON_REQUIRED_KEYS["deadline_meta"]) <= set(markers.write_deadline_meta(out, "V1", config_sha256=CFG, epochs=10))
    root = fake_v4(tmp_path)
    assert set(JSON_REQUIRED_KEYS["run_identity"]) <= set(identity(root))
    assert out_path(out, "run_identity").name == "run_identity.json"
