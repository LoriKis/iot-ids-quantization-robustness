"""Stage call-order tests with stub deps: test-set isolation, skip/quarantine logic, marker discipline."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from stub_support import make_ctx, preload_models

from m2_deadline import markers, protocol, sampling, stages
from m2_deadline.common import (
    ATTACK_UNITS,
    CSV_COLUMNS,
    GuardError,
    MODEL_IDS,
    ProtocolChangedError,
    check_required_keys,
    checkpoint_path,
    marker_path,
    model_dir,
    out_path,
    read_json,
    sha256_file,
    unit_dir,
    unit_id,
)


@pytest.fixture
def ctx(tmp_path):
    return make_ctx(tmp_path)


def freeze(ctx):
    stages.run_stage(ctx, "freeze")


def marker_exists(ctx, stage):
    return marker_path(ctx.out, stage).is_file()


# ---------------------------------------------------------------- freeze
def test_freeze_never_loads_test_arrays_nor_verifies_the_freeze(ctx):
    freeze(ctx)
    d = ctx.deps.log
    assert d.of("load_arrays") == [] and d.of("verify_eval_freeze") == []
    assert marker_exists(ctx, "freeze")
    for key in ("protocol", "protocol_sha256", "subset_ids", "subset_counts", "protocol_frozen", "run_identity"):
        assert out_path(ctx.out, key).is_file(), key
    proto = read_json(out_path(ctx.out, "protocol"))
    assert proto["epochs"] == 10 and proto["robustness"]["per_class_target"] == 10
    ident = read_json(out_path(ctx.out, "run_identity"))
    check_required_keys(ident, "run_identity")
    assert ident["protocol_sha256"] == sha256_file(out_path(ctx.out, "protocol"))
    assert ident["epochs"] == 10


def test_freeze_is_idempotent_and_refuses_changed_content(ctx):
    freeze(ctx)
    first = out_path(ctx.out, "protocol").read_bytes()
    stages.run_stage(ctx, "freeze")  # same inputs: validates and rewrites only the marker
    assert out_path(ctx.out, "protocol").read_bytes() == first
    ctx.lock = dict(ctx.lock, per_class_target=5)  # would freeze another subset
    with pytest.raises(GuardError):
        stages.run_stage(ctx, "freeze")
    assert out_path(ctx.out, "protocol").read_bytes() == first
    assert not marker_exists(ctx, "freeze")


def test_freeze_subset_uses_labels_from_dataset_only(ctx):
    freeze(ctx)
    ids, labels = protocol.parse_ids_csv(out_path(ctx.out, "subset_ids").read_bytes())
    test_ids, y = protocol.read_test_labels(ctx.split_doc(), None, ctx.v4_root / ctx.split_doc()["dataset"]["path"])
    assert set(ids.tolist()) <= set(test_ids.tolist())
    assert len(ids) == 27 * 10
    lookup = dict(zip(test_ids.tolist(), y.tolist()))
    assert [lookup[int(i)] for i in ids] == labels.tolist()


# ---------------------------------------------------------------- test_clean (the only guarded test load)
def test_test_clean_is_the_only_caller_of_verify_eval_freeze_and_test_loader(ctx):
    freeze(ctx)
    preload_models(ctx)
    stages.run_stage(ctx, "test_clean")
    calls = ctx.deps.log
    assert len(calls.of("verify_eval_freeze")) == 1
    loads = calls.of("load_arrays")
    assert [c["partitions"] for c in loads] == [("test",)]
    assert loads[0]["purpose"] == "final_test" and loads[0]["freeze"] is not None
    # robustness takes its rows from the file test_clean wrote; it neither verifies the freeze nor loads arrays
    stages.run_stage(ctx, "robustness")
    assert len(calls.of("verify_eval_freeze")) == 1
    assert [c["partitions"] for c in calls.of("load_arrays")] == [("test",)]
    # the call order inside test_clean: verify, then load
    names = calls.names()
    assert names.index("verify_eval_freeze") < names.index("load_arrays")


def test_other_stage_functions_never_call_verify_eval_freeze():
    import inspect

    for name, func in stages.STAGE_FUNCS.items():
        src = inspect.getsource(func)
        assert ("verify_eval_freeze" in src) == (name == "test_clean"), name
        assert ('purpose="final_test"' in src) == (name == "test_clean"), name


def test_test_clean_outputs(ctx):
    freeze(ctx)
    preload_models(ctx)
    stages.run_stage(ctx, "test_clean")
    results = read_json(out_path(ctx.out, "clean_results"))
    check_required_keys(results, "clean_results")
    n_test = len(ctx.split_doc()["row_ids"]["test"])
    assert results["n_test"] == n_test and results["batch_size"] == 64
    with open(str(out_path(ctx.out, "predictions_full_test")), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0].keys()) == CSV_COLUMNS["predictions_full_test"] and len(rows) == n_test
    assert sum(r["in_robust_subset"] == "true" for r in rows) == 27 * 10
    # accuracy equals the value recomputed from the CSV (V1 predicts exactly the labels)
    assert results["models"]["V1"]["accuracy"] == 1.0
    assert results["models"]["V4"]["accuracy"] < 1.0
    with np.load(str(out_path(ctx.out, "subset_rows"))) as rows_npz:
        ids, _ = protocol.parse_ids_csv(out_path(ctx.out, "subset_ids").read_bytes())
        assert rows_npz["ids"].tolist() == ids.tolist()
    assert marker_exists(ctx, "test_clean")


def test_test_clean_refuses_a_changed_protocol_before_touching_test_data(ctx):
    freeze(ctx)
    preload_models(ctx)
    path = out_path(ctx.out, "protocol")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ProtocolChangedError):
        stages.run_stage(ctx, "test_clean")
    assert ctx.deps.log.of("verify_eval_freeze") == [] and ctx.deps.log.of("load_arrays") == []
    assert not marker_exists(ctx, "test_clean")


# ---------------------------------------------------------------- robustness
def run_to_robustness(ctx):
    freeze(ctx)
    preload_models(ctx)
    stages.run_stage(ctx, "test_clean")
    stages.run_stage(ctx, "robustness")


def test_robustness_runs_20_units_with_the_frozen_subset(ctx):
    run_to_robustness(ctx)
    calls = ctx.deps.log.of("evaluate_unit")
    assert len(calls) == 20
    subset_hash = read_json(out_path(ctx.out, "protocol"))["robustness"]["subset_ids_sha256"]
    assert {c["expected_ids_sha256"] for c in calls} == {subset_hash}
    assert {c["evaluation_partition"] for c in calls} == {"test"}
    by_model = {}
    for c in calls:
        by_model.setdefault(c["model_id"], []).append((c["attack_name"], c["epsilon"]))
    assert set(by_model) == set(MODEL_IDS)
    for model, units in by_model.items():
        assert units == [("fgsm", 0.0), ("fgsm", 0.05), ("fgsm", 0.1), ("pgd5", 0.05), ("pgd5", 0.1)]
    assert {c["source"] for c in calls if c["model_id"] in ("V1", "V3")} == {"fp32_direct"}
    assert {c["source"] for c in calls if c["model_id"] in ("V2", "V4")} == {"bpda_int8"}
    assert all(c["notes"] for c in calls if c["model_id"] in ("V2", "V4"))
    results = read_json(out_path(ctx.out, "robustness_results"))
    check_required_keys(results, "robustness_results")
    for m in MODEL_IDS:
        assert sorted(results["units"][m]) == sorted(unit_id(a, e) for a, e in ATTACK_UNITS)
        assert {"eps0_vs_full_test_mismatches", "eps0_max_abs_logit_diff", "n_compared"} <= set(results["eps0_consistency"][m])
    with open(str(out_path(ctx.out, "predictions_subset")), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20 * results["n"] and {r["epsilon"] for r in rows} == {"0.0", "0.05", "0.1"}
    assert marker_exists(ctx, "robustness")


def test_eps0_mismatch_is_recorded_and_never_fails_the_run(ctx):
    preload_models(ctx, shifts={"V1": 0, "V2": 1, "V3": 2, "V4": 3})
    freeze(ctx)
    stages.run_stage(ctx, "test_clean")  # full-test predictions of V2..V4 differ from the labels on some rows
    stages.run_stage(ctx, "robustness")  # the stub eps-0 predictions equal the labels: mismatches are recorded
    e0 = read_json(out_path(ctx.out, "robustness_results"))["eps0_consistency"]
    assert e0["V1"]["eps0_vs_full_test_mismatches"] == 0
    assert e0["V4"]["eps0_vs_full_test_mismatches"] > 0 and marker_exists(ctx, "robustness")


def test_rerun_skips_finished_units_and_quarantines_incomplete_ones(ctx):
    run_to_robustness(ctx)
    n_calls = len(ctx.deps.log.of("evaluate_unit"))
    stages.run_stage(ctx, "robustness")
    assert len(ctx.deps.log.of("evaluate_unit")) == n_calls  # every unit_done validated: nothing reran
    victim = unit_dir(ctx.out, "V3", "pgd5", 0.05)
    (victim / "unit_done.json").unlink()
    stages.run_stage(ctx, "robustness")
    assert len(ctx.deps.log.of("evaluate_unit")) == n_calls + 1
    quarantined = list((out_path(ctx.out, "robustness_dir") / "_incomplete").rglob("attack_checks.json"))
    assert len(quarantined) == 1
    assert (victim / "unit_done.json").is_file()


def test_tampered_unit_files_invalidate_unit_done(ctx):
    run_to_robustness(ctx)
    n_calls = len(ctx.deps.log.of("evaluate_unit"))
    target = unit_dir(ctx.out, "V1", "fgsm", 0.1) / "arrays.npz"
    target.write_bytes(target.read_bytes() + b"x")
    stages.run_stage(ctx, "robustness")
    assert len(ctx.deps.log.of("evaluate_unit")) == n_calls + 1


def test_id_hash_mismatch_exits_before_any_attack(ctx):
    freeze(ctx)
    preload_models(ctx)
    stages.run_stage(ctx, "test_clean")
    ctx.deps.id_hash_override = "0" * 64
    with pytest.raises(GuardError):
        stages.run_stage(ctx, "robustness")
    assert ctx.deps.log.of("evaluate_unit") == []
    assert not marker_exists(ctx, "robustness")


def test_invalid_unit_fails_the_stage_without_marker(ctx):
    freeze(ctx)
    preload_models(ctx)
    stages.run_stage(ctx, "test_clean")
    ctx.deps.invalid_unit = ("V2", "pgd5", 0.1)
    with pytest.raises(stages.GateFailedError) as err:
        stages.run_stage(ctx, "robustness")
    assert err.value.exit_code == 4
    assert not marker_exists(ctx, "robustness")
    assert not (unit_dir(ctx.out, "V2", "pgd5", 0.1) / "unit_done.json").exists()


# ---------------------------------------------------------------- training
def test_train_stage_calls_train_unit_with_matched_arguments(ctx):
    freeze(ctx)
    stages.run_stage(ctx, "train_ordinary")
    stages.run_stage(ctx, "train_adversarial")
    calls = ctx.deps.log.of("train_unit")
    assert [c["adversarial"] for c in calls] == [False, True]
    for c, model in zip(calls, ("V1", "V3")):
        assert c["epochs"] == 10 and c["seed"] == 1 and c["threads"] == 2 and c["keep_last"] == 2
        assert Path(c["out_dir"]) == model_dir(ctx.out, model)
        assert Path(c["checkpoint_dir"]) == model_dir(ctx.out, model) / "ckpt"
        assert c["fingerprints"]["kind"] == "deadline-real" and c["fingerprints"]["epochs"] == 10
        assert c["stop_flag"] is ctx.stop_flag
    assert calls[0]["fingerprints"]["adversarial"] is False and calls[1]["fingerprints"]["adversarial"] is True
    for model in ("V1", "V3"):
        meta = markers.verify_deadline_meta(ctx.out, model, config_sha256=ctx.config_sha256, epochs=10)
        assert meta["extra"]["optimizer_updates"] == 7
    assert marker_exists(ctx, "train_ordinary") and marker_exists(ctx, "train_adversarial")


def test_train_stage_skips_train_unit_when_outputs_exist(ctx):
    freeze(ctx)
    stages.run_stage(ctx, "train_ordinary")
    assert len(ctx.deps.log.of("train_unit")) == 1
    marker_path(ctx.out, "train_ordinary").unlink()  # crash between the outputs and the marker
    stages.run_stage(ctx, "train_ordinary")
    assert len(ctx.deps.log.of("train_unit")) == 1  # validated only
    assert marker_exists(ctx, "train_ordinary")


def test_train_stage_quarantines_a_lone_checkpoint_and_retrains(ctx):
    freeze(ctx)
    mdir = model_dir(ctx.out, "V1")
    mdir.mkdir(parents=True)
    (mdir / "checkpoint.pt").write_bytes(b"half written")
    stages.run_stage(ctx, "train_ordinary")
    assert len(ctx.deps.log.of("train_unit")) == 1
    assert list((mdir / "_incomplete").rglob("checkpoint.pt"))


def test_train_stage_rejects_wrong_epoch_count(tmp_path):
    ctx = make_ctx(tmp_path, epochs_in_log=7)
    freeze(ctx)
    with pytest.raises(GuardError, match="epochs"):
        stages.run_stage(ctx, "train_ordinary")
    assert not marker_exists(ctx, "train_ordinary")


def test_train_stage_rejects_unpaired_adversarial_parent(ctx):
    freeze(ctx)
    stages.run_stage(ctx, "train_ordinary")
    log = read_json(model_dir(ctx.out, "V1") / "train_log.json")
    log["init_state_hash"] = "other"
    (model_dir(ctx.out, "V1") / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    with pytest.raises(GuardError, match="pair check"):
        stages.run_stage(ctx, "train_adversarial")


def test_existing_outputs_from_another_run_are_refused(ctx):
    freeze(ctx)
    stages.run_stage(ctx, "train_ordinary")
    meta = model_dir(ctx.out, "V1") / "deadline_meta.json"
    doc = read_json(meta)
    doc["config_sha256"] = "f" * 64
    meta.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(GuardError):
        stages.run_stage(ctx, "train_ordinary")


def test_stop_request_inside_training_becomes_exit_130(ctx):
    freeze(ctx)

    def interrupted(*a, **k):
        raise ctx.deps.mt.TrainingInterrupted("stopped after epoch 3 of 10")

    ctx.deps.mt.train_unit = interrupted
    with pytest.raises(stages.StageInterrupted) as err:
        stages.run_stage(ctx, "train_ordinary")
    assert err.value.exit_code == 130 and not marker_exists(ctx, "train_ordinary")


# ---------------------------------------------------------------- ptq, val_check, efficiency, preflight
def test_ptq_uses_train_only_calibration_and_writes_meta(ctx):
    freeze(ctx)
    stages.run_stage(ctx, "train_ordinary")
    stages.run_stage(ctx, "train_adversarial")
    preload_models(ctx, write_conversion=False)
    stages.run_stage(ctx, "ptq")
    assert ctx.deps.log.of("calibration_ids")[0]["per_class"] == 20
    assert [c["variant"] for c in ctx.deps.log.of("ptq_unit")] == ["V2", "V4"]
    for m in ("V2", "V4"):
        meta = markers.verify_deadline_meta(ctx.out, m, config_sha256=ctx.config_sha256, epochs=10)
        assert meta["extra"]["parent_checkpoint_sha256"] == sha256_file(checkpoint_path(ctx.out, "V1" if m == "V2" else "V3"))
    stages.run_stage(ctx, "ptq")  # second pass validates without converting again
    assert len(ctx.deps.log.of("ptq_unit")) == 2


def test_ptq_refuses_calibration_outside_train(ctx):
    freeze(ctx)
    stages.run_stage(ctx, "train_ordinary")
    stages.run_stage(ctx, "train_adversarial")
    preload_models(ctx, write_conversion=False)
    ctx.deps.calibration_outside_train = True
    with pytest.raises(GuardError, match="train partition"):
        stages.run_stage(ctx, "ptq")
    assert ctx.deps.log.of("ptq_unit") == []


def test_ptq_wrong_island_count_is_refused(tmp_path):
    ctx = make_ctx(tmp_path, quantized=22)
    freeze(ctx)
    stages.run_stage(ctx, "train_ordinary")
    stages.run_stage(ctx, "train_adversarial")
    preload_models(ctx, write_conversion=False)
    with pytest.raises(GuardError, match="quantized linears"):
        stages.run_stage(ctx, "ptq")
    assert not marker_exists(ctx, "ptq")


def test_val_check_computes_no_accuracy_and_blocks_on_bad_logits(ctx):
    freeze(ctx)
    preload_models(ctx)
    for m in ("V2", "V4"):
        conv = {"requested": ["l%d" % i for i in range(23)], "reload_check": {"logits_bitwise_equal": True, "argmax_identical": True}}
        (model_dir(ctx.out, m) / "conversion.json").write_text(json.dumps(conv), encoding="utf-8")
    stages.run_stage(ctx, "val_check")
    doc = read_json(out_path(ctx.out, "val_check"))
    assert doc["passed"] is True

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield str(k).lower()
                yield from keys(v)

    assert not [k for k in keys(doc) if "accuracy" in k or "f1" in k or "loss" in k]
    check_required_keys(read_json(out_path(ctx.out, "val_check")), "val_check")
    # a failing gate writes the file, raises exit 4 and writes no marker
    ctx.deps.logits_nonfinite = True
    ctx._cache["model_V2"].tag = "bad"
    with pytest.raises(stages.GateFailedError) as err:
        stages.run_stage(ctx, "val_check")
    assert err.value.exit_code == 4 and not marker_exists(ctx, "val_check")
    assert read_json(out_path(ctx.out, "val_check"))["passed"] is False


def test_efficiency_uses_one_thread_not_pilot_and_reuses_a_valid_record(ctx):
    freeze(ctx)
    preload_models(ctx)
    # create real checkpoints + meta so verify_deadline_meta passes
    for m in MODEL_IDS:
        mdir = model_dir(ctx.out, m)
        mdir.mkdir(parents=True, exist_ok=True)
        checkpoint_path(ctx.out, m).write_bytes(b"x" + m.encode())
        markers.write_deadline_meta(ctx.out, m, config_sha256=ctx.config_sha256, epochs=ctx.epochs)
        (mdir / "conversion.json").write_text(json.dumps({"requested": ["a"]}), encoding="utf-8")
    stages.run_stage(ctx, "efficiency")
    call = ctx.deps.log.of("run_efficiency")[0]
    assert call["pilot"] is False and call["threads"] == 1 and call["models"] == ["V1", "V2", "V3", "V4"]
    assert Path(call["out_dir"]) == out_path(ctx.out, "efficiency_dir")
    assert marker_exists(ctx, "efficiency")
    stages.run_stage(ctx, "efficiency")
    assert len(ctx.deps.log.of("run_efficiency")) == 1


def test_efficiency_with_failed_measurement_has_no_marker(ctx):
    freeze(ctx)
    preload_models(ctx)
    for m in MODEL_IDS:
        mdir = model_dir(ctx.out, m)
        mdir.mkdir(parents=True, exist_ok=True)
        checkpoint_path(ctx.out, m).write_bytes(b"x" + m.encode())
        markers.write_deadline_meta(ctx.out, m, config_sha256=ctx.config_sha256, epochs=ctx.epochs)
    ctx.deps.efficiency_failed = True
    with pytest.raises(stages.GateFailedError):
        stages.run_stage(ctx, "efficiency")
    assert not marker_exists(ctx, "efficiency")


def test_preflight_checks_lock_files_and_overlay_verification(tmp_path):
    ctx = make_ctx(tmp_path)
    # hand-built lock/estimate are not on disk: preflight must refuse
    with pytest.raises(GuardError):
        stages.run_stage(ctx, "preflight")
    assert not marker_exists(ctx, "preflight")


# ---------------------------------------------------------------- marker discipline
def test_marker_is_written_only_after_validation_passes(ctx, monkeypatch):
    def bad_stage(c):
        return {"outputs": [], "validate": lambda: False}

    monkeypatch.setitem(stages.STAGE_FUNCS, "preflight", bad_stage)
    with pytest.raises(GuardError):
        stages.run_stage(ctx, "preflight")
    assert not marker_exists(ctx, "preflight")

    def raising(c):
        def validate():
            raise RuntimeError("boom")

        return {"outputs": [], "validate": validate}

    monkeypatch.setitem(stages.STAGE_FUNCS, "preflight", raising)
    with pytest.raises(RuntimeError):
        stages.run_stage(ctx, "preflight")
    assert not marker_exists(ctx, "preflight")


def test_run_stage_clears_a_stale_marker_before_starting(ctx, monkeypatch):
    freeze(ctx)
    assert marker_exists(ctx, "freeze")
    seen = []

    def spy(c):
        seen.append(marker_exists(c, "freeze"))
        raise RuntimeError("fail after start")

    monkeypatch.setitem(stages.STAGE_FUNCS, "freeze", spy)
    with pytest.raises(RuntimeError):
        stages.run_stage(ctx, "freeze")
    assert seen == [False] and not marker_exists(ctx, "freeze")


def test_only_run_stage_writes_markers():
    import inspect

    src = inspect.getsource(stages)
    assert src.count("markers.write_after_validate(") == 1
    assert "marker_path(" not in src.replace("def marker_path", "")


def test_smoke_paths_are_refused(tmp_path):
    ctx = make_ctx(tmp_path)
    bad = tmp_path / "output" / "m2b-v1-smoke" / "x"
    ctx.out = bad
    with pytest.raises(GuardError, match="smoke"):
        stages.run_stage(ctx, "freeze")
    assert not bad.exists()
