"""Campaign launch/resume/status guards and the stage loop, with a stub stage runner (no stage does real work)."""
from __future__ import annotations

import json
import shutil

import pytest

from conftest import CONFIG_PATH
from stub_support import make_fake_v4, make_locked_run

from m2_deadline import budget, campaign, markers, stages
from m2_deadline.common import (
    STAGES,
    atomic_write_json,
    marker_path,
    model_dir,
    out_path,
    read_json,
    sha256_file,
)


@pytest.fixture
def run(tmp_path):
    return make_locked_run(tmp_path)


def argv(run, hours, cmd="launch", *extra, out=None, config=None):
    a = [cmd, "--out", str(out or run.out), "--v4-root", str(run.v4), "--config", str(config or CONFIG_PATH)]
    if hours is not None:
        a += ["--hours-remaining", str(hours)]
    return a + list(extra)


def make_runner(order):
    def runner(ctx, stage):
        if stage == "freeze":
            path = out_path(ctx.out, "protocol")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"{}")
            ctx.refresh_protocol_sha()
        markers.write_after_validate(ctx.out, stage, lambda: True, [], ctx.marker_ctx())
        order.append(stage)

    return runner


def go(run, hours, cmd="launch", *extra, runner=None, **kw):
    return campaign.main(argv(run, hours, cmd, *extra, **kw), stage_runner=runner or make_runner([]), install_signals=False)


def lock_epochs(run):
    return read_json(out_path(run.out, "budget_lock"))["epochs"]


# ---------------------------------------------------------------- exit 2: usage and missing inputs
@pytest.mark.parametrize("cmd", ["launch", "resume"])
@pytest.mark.parametrize("hours", [None, 0, -2, 13])
def test_hours_are_required_and_validated(run, cmd, hours):
    order = []
    assert go(run, hours, cmd, runner=make_runner(order)) == 2
    assert order == [] and not out_path(run.out, "markers_dir").exists()


def test_launch_without_a_budget_lock_exits_2(tmp_path, capsys):
    v4, _ = make_fake_v4(tmp_path / "v4")
    out = tmp_path / "artifacts" / "m2-deadline-v1"
    code = campaign.main(["launch", "--hours-remaining", "11", "--out", str(out), "--v4-root", str(v4)], install_signals=False)
    assert code == 2
    assert "timing" in capsys.readouterr().err
    assert not out.exists()


def test_there_is_no_epochs_flag(run):
    with pytest.raises(SystemExit) as err:
        campaign.main(argv(run, 11, "launch", "--epochs", "5"), install_signals=False)
    assert err.value.code == 2


def test_lock_with_a_disallowed_epoch_count_exits_2(run):
    lock = read_json(out_path(run.out, "budget_lock"))
    lock["epochs"] = 7
    atomic_write_json(out_path(run.out, "budget_lock"), lock)
    assert go(run, 11) == 2


def test_resume_on_an_empty_out_exits_2(run, capsys):
    assert go(run, 11, "resume") == 2
    assert "nothing to resume" in capsys.readouterr().err


def test_launch_refuses_an_existing_run(run):
    order = []
    assert go(run, 11, "launch", "--stop-after", "preflight", runner=make_runner(order)) == 0
    assert go(run, 11, "launch", runner=make_runner(order)) == 2
    assert order == ["preflight"]


# ---------------------------------------------------------------- exit 3: guards
def test_estimate_edited_after_locking_exits_3(run):
    est = read_json(out_path(run.out, "timing_estimate"))
    est["host"]["cpu_count"] = 999
    atomic_write_json(out_path(run.out, "timing_estimate"), est)
    assert go(run, 11) == 3


def test_config_change_after_locking_exits_3(run, tmp_path):
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["claim"] = cfg["claim"] + " Changed."
    other = tmp_path / "other_config.json"
    other.write_text(json.dumps(cfg), encoding="utf-8")
    assert go(run, 11, config=other) == 3


def test_lock_that_inspected_performance_exits_3(run):
    lock = read_json(out_path(run.out, "budget_lock"))
    lock["performance_inspected"] = True
    atomic_write_json(out_path(run.out, "budget_lock"), lock)
    assert go(run, 11) == 3


def test_v4_manifest_change_exits_3(run):
    (run.v4 / "MANIFEST.sha256").write_text("1" * 64 + "  other\n", encoding="utf-8")
    assert go(run, 11) == 3


@pytest.mark.parametrize("cmd", ["launch", "resume"])
def test_run_identity_with_other_epochs_exits_3(run, cmd):
    lock = read_json(out_path(run.out, "budget_lock"))
    ident = markers.build_run_identity(
        config_sha256=sha256_file(CONFIG_PATH), lock_sha256=sha256_file(out_path(run.out, "budget_lock")),
        epochs=lock["epochs"] + 1, protocol_sha256=None,
        v4_manifest_sha256=sha256_file(run.v4 / "MANIFEST.sha256"), source_sha256=markers.source_hashes(run.v4),
    )
    markers.write_run_identity(run.out, ident)
    order = []
    assert go(run, 11, cmd, runner=make_runner(order)) == 3
    assert order == []


def test_changed_v4_sources_on_resume_exit_3(run):
    lock = read_json(out_path(run.out, "budget_lock"))
    ident = markers.build_run_identity(
        config_sha256=sha256_file(CONFIG_PATH), lock_sha256=sha256_file(out_path(run.out, "budget_lock")),
        epochs=lock["epochs"], protocol_sha256=None,
        v4_manifest_sha256=sha256_file(run.v4 / "MANIFEST.sha256"), source_sha256=markers.source_hashes(run.v4),
    )
    markers.write_run_identity(run.out, ident)
    (run.v4 / "research" / "m2_training.py").write_text("# edited\n", encoding="utf-8")
    assert go(run, 11, "resume") == 3


def test_smoke_out_path_exits_3(run, tmp_path):
    assert go(run, 11, out=tmp_path / "output" / "m2b-v1-smoke") == 3


def test_out_under_v4_output_exits_3(run):
    assert go(run, 11, out=run.v4 / "output" / "m2-deadline") == 3


# ---------------------------------------------------------------- exit 4: deadline checks
def test_launch_full_plan_check_exits_4_and_advises_force_relock(run, capsys):
    lock = read_json(out_path(run.out, "budget_lock"))
    hours = lock["planned_hours"] + 3.0 - 0.2  # H - 3 < planned
    order = []
    assert go(run, hours, runner=make_runner(order)) == 4
    err = capsys.readouterr().err
    assert "force-relock" in err and order == []


def test_launch_with_enough_hours_runs_every_stage_in_order(run):
    order = []
    assert go(run, 11, runner=make_runner(order)) == 0
    assert order == list(STAGES)
    ctx = markers.context(config_sha256=sha256_file(CONFIG_PATH), lock_sha256=sha256_file(out_path(run.out, "budget_lock")),
                          protocol_sha256=sha256_file(out_path(run.out, "protocol")), epochs=lock_epochs(run))
    assert markers.first_invalid_stage(run.out, ctx) is None


def test_resume_remaining_plan_check_exits_4_without_relock_advice(run, capsys):
    order = []
    assert go(run, 11, "launch", "--stop-after", "preflight", runner=make_runner(order)) == 0
    est = read_json(out_path(run.out, "timing_estimate"))
    lock = read_json(out_path(run.out, "budget_lock"))
    ctx = markers.context(config_sha256=sha256_file(CONFIG_PATH), lock_sha256=sha256_file(out_path(run.out, "budget_lock")),
                          protocol_sha256=None, epochs=lock["epochs"])
    remaining = budget.resume_check(est, lock, markers.valid_stages(run.out, ctx), 11.0)["remaining_planned_hours"]
    tight = remaining / 1.0 + 3.0 - 0.1
    capsys.readouterr()
    assert go(run, tight, "resume", runner=make_runner(order)) == 4
    err = capsys.readouterr().err
    assert "force-relock" not in err and "remaining plan" in err
    assert order == ["preflight"]


# ---------------------------------------------------------------- the stage loop
def test_stop_after_then_resume_continues_at_the_next_stage(run):
    order = []
    assert go(run, 11, "launch", "--stop-after", "ptq", runner=make_runner(order)) == 0
    assert order == ["preflight", "freeze", "train_ordinary", "train_adversarial", "ptq"]
    order2 = []
    assert go(run, 11, "resume", runner=make_runner(order2)) == 0
    assert order2 == ["val_check", "test_clean", "robustness", "efficiency", "report"]
    order3 = []
    assert go(run, 11, "resume", runner=make_runner(order3)) == 0  # everything valid: nothing to do
    assert order3 == []


def test_stop_after_an_already_valid_stage_does_nothing(run):
    order = []
    assert go(run, 11, "launch", "--stop-after", "ptq", runner=make_runner(order)) == 0
    order2 = []
    assert go(run, 11, "resume", "--stop-after", "freeze", runner=make_runner(order2)) == 0
    assert order2 == []


def test_resume_reruns_from_the_first_invalid_marker(run):
    order = []
    assert go(run, 11, "launch", "--stop-after", "train_adversarial", runner=make_runner(order)) == 0
    marker_path(run.out, "train_ordinary").write_text("{}", encoding="utf-8")  # corrupt marker is invalid, not skipped
    order2 = []
    assert go(run, 11, "resume", "--stop-after", "train_adversarial", runner=make_runner(order2)) == 0
    assert order2 == ["train_ordinary", "train_adversarial"]


def test_stop_flag_stops_cleanly_with_exit_130(run):
    order = []
    base = make_runner(order)

    def runner(ctx, stage):
        base(ctx, stage)
        if stage == "freeze":
            ctx.stop_flag.requested = True

    assert go(run, 11, runner=runner) == 130
    assert order == ["preflight", "freeze"]
    order2 = []
    assert go(run, 11, "resume", runner=make_runner(order2)) == 0
    assert order2[0] == "train_ordinary"


def test_stage_interrupted_maps_to_exit_130(run):
    def raising(exc):
        def runner(ctx, stage):
            raise exc

        return runner

    assert go(run, 11, runner=raising(stages.StageInterrupted("stop"))) == 130


def test_gate_failure_exit_code_is_4(run):
    def runner(ctx, stage):
        raise stages.GateFailedError("blocking gate failed")

    assert go(run, 11, runner=runner) == 4


def test_unexpected_error_exits_1(run):
    def runner(ctx, stage):
        raise RuntimeError("boom")

    assert go(run, 11, runner=runner) == 1


def test_training_progress_reads_the_checkpoint_index(run):
    ckpt = model_dir(run.out, "V1") / "ckpt"
    ckpt.mkdir(parents=True)
    (ckpt / "latest.json").write_text(json.dumps({"epoch": 6}), encoding="utf-8")
    assert campaign.training_progress(run.out) == {"train_ordinary": 6, "train_adversarial": 0}


def test_resume_counts_only_unfinished_epochs_of_the_training_stage(run):
    order = []
    assert go(run, 11, "launch", "--stop-after", "freeze", runner=make_runner(order)) == 0
    est = read_json(out_path(run.out, "timing_estimate"))
    lock = read_json(out_path(run.out, "budget_lock"))
    valid = ["preflight", "freeze"]
    full = budget.resume_check(est, lock, valid, 11.0)["breakdown_s"]["train_ordinary"]
    half = budget.resume_check(est, lock, valid, 11.0, {"train_ordinary": lock["epochs"] // 2})["breakdown_s"]["train_ordinary"]
    assert half < full


def test_status_lists_every_stage(run, capsys):
    order = []
    assert go(run, 11, "launch", "--stop-after", "freeze", runner=make_runner(order)) == 0
    capsys.readouterr()
    assert campaign.main(["status", "--out", str(run.out), "--v4-root", str(run.v4)], install_signals=False) == 0
    text = capsys.readouterr().out
    for stage in STAGES:
        assert stage in text
    assert "locked plan" in text


def test_status_without_a_lock_is_harmless(tmp_path, capsys):
    out = tmp_path / "artifacts" / "m2-deadline-v1"
    assert campaign.main(["status", "--out", str(out)], install_signals=False) == 0
    assert "timing command first" in capsys.readouterr().out
