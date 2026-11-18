"""Scheduler tests with stub executors (workers=1, in process): no torch compute, no data, no training."""
import json

import pytest
from m2b_stub_support import BASE_SEED1_STAGES, cli, run_dir, snapshot, stub_executors, unit_record

from research import m2b_plan as plan_mod
from research.m2b_campaign import load_campaign

CFG = None


def cfg():
    global CFG
    CFG = CFG or load_campaign("configs/m2b/campaign_v2.json")
    return CFG


EARLY = ["train", "calib", "paircheck", "ptq", "clean"]


def test_units_run_in_topological_order(tmp_path):
    stubs = stub_executors()
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=stubs) == 0
    calls = stubs["train"].calls
    units = plan_mod.build_plan(cfg(), [1], "validation", EARLY)
    assert sorted(calls) == sorted(u.id for u in units) and len(calls) == 10
    position = {uid: i for i, uid in enumerate(calls)}
    for unit in units:
        assert all(position[d] < position[unit.id] for d in unit.deps if d in position)


def test_failed_unit_blocks_dependents_and_exits_4(tmp_path):
    stubs = stub_executors(fail={"s1.val.train.V1"})
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=stubs) == 4
    assert unit_record(tmp_path, "s1.val.train.V1")["status"] == "failed"
    assert unit_record(tmp_path, "s1.val.train.V1")["extra"]["retryable"] is False
    for blocked in ("s1.val.ptq.V2", "s1.val.clean.V1", "s1.val.clean.V2", "s1.val.paircheck"):
        assert unit_record(tmp_path, blocked)["status"] == "blocked", blocked
    assert "train.V1" in unit_record(tmp_path, "s1.val.ptq.V2")["error"]["message"]
    assert "s1.val.ptq.V2" not in stubs["train"].calls
    assert unit_record(tmp_path, "s1.val.train.V3")["status"] == "completed"
    # a plain rerun does not retry a non-retryable failure; --retry-failed does, and then everything completes
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=stubs) == 4
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, "--retry-failed",
               executors=stub_executors()) == 0
    assert unit_record(tmp_path, "s1.val.clean.V2")["status"] == "completed"


def test_oserror_is_retryable_without_the_flag(tmp_path):
    def hook(unit, env):
        if unit.id == "s1.val.calib" and not hook.done:
            hook.done = True
            raise OSError("disk full")
    hook.done = False
    stubs = stub_executors(hook=hook)
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=stubs) == 4
    assert unit_record(tmp_path, "s1.val.calib")["extra"]["retryable"] is True
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=stubs) == 0


def test_invalid_units_are_counted_but_do_not_block(tmp_path):
    stubs = stub_executors(invalid={"s1.val.ptq.V2"})
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=stubs) == 5
    assert unit_record(tmp_path, "s1.val.ptq.V2")["status"] == "invalid"
    assert unit_record(tmp_path, "s1.val.clean.V2")["status"] == "completed"  # dependent still ran
    # invalid units are recorded evidence: a rerun does not repeat them
    again = stub_executors()
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=again) == 5
    assert again["train"].calls == []


def test_system_exit_marks_interrupted_and_restart_skips_completed(tmp_path):
    first = stub_executors(die={"s1.val.train.V3"})
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=first) == 4
    record = unit_record(tmp_path, "s1.val.train.V3")
    assert record["status"] == "interrupted" and record["extra"]["worker_died"] is True
    finished = {uid for uid in first["train"].calls if unit_record(tmp_path, uid)["status"] == "completed"}
    second = stub_executors()
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=second) == 0
    assert finished and not finished & set(second["train"].calls)          # nothing finished was redone
    assert "s1.val.train.V3" in second["train"].calls
    assert unit_record(tmp_path, "s1.val.clean.V4")["status"] == "completed"


def test_stop_flag_stops_dispatch_and_exits_130(tmp_path):
    def hook(unit, env):
        if unit.kind == "calib":
            env.stop_flag.set()
    stubs = stub_executors(hook=hook)
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=stubs) == 130
    assert unit_record(tmp_path, "s1.val.calib")["status"] == "completed"
    assert len(stubs["train"].calls) < 10
    # the next run clears the marker and finishes the rest
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=stub_executors()) == 0


def test_second_run_reuses_everything(tmp_path, capsys):
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=stub_executors()) == 0
    again = stub_executors()
    capsys.readouterr()
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY, executors=again) == 0
    assert again["train"].calls == []
    assert "reused 10 completed units (seed 1)" in capsys.readouterr().out


def test_crash_debris_is_archived_not_overwritten(tmp_path):
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "calib", executors=stub_executors()) == 0
    record_path = run_dir(tmp_path) / "state/units/s1.val.calib.json"
    record_path.unlink()                                 # crash after the files, before the completion marker
    leftover = run_dir(tmp_path) / "seeds/s1/calibration/calibration_ids.json"
    assert leftover.is_file()
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "calib", executors=stub_executors()) == 0
    archived = list((run_dir(tmp_path) / "archive").glob("incomplete-*/calibration"))
    assert len(archived) == 1 and (archived[0] / "calibration_ids.json").is_file()


def test_fingerprint_mismatch_exits_2_and_writes_nothing(tmp_path):
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "calib", executors=stub_executors()) == 0
    path = run_dir(tmp_path) / "state/fingerprints.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["components"]["source_quant"] = "0" * 64
    path.write_text(json.dumps(doc), encoding="utf-8")
    before = snapshot(run_dir(tmp_path))
    stubs = stub_executors()
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "calib", "train", executors=stubs) == 2
    assert stubs["train"].calls == []
    assert snapshot(run_dir(tmp_path)) == before


def test_archive_flag_moves_only_affected_units(tmp_path):
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "train", "calib", "ptq",
               executors=stub_executors()) == 0
    root = run_dir(tmp_path)
    # pretend calib and ptq ran under an older quantization source, train did not
    fp = json.loads((root / "state/fingerprints.json").read_text(encoding="utf-8"))
    fp["components"]["source_quant"] = "1" * 64
    (root / "state/fingerprints.json").write_text(json.dumps(fp), encoding="utf-8")
    for uid in ("s1.val.calib", "s1.val.ptq.V2", "s1.val.ptq.V4"):
        path = root / "state/units" / f"{uid}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["fingerprints_used"]["source_quant"] = "1" * 64
        path.write_text(json.dumps(record), encoding="utf-8")
    train_before = unit_record(tmp_path, "s1.val.train.V1")
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "train", "calib", "ptq",
               executors=stub_executors()) == 2
    stubs = stub_executors()
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "train", "calib", "ptq",
               "--archive-and-restart-affected", "fixq", executors=stubs) == 0
    assert sorted(stubs["train"].calls) == ["s1.val.calib", "s1.val.ptq.V2", "s1.val.ptq.V4"]
    assert unit_record(tmp_path, "s1.val.train.V1") == train_before                       # untouched
    archived = [p.name for p in (root / "archive").glob("fixq-*/*")]
    assert "s1.val.calib.json" in archived and "calibration" in archived and "s1.val.train.V1.json" not in archived
    assert json.loads((root / "state/fingerprints.json").read_text(encoding="utf-8"))["components"]["source_quant"] != "1" * 64


def test_adding_seeds_later_leaves_seed1_untouched(tmp_path, capsys):
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *BASE_SEED1_STAGES[:-1],
               executors=stub_executors()) == 0
    seed1 = {p.name: p.read_bytes() for p in (run_dir(tmp_path) / "state/units").glob("s1.*.json")}
    assert len(seed1) == 1 + 2 + 1 + 2 + 4 + 12 + 1 + 2      # calib, train, paircheck, ptq, clean, e4, decision, diag
    stubs = stub_executors()
    capsys.readouterr()
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "7", "13", "29", "--stages", *EARLY,
               executors=stubs) == 0
    out = capsys.readouterr().out
    assert f"reused {len(seed1)} completed units (seed 1)" in out
    assert not any(uid.startswith("s1.") for uid in stubs["train"].calls)
    assert {p.name: p.read_bytes() for p in (run_dir(tmp_path) / "state/units").glob("s1.*.json")} == seed1
    assert unit_record(tmp_path, "s29.val.clean.V4")["status"] == "completed"


def test_later_seeds_wait_for_the_step0_decision(tmp_path):
    stubs = stub_executors()
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "7", "--stages", "train", "calib", "step0",
               executors=stubs) == 2   # step0 needs paircheck/ptq/diag units that are not selected: refused, not added
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "7", "--stages", "calib", executors=stubs) == 2
    assert stubs["train"].calls == []


def test_step0_decision_is_made_first_then_the_remaining_units_run_with_the_decided_budget(tmp_path):
    stubs = stub_executors()
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "7", "--stages", *BASE_SEED1_STAGES,
               executors=stubs) == 0
    calls = stubs["train"].calls
    assert calls.index("s1.val.step0.decision") < min(i for i, u in enumerate(calls) if u.startswith("s7."))
    assert any(u.startswith("s7.val.attack.") and ".pgd20." in u for u in calls)


def test_qat_stage_is_opt_in_and_a_blocked_arm_is_documented_not_fatal(tmp_path, monkeypatch):
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *BASE_SEED1_STAGES,
               executors=stub_executors()) == 0
    monkeypatch.delenv("M2_ENABLE_QAT", raising=False)
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "qat", executors=stub_executors()) == 3
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "qat", "--opt-in", "qat",
               executors=stub_executors()) == 3
    monkeypatch.setenv("M2_ENABLE_QAT", "yes")
    stubs = stub_executors(blocked={"s1.val.qat.FP32-FT"})
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "qat", "--opt-in", "qat",
               executors=stubs) == 0
    blocked = unit_record(tmp_path, "s1.val.qat.FP32-FT")
    assert blocked["status"] == "blocked" and blocked["error"]["stage"] == "finetune" and blocked["extra"]["documented"]
    assert unit_record(tmp_path, "s1.val.qat.FP32-FT-PTQ")["status"] == "blocked"   # dependent, documented
    assert unit_record(tmp_path, "s1.val.qat.QAT-FT")["status"] == "completed"
    assert unit_record(tmp_path, "s1.val.clean.QAT-FT")["status"] == "completed"
    assert unit_record(tmp_path, "s1.val.clean.FP32-FT")["status"] == "blocked"
    # a blocked arm is not retried by a plain rerun
    again = stub_executors()
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", "qat", "--opt-in", "qat", executors=again) == 0
    assert again["train"].calls == []


def test_status_lists_completed_and_open_units(tmp_path, capsys):
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *EARLY,
               executors=stub_executors(fail={"s1.val.train.V1"})) == 4
    capsys.readouterr()
    assert cli(tmp_path, "status", "--json") == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["counts"]["validation"]["train"] == {"failed": 1, "completed": 1}
    assert cli(tmp_path, "status", "--verify") == 0
    victim = run_dir(tmp_path) / "seeds/s1/models/V3/checkpoint.pt"
    victim.write_bytes(b"corrupt")
    assert cli(tmp_path, "status", "--verify") == 2


def test_estimate_uses_completed_units(tmp_path, capsys):
    assert cli(tmp_path, "run", "--phase", "validation", "--seeds", "1", "--stages", *BASE_SEED1_STAGES,
               executors=stub_executors()) == 0
    capsys.readouterr()
    assert cli(tmp_path, "estimate", "--workers", "2") == 0
    report = json.loads((run_dir(tmp_path) / "estimate.json").read_text(encoding="utf-8"))
    assert report["workers"] == 2 and report["samples"] and report["remaining_units"] > 0


def test_linux_peak_memory_reader_uses_vmhwm_from_an_injected_status_file(tmp_path):
    from research import m2_efficiency as me
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmPeak:\t  999999 kB\nVmHWM:\t    2048 kB\nVmRSS:\t    1024 kB\n", encoding="utf-8")
    assert me.peak_memory_bytes(status, "linux") == (2048 * 1024, "VmHWM")
    fallback, metric = me.peak_memory_bytes(tmp_path / "missing", "linux") if hasattr(__import__("os"), "uname") else (1, "ru_maxrss")
    assert metric == "ru_maxrss" and fallback > 0
