"""Timing CLI decision logic with injected measurements, plus the probe call pattern with a stub deps object."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import CONFIG_PATH, fake_efficiency_cfg, make_measurements
from stub_support import cheap_contention, make_fake_v4

from m2_deadline import markers, timing
from m2_deadline.common import out_path, read_json, sha256_file


@pytest.fixture
def v4(tmp_path):
    root, _split = make_fake_v4(tmp_path / "v4")
    return root


def run_timing(tmp_path, v4, hours, extra=(), measurements=None, contention=cheap_contention, out=None):
    out = out or (tmp_path / "artifacts" / "m2-deadline-v1")
    argv = ["--hours-remaining", str(hours), "--v4-root", str(v4), "--out", str(out), "--config", str(CONFIG_PATH)] + list(extra)
    m = measurements if measurements is not None else make_measurements()
    return timing.main(argv, measure_fn=lambda env: m, contention_fn=contention), out


def test_feasible_writes_estimate_and_lock(tmp_path, v4, capsys):
    code, out = run_timing(tmp_path, v4, 11.0)
    assert code == 0
    est = read_json(out_path(out, "timing_estimate"))
    lock = read_json(out_path(out, "budget_lock"))
    assert est["decision"]["status"] == "feasible"
    assert (lock["epochs"], lock["per_class_target"]) == (25, 10)  # cheap synthetic timings fit the 8 h cap
    assert lock["estimate_sha256"] == sha256_file(out_path(out, "timing_estimate"))
    assert lock["config_sha256"] == sha256_file(CONFIG_PATH)
    assert lock["performance_inspected"] is False and est["no_performance_inspected"] is True
    text = capsys.readouterr().out
    assert "planned hours" in text and "launch" in text


def test_estimate_records_no_performance_numbers(tmp_path, v4):
    code, out = run_timing(tmp_path, v4, 11.0)
    assert code == 0

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    names = [str(k).lower() for k in keys(read_json(out_path(out, "timing_estimate")))]
    assert not [k for k in names if "accuracy" in k or "f1" in k or "loss" in k]


def test_lock_follows_runtime_only(tmp_path, v4):
    # slower training pushes the choice down the grid; the inputs are only timings
    code, out = run_timing(tmp_path, v4, 11.0, measurements=make_measurements(step_adv=40.0))
    assert code == 0
    lock = read_json(out_path(out, "budget_lock"))
    assert lock["epochs"] == 10 and lock["per_class_target"] == 10


def test_infeasible_exits_4_and_writes_no_lock(tmp_path, v4):
    code, out = run_timing(tmp_path, v4, 3.5, measurements=make_measurements(step_adv=30.0))
    assert code == 4
    assert out_path(out, "timing_estimate").is_file()
    assert not out_path(out, "budget_lock").exists()
    assert read_json(out_path(out, "timing_estimate"))["decision"]["status"] == "infeasible"


def test_hours_three_or_less_is_infeasible(tmp_path, v4):
    code, out = run_timing(tmp_path, v4, 3.0)
    assert code == 4 and not out_path(out, "budget_lock").exists()


@pytest.mark.parametrize("hours", [None, 0, -1, 13, float("nan")])
def test_bad_hours_exit_2_and_write_nothing(tmp_path, v4, hours):
    out = tmp_path / "artifacts" / "m2-deadline-v1"
    argv = ["--v4-root", str(v4), "--out", str(out), "--config", str(CONFIG_PATH)]
    if hours is not None:
        argv += ["--hours-remaining", str(hours)]
    called = []
    code = timing.main(argv, measure_fn=lambda env: called.append(1) or make_measurements(), contention_fn=cheap_contention)
    assert code == 2 and not called
    assert not out.exists()


def test_missing_v4_root_exits_2(tmp_path, monkeypatch):
    monkeypatch.delenv("V4_ROOT", raising=False)
    out = tmp_path / "out"
    code = timing.main(["--hours-remaining", "11", "--out", str(out)], measure_fn=lambda env: make_measurements(), contention_fn=cheap_contention)
    assert code == 2 and not out.exists()


def test_smoke_out_path_exits_3(tmp_path, v4):
    code, out = run_timing(tmp_path, v4, 11.0, out=tmp_path / "output" / "m2b-v1-smoke")
    assert code == 3 and not out.exists()


def test_contention_before_probe_exits_5_and_writes_no_estimate(tmp_path, v4):
    busy = lambda cfg: {"contended": True, "load": 90.0, "busy_processes": [{"pid": 1, "name": "research.m2b_campaign"}]}
    called = []
    out = tmp_path / "artifacts" / "m2-deadline-v1"
    argv = ["--hours-remaining", "11", "--v4-root", str(v4), "--out", str(out), "--config", str(CONFIG_PATH)]
    code = timing.main(argv, measure_fn=lambda env: called.append(1) or make_measurements(), contention_fn=busy)
    assert code == 5 and not called
    assert not out_path(out, "timing_estimate").exists() and not out_path(out, "budget_lock").exists()


def test_allow_contention_records_and_locks(tmp_path, v4):
    busy = lambda cfg: {"contended": True, "load": 90.0, "busy_processes": []}
    code, out = run_timing(tmp_path, v4, 11.0, extra=["--allow-contention"], contention=busy)
    assert code == 0
    est = read_json(out_path(out, "timing_estimate"))
    assert est["contention"]["contended"] is True and est["contention"]["contention_accepted"] is True


def test_contention_after_probe_writes_estimate_but_refuses_lock(tmp_path, v4):
    states = iter([False, True])
    flip = lambda cfg: {"contended": next(states), "load": 5.0, "busy_processes": []}
    out = tmp_path / "artifacts" / "m2-deadline-v1"
    argv = ["--hours-remaining", "11", "--v4-root", str(v4), "--out", str(out), "--config", str(CONFIG_PATH)]
    code = timing.main(argv, measure_fn=lambda env: make_measurements(), contention_fn=flip)
    assert code == 5
    assert read_json(out_path(out, "timing_estimate"))["contention"]["contended"] is True
    assert not out_path(out, "budget_lock").exists()


def noisy_measurements():
    m = make_measurements()
    m["train_step_seconds"]["adversarial"]["step_cv"] = 0.9
    return m


def test_noisy_timings_refuse_the_lock(tmp_path, v4):
    code, out = run_timing(tmp_path, v4, 11.0, measurements=noisy_measurements())
    assert code == 5
    est = read_json(out_path(out, "timing_estimate"))
    assert est["measurements"]["noisy"] is True and est["measurements"]["noisy_accepted"] is False
    assert not out_path(out, "budget_lock").exists()


def test_noisy_timings_with_allow_contention_lock_and_record(tmp_path, v4):
    code, out = run_timing(tmp_path, v4, 11.0, extra=["--allow-contention"], measurements=noisy_measurements())
    assert code == 0
    assert read_json(out_path(out, "timing_estimate"))["measurements"]["noisy_accepted"] is True
    assert out_path(out, "budget_lock").is_file()


def test_existing_lock_is_refused_without_force_relock(tmp_path, v4):
    code, out = run_timing(tmp_path, v4, 11.0)
    assert code == 0
    before = out_path(out, "budget_lock").read_bytes()
    code2, _ = run_timing(tmp_path, v4, 11.0, out=out)
    assert code2 == 3
    assert out_path(out, "budget_lock").read_bytes() == before


def test_force_relock_archives_old_files_and_writes_new(tmp_path, v4):
    code, out = run_timing(tmp_path, v4, 11.0)
    assert code == 0
    old_lock = out_path(out, "budget_lock").read_bytes()
    code2, _ = run_timing(tmp_path, v4, 11.0, extra=["--force-relock"], out=out, measurements=make_measurements(step_adv=2.0))
    assert code2 == 0
    archive = list(out_path(out, "locks_archive_dir").iterdir())
    assert len(archive) == 1
    assert (archive[0] / "budget_lock.json").read_bytes() == old_lock
    assert (archive[0] / "timing_estimate.json").is_file()
    assert out_path(out, "budget_lock").read_bytes() != old_lock


def test_force_relock_refused_when_a_stage_marker_exists(tmp_path, v4):
    code, out = run_timing(tmp_path, v4, 11.0)
    assert code == 0
    ctx = markers.context(config_sha256="c" * 64, lock_sha256="l" * 64, protocol_sha256=None, epochs=25)
    markers.write_after_validate(out, "preflight", lambda: True, [], ctx)
    before = out_path(out, "budget_lock").read_bytes()
    code2, _ = run_timing(tmp_path, v4, 11.0, extra=["--force-relock"], out=out)
    assert code2 == 3
    assert out_path(out, "budget_lock").read_bytes() == before
    assert not out_path(out, "locks_archive_dir").exists()


def test_probe_time_limit_exits_2_and_locks_nothing(tmp_path, v4):
    def slow(env):
        env.limit_s = -1.0
        env.check_deadline("test")
        return make_measurements()

    out = tmp_path / "artifacts" / "m2-deadline-v1"
    argv = ["--hours-remaining", "11", "--v4-root", str(v4), "--out", str(out), "--config", str(CONFIG_PATH), "--max-minutes", "1"]
    assert timing.main(argv, measure_fn=slow, contention_fn=cheap_contention) == 2
    assert not out_path(out, "budget_lock").exists() and not out_path(out, "timing_estimate").exists()


def test_prior_evidence_windows_pilot_is_recorded_not_used(tmp_path, v4):
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"epochs": 2, "threads": {"training": 2}, "environment": {"platform": "Windows-10"},
                                 "at_cost": {"measured_wall_clock_ratio": 6.77}}), encoding="utf-8")
    code, out = run_timing(tmp_path, v4, 11.0, extra=["--prior-evidence", str(pilot)])
    assert code == 0
    rec = read_json(out_path(out, "timing_estimate"))["prior_evidence"][0]
    assert rec["used_in_estimate"] is False and rec["sha256"] == sha256_file(pilot)
    assert rec["compatibility"]["compatible"] is False
    assert rec["compatibility"]["epochs_match"] is False and rec["compatibility"]["split_match"] is False
    assert "within_factor_2" in rec["adv_over_ord_ratio"]


# ---------------------------------------------------------------- the probe itself, with stub deps
class ProbeStub:
    def __init__(self, split, n_train=7000, n_val=270):
        self.calls = []
        rng = np.random.default_rng(1)
        self.train = {"ids": np.arange(n_train) + 10000, "x": rng.random((n_train, 39)).astype(np.float32),
                      "y": (np.arange(n_train) % 27).astype(np.int64)}
        self.val = {"ids": np.arange(n_val) + 50000, "x": rng.random((n_val, 39)).astype(np.float32),
                    "y": (np.arange(n_val) % 27).astype(np.int64)}
        outer = self

        def load_arrays(split_path, scaler_path, partitions=("train", "validation"), **kw):
            outer.calls.append(("load_arrays", tuple(partitions), kw))
            assert "test" not in partitions, "the timing probe asked for test arrays"
            return {p: (outer.train if p == "train" else outer.val) for p in partitions}

        def train_unit(cfg, X, y, seed, out_dir=None, **kw):
            outer.calls.append(("train_unit", len(X), out_dir, kw))
            return {"model": SimpleNamespace(tag=kw["adversarial"]), "log": {"step_seconds": [0.1 + 0.001 * i for i in range(len(X) // 512)]}}

        def evaluate_unit(**kw):
            outer.calls.append(("evaluate_unit", kw["source"], kw["attack_name"], kw["epsilon"], kw["evaluation_partition"]))
            return {}

        def run_efficiency(models, rows, cfg, out_dir, **kw):
            outer.calls.append(("run_efficiency", sorted(models), cfg["repetitions"], kw))
            return {"any_failed": False}

        self.m2_split = SimpleNamespace(load_arrays=load_arrays)
        self.mt = SimpleNamespace(
            load_config=lambda p: {"architecture": {}, "training": {"batch_size": 512}},
            train_unit=train_unit,
            save_epoch_checkpoint=lambda *a, **k: outer.calls.append(("save_epoch_checkpoint", k["epoch"])),
        )
        self.mq = SimpleNamespace(
            load_config=lambda p: {"ptq": {"calibration_per_class": 20, "scope": "all_linear", "backend": "fbgemm"}},
            calibration_ids=lambda arrays, per_class, seed: {"x": np.zeros((540, 50, 39), dtype=np.float32)},
            ptq_unit=lambda model, cal, cfg, out_dir, **kw: {"model": SimpleNamespace(tag="int8"), "conversion": {"requested": ["a"]}},
        )
        self.ma = SimpleNamespace(
            load_config=lambda p: {"attacks": {"fgsm": {}}, "expected_selector_calls_per_batch": {}},
            validate_config=lambda cfg: cfg,
            evaluate_unit=evaluate_unit,
        )
        self.me = SimpleNamespace(
            load_config=lambda p: fake_efficiency_cfg(),
            select_rows=lambda ids, x, count, seed: {"ids": [], "x": x[:count]},
            run_efficiency=run_efficiency,
        )
        self.quantization = SimpleNamespace(bpda_model=lambda q, requested, example: SimpleNamespace(tag="bpda"))

    def configure_inference(self, threads=1, engine="fbgemm"):
        self.calls.append(("configure_inference", threads))

    def make_probe_optimizer(self, cfg, model):
        return object()

    def sequences(self, x):
        return np.asarray(x)

    def predict(self, model, x, batch_size=64):
        self.calls.append(("predict", getattr(model, "tag", None), len(x), batch_size))
        return np.zeros(len(x), dtype=np.int64)

    def save_fp32_checkpoint(self, model, path, arch):
        pass

    def save_tensor(self, value, path):
        pass

    def parameter_count(self, model):
        return 1


def test_probe_requests_only_train_and_validation_and_uses_the_real_code_paths(tmp_path, v4):
    split = json.loads((v4 / "reports/milestone2b/split_v2/split_m2.json").read_text(encoding="utf-8"))
    deps = ProbeStub(split)
    out = tmp_path / "artifacts" / "m2-deadline-v1"
    code = timing.main(
        ["--hours-remaining", "11", "--v4-root", str(v4), "--out", str(out), "--config", str(CONFIG_PATH)],
        contention_fn=cheap_contention, deps=deps,
    )
    assert code == 0, code
    loads = [c for c in deps.calls if c[0] == "load_arrays"]
    assert [c[1] for c in loads] == [("train", "validation")]
    assert all("purpose" not in c[2] and "freeze" not in c[2] for c in loads)
    trains = [c for c in deps.calls if c[0] == "train_unit"]
    assert [(c[1], c[3]["adversarial"], c[3]["epochs"], c[2]) for c in trains] == [(8 * 512, False, 1, None), (12 * 512, True, 1, None)]
    attacks = [c for c in deps.calls if c[0] == "evaluate_unit"]
    assert len(attacks) == 6 and all(c[4] == "validation" for c in attacks)
    assert {(c[1], c[2], c[3]) for c in attacks} == {
        (s, a, e) for s in ("fp32_direct", "bpda_int8") for a, e in (("fgsm", 0.0), ("fgsm", 0.05), ("pgd5", 0.05))
    }
    preds = [c for c in deps.calls if c[0] == "predict"]
    assert [(c[1], c[2], c[3]) for c in preds] == [(False, 270, 64), ("int8", 270, 64)]
    eff = [c for c in deps.calls if c[0] == "run_efficiency"][0]
    assert eff[1] == ["V1", "V2", "V3", "V4"] and eff[2] == {"full": 1, "pilot": 1}
    assert eff[3]["pilot"] is False and eff[3]["threads"] == 1
    est = read_json(out_path(out, "timing_estimate"))
    assert est["data"]["n_test"] == len(split["row_ids"]["test"])
    assert est["measurements"]["attack_seconds_per_example"]["probe_n"] == 81
    assert est["measurements"]["val_eval_s"]["n_rows"] == 270
    assert est["measurements"]["efficiency_s"] > 0


def test_modules_import_without_torch():
    import subprocess

    code = (
        "import sys, m2_deadline.timing, m2_deadline.stages, m2_deadline.campaign;"
        "assert 'torch' not in sys.modules and not [m for m in sys.modules if m.startswith('research')]"
    )
    root = Path(CONFIG_PATH).parents[1]
    done = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(root), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
