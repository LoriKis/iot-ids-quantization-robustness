"""Efficiency harness tests on tiny models: structure, warm-up exclusion, memory fields, interleaved schedule."""
import copy
import json
import sys

import numpy as np
import pytest
import torch

from m2_helpers import TINY_ARCH, TINY_ISLANDS, split_arrays, tiny_model
from research import m2_efficiency as me
from research import m2_quant as mq
from research.m2_split import ConfigError

ROWS = 16


@pytest.fixture(autouse=True)
def restore_threads():
    threads, engine = torch.get_num_threads(), torch.backends.quantized.engine
    yield
    torch.set_num_threads(threads)
    torch.backends.quantized.engine = engine


@pytest.fixture(scope="module")
def cfg():
    cfg = me.load_config()
    cfg["fixed_inputs"]["count"] = ROWS
    cfg["batch256"]["batch_size"] = ROWS
    for block in (cfg["batch1"], cfg["batch256"]):
        block.update(warmup=2, timed=4, pilot_warmup=1, pilot_timed=3)
    cfg["repetitions"].update(full=3, pilot=2)
    return me.validate_config(cfg)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """FP32 checkpoint, packed INT8 state, calibration tensor and V1..V4 specs of a tiny model."""
    d = tmp_path_factory.mktemp("eff_models")
    model = tiny_model()
    ckpt = d / "ckpt.pt"
    torch.save({"state_dict": model.state_dict(), "architecture": TINY_ARCH}, ckpt)
    qcfg = mq.load_config()
    cal = mq.calibration_ids(split_arrays(), 20, 20260)
    out = mq.ptq_unit(model, cal, qcfg, d / "q", variant="V2", expected_islands=TINY_ISLANDS)
    torch.save(cal["x"], d / "cal.pt")
    fp32 = {"kind": "fp32", "checkpoint": ckpt, "architecture": TINY_ARCH}
    int8 = {"kind": "int8", "state": d / "q/state.pt", "float_checkpoint": ckpt, "calibration_path": d / "cal.pt",
            "scope": "all_linear", "backend": "fbgemm", "architecture": TINY_ARCH}
    arrays = split_arrays()
    rows = me.select_rows(arrays["validation"]["ids"], arrays["validation"]["x"], ROWS, 20260)
    return {"models": {"V1": fp32, "V2": int8, "V3": fp32, "V4": int8}, "conversion": out["conversion"], "rows": rows}


def in_process(spec, index):
    return me.child_measure(spec)


def test_config_is_valid_and_frozen_values():
    cfg = me.load_config()
    assert cfg["variants"] == ["V1", "V2", "V3", "V4"] and cfg["threads"]["primary"] == 1
    assert (cfg["batch1"]["warmup"], cfg["batch1"]["timed"], cfg["batch1"]["pilot_warmup"], cfg["batch1"]["pilot_timed"]) == (20, 200, 10, 50)
    b = cfg["batch256"]
    assert (b["batch_size"], b["warmup"], b["timed"], b["pilot_warmup"], b["pilot_timed"]) == (256, 5, 30, 2, 10)
    assert cfg["repetitions"]["full"] == 3 and cfg["child"] == {"command": cfg["child"]["command"], "timeout_seconds": 300, "max_retries": 1}


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(extra=1),
    lambda c: c.pop("memory"),
    lambda c: c.update(variants=["V2", "V1", "V3", "V4"]),
    lambda c: c["threads"].update(primary=2),
    lambda c: c["batch256"].update(batch_size=5000),
    lambda c: c["batch256"].update(batch_size=128),
    lambda c: c["batch1"].update(pilot_timed=500),
    lambda c: c["batch1"].update(warmup=0),
    lambda c: c["repetitions"].update(pilot=4),
    lambda c: c["child"].update(timeout_seconds=0),
    lambda c: c["child"].update(max_retries=2),
    lambda c: c["inference"].update(quantized_engine="qnnpack"),
])
def test_invalid_config_raises(mutate):
    cfg = copy.deepcopy(me.load_config())
    mutate(cfg)
    with pytest.raises(ConfigError):
        me.validate_config(cfg)


def test_missing_or_broken_config_file(tmp_path):
    with pytest.raises(ConfigError):
        me.load_config(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(ConfigError):
        me.load_config(bad)


def test_measure_excludes_warmup_calls_from_samples():
    calls = []
    samples = me.measure(lambda: calls.append(1), warmup=7, timed=5)
    assert len(calls) == 12 and len(samples) == 5 and all(s >= 0 for s in samples)


def test_summarize_statistics_and_raw_samples():
    s = me.summarize([1e6, 2e6, 3e6, 4e6, 100e6], batch_size=4)
    assert s["median_ms"] == 3.0 and s["min_ms"] == 1.0 and s["samples_ms"] == [1.0, 2.0, 3.0, 4.0, 100.0]
    assert s["p95_ms"] == pytest.approx(np.percentile([1, 2, 3, 4, 100], 95))
    assert s["throughput_samples_per_s"] == pytest.approx(4 / 0.003)
    with pytest.raises(ValueError):
        me.summarize([], 1)


def test_scheduler_yields_interleaved_order_three_repetitions():
    order = me.interleaved_order(["V1", "V2", "V3", "V4"], 3)
    assert [v for _, v in order] == ["V1", "V2", "V3", "V4"] * 3
    assert [r for r, _ in order] == [1] * 4 + [2] * 4 + [3] * 4


def test_select_rows_is_fixed_saved_and_hashed():
    a = split_arrays()["validation"]
    first = me.select_rows(a["ids"], a["x"], 32, 20260)
    again = me.select_rows(a["ids"], a["x"], 32, 20260)
    other = me.select_rows(a["ids"], a["x"], 32, 20261)
    assert first["ids"] == again["ids"] and first["x_sha256"] == again["x_sha256"] and first["id_sha256"] == again["id_sha256"]
    assert first["ids"] != other["ids"] and len(set(first["ids"])) == 32 and set(first["ids"]) <= set(a["ids"].tolist())
    with pytest.raises(ValueError):
        me.select_rows(a["ids"], a["x"], len(a["ids"]) + 1, 1)


def test_child_measure_fp32_and_int8_fields(tiny, cfg, tmp_path):
    rows = tiny["rows"]
    np.save(tmp_path / "rows.npy", rows["x"])
    for variant in ("V1", "V2"):
        spec = {**{k: (str(v) if not isinstance(v, (dict, str)) else v) for k, v in tiny["models"][variant].items()},
                "variant": variant, "threads": 1, "rows_path": str(tmp_path / "rows.npy"), "rows_sha256": rows["x_sha256"],
                "batch_size": ROWS, "batch1": {"warmup": 1, "timed": 3}, "batch256": {"warmup": 1, "timed": 2}}
        result = me.child_measure(spec)
        assert result["torch_threads"] == 1 and result["inference_mode"] is True and result["model_training_flag"] is False
        assert result["batch1"]["n"] == 3 and result["batch256"]["n"] == 2 and len(result["batch1"]["samples_ms"]) == 3
        assert result["batch256"]["batch_size"] == ROWS and result["state_dict_bytes"] > 0
        assert result["baseline_rss_bytes"] > 0 and result["sampled_peak_rss_bytes"] > 0
        assert result["sampler_interval_seconds"] == 0.01
        if sys.platform == "win32":
            assert result["peak_wset_bytes"] >= result["baseline_rss_bytes"] * 0.5
        assert result["quantized_engine"] == "fbgemm" or variant == "V1"
    with pytest.raises(RuntimeError, match="hash"):
        me.child_measure({**spec, "rows_sha256": "0" * 64})


def test_run_efficiency_records_interleaved_run_order(tiny, cfg, tmp_path):
    record = me.run_efficiency(tiny["models"], tiny["rows"], cfg, tmp_path / "eff", pilot=True, run_one=in_process,
                               conversions={"V2": tiny["conversion"], "V4": tiny["conversion"]}, total_parameters=1234,
                               banner="banner")
    order = [e["variant"] for e in record["run_order"]]
    assert order == ["V1", "V2", "V3", "V4"] * cfg["repetitions"]["pilot"] and record["interleaved"] is True
    assert [e["repetition"] for e in record["run_order"]] == [1] * 4 + [2] * 4 and not record["any_retry"] and not record["any_failed"]
    assert all(e["started_unix"] > 0 and e["started_utc"] and e["cpu_load_percent_before"] is not None
               and e["ram_available_bytes_before"] > 0 for e in record["run_order"])
    assert record["planned_order"] == order and record["mode"] == "pilot" and record["banner"] == "banner"
    assert record["warmup_and_timed"] == {"batch1": {"warmup": 1, "timed": 3}, "batch256": {"warmup": 1, "timed": 3}}
    saved = json.loads((tmp_path / "eff/efficiency.json").read_text())
    assert saved["run_order"] == record["run_order"] and set(saved["summary"]) == {"V1", "V2", "V3", "V4"}
    assert saved["summary"]["V2"]["n_processes"] == 2 and saved["summary"]["V2"]["state_dict_bytes"]["n"] == 2
    assert saved["summary"]["V1"]["ratio_to_V1"]["batch1_median"] == pytest.approx(1.0)
    # identical weights and rows give identical predictions for V1 and V3
    assert saved["summary"]["V1"]["predictions_sha256"] == saved["summary"]["V3"]["predictions_sha256"]
    cov = saved["mixed_precision_coverage"]
    assert cov["V1"]["int8_weight_elements"] == 0 and cov["V2"]["n_int8_linear_islands"] == TINY_ISLANDS
    assert cov["V2"]["quant_dequant_boundaries"] == 2 * TINY_ISLANDS and cov["V2"]["int8_weight_share"] > 0
    ids = json.loads((tmp_path / "eff/fixed_rows.json").read_text())
    assert ids["id_sha256"] == tiny["rows"]["id_sha256"] and len(ids["ids"]) == ROWS
    assert "no speed or memory advantage" in saved["no_claim"]
    with pytest.raises(FileExistsError):
        me.run_efficiency(tiny["models"], tiny["rows"], cfg, tmp_path / "eff", pilot=True, run_one=in_process)


def test_failed_measurement_is_retried_once_appended_and_flagged():
    calls = []

    def flaky(spec, index):
        calls.append((spec["variant"], index))
        if spec["variant"] == "V2" and len([c for c in calls if c[0] == "V2"]) == 1:
            raise RuntimeError("boom")
        return {"ok": index}

    specs = {v: {"variant": v} for v in ("V1", "V2", "V3", "V4")}
    result = me.run_schedule(specs, me.interleaved_order(list(specs), 1), flaky, max_retries=1)
    order = result["run_order"]
    assert [e["variant"] for e in order] == ["V1", "V2", "V3", "V4", "V2"]
    assert [e["status"] for e in order] == ["ok", "failed", "ok", "ok", "ok"]
    assert [e["retry"] for e in order] == [False] * 4 + [True] and "boom" in order[1]["error"]
    assert len(result["measurements"]) == 4 and result["measurements"][-1]["retry"] is True
    never = me.run_schedule(specs, [(1, "V1")], lambda s, i: (_ for _ in ()).throw(RuntimeError("x")), max_retries=1)
    assert [e["status"] for e in never["run_order"]] == ["failed", "failed"] and not never["measurements"]
    no_retry = me.run_schedule(specs, [(1, "V1")], lambda s, i: (_ for _ in ()).throw(RuntimeError("x")), max_retries=0)
    assert len(no_retry["run_order"]) == 1


def test_models_must_be_v1_to_v4_in_order(tiny, cfg, tmp_path):
    with pytest.raises(ValueError):
        me.run_efficiency({"V2": tiny["models"]["V2"], "V1": tiny["models"]["V1"]}, tiny["rows"], cfg, tmp_path / "x",
                          pilot=True, run_one=in_process)


def test_fresh_subprocess_child_reports_memory_and_latency(tiny, cfg, tmp_path):
    rows = tiny["rows"]
    np.save(tmp_path / "rows.npy", rows["x"])
    spec = {**{k: (str(v) if not isinstance(v, (dict, str)) else v) for k, v in tiny["models"]["V2"].items()},
            "variant": "V2", "threads": 1, "rows_path": str(tmp_path / "rows.npy"), "rows_sha256": rows["x_sha256"],
            "batch_size": ROWS, "batch1": {"warmup": 1, "timed": 3}, "batch256": {"warmup": 1, "timed": 2}}
    result = me.subprocess_runner(tmp_path, timeout=120, threads=1)(spec, 1)
    assert result["variant"] == "V2" and result["torch_threads"] == 1
    assert result["omp_num_threads_env"] == "1" and result["mkl_num_threads_env"] == "1"
    assert result["baseline_rss_bytes"] > 0 and result["sampled_peak_rss_bytes"] > 0 and result["batch1"]["median_ms"] > 0
    assert (tmp_path / "specs/01.json").is_file() and (tmp_path / "results/01.json").is_file()
    with pytest.raises(RuntimeError, match="exited"):
        me.subprocess_runner(tmp_path, timeout=120, threads=1)({**spec, "rows_sha256": "0" * 64}, 2)
