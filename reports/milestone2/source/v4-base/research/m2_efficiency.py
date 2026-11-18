"""Milestone 2A efficiency harness: artifact size, single-thread latency/throughput and process memory for V1-V4.

Every measurement runs in a fresh child process (`py -B -m research.m2_efficiency --child --spec S --out R`)
so that memory numbers are not shared between models. The parent schedules the children interleaved
(V1, V2, V3, V4, V1, V2, ...) so that laptop thermal drift is spread over all models, records the executed
order with timestamps and the CPU load before each start, and appends any retried measurement at the end.
Design: docs/MILESTONE2_PROTOCOL.md section F; configuration: configs/m2/efficiency.json.
No speed or memory claim is made; results are reported as measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

from .common import ROOT, atomic_json, make_model, state_bytes
from .m2_split import ConfigError, id_hash
from .paper_described_adam import derived_seed
from .provenance import MemoryMonitor
from .quantization import load_int8

log = logging.getLogger("m2_efficiency")

DEFAULT_CONFIG = ROOT / "configs/m2/efficiency.json"
TOP_KEYS = {"schema_version", "frozen_date_local", "claim", "threads", "inference", "fixed_inputs", "batch1",
            "batch256", "repetitions", "variants", "schedule", "child", "timed_region", "excluded_from_timing",
            "memory", "statistics", "artifact_size"}
SEQUENCE_LENGTH = 50
VARIANTS = ("V1", "V2", "V3", "V4")
QAT_VARIANTS = ("FP32-FT", "FP32-FT-PTQ", "QAT-FT")  # appended by --include-qat; same rows, threads, batch sizes
PROC_STATUS = "/proc/self/status"
MEMORY_COMPARISON_NOTE = "Windows pilot (peak working set) and Linux (VmHWM) memory numbers are not comparable"


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict) or set(cfg) != TOP_KEYS:
        raise ConfigError("unknown or missing efficiency configuration keys: "
                          f"{sorted(set(cfg) ^ TOP_KEYS) if isinstance(cfg, dict) else 'not an object'}")
    if cfg["schema_version"] != 1:
        raise ConfigError("unsupported efficiency schema_version")
    if list(cfg["variants"]) != list(VARIANTS):
        raise ConfigError(f"variants must be exactly {list(VARIANTS)} in that order")
    if cfg["threads"].get("primary") != 1 or not _is_int(cfg["threads"].get("secondary")) or cfg["threads"]["secondary"] < 1:
        raise ConfigError("threads.primary must be 1 and threads.secondary a positive integer")
    fixed = cfg["fixed_inputs"]
    if not _is_int(fixed["count"]) or not 1 <= fixed["count"] <= 4096 or fixed["partition"] != "validation":
        raise ConfigError("fixed_inputs must be 1..4096 validation rows")
    b256 = cfg["batch256"]
    if not _is_int(b256["batch_size"]) or not 1 <= b256["batch_size"] <= 4096:
        raise ConfigError("batch256.batch_size must be an integer in 1..4096")
    if b256["batch_size"] != fixed["count"]:
        raise ConfigError("the throughput batch must equal the fixed input count")
    for name, block in (("batch1", cfg["batch1"]), ("batch256", b256)):
        keys = ("warmup", "timed", "pilot_warmup", "pilot_timed")
        if any(not _is_int(block[k]) or block[k] < 1 for k in keys):
            raise ConfigError(f"{name} warm-up and timed counts must be positive integers")
        if block["pilot_warmup"] > block["warmup"] or block["pilot_timed"] > block["timed"]:
            raise ConfigError(f"{name} pilot counts must not exceed the full counts")
    rep = cfg["repetitions"]
    if not (_is_int(rep["full"]) and _is_int(rep["pilot"]) and 1 <= rep["pilot"] <= rep["full"]):
        raise ConfigError("repetitions need integers with 1 <= pilot <= full")
    child = cfg["child"]
    if not _is_int(child["timeout_seconds"]) or child["timeout_seconds"] < 1 or not _is_int(child["max_retries"]) \
            or not 0 <= child["max_retries"] <= 1:
        raise ConfigError("child.timeout_seconds must be positive and child.max_retries 0 or 1")
    if cfg["inference"]["quantized_engine"] != "fbgemm":
        raise ConfigError("quantized engine must be fbgemm")
    return cfg


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"efficiency configuration not found: {path}")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    return validate_config(cfg)


# ------------------------------------------------------------------ pure helpers
def summarize(samples_ns, batch_size: int) -> dict:
    """Median, p95 and min in milliseconds plus the raw samples; throughput from the median batch time."""
    ms = np.asarray(samples_ns, dtype=np.float64) / 1e6
    if ms.size == 0:
        raise ValueError("no timing samples")
    median = float(np.median(ms))
    return {"unit": "ms", "n": int(ms.size), "batch_size": int(batch_size), "median_ms": median,
            "p95_ms": float(np.percentile(ms, 95)), "min_ms": float(ms.min()),
            "throughput_samples_per_s": batch_size / (median / 1e3), "samples_ms": ms.tolist()}


def measure(fn, warmup: int, timed: int) -> list:
    """Call `fn()` `warmup` times without recording, then `timed` times; return per-call nanoseconds."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(timed):
        start = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - start)
    return samples


def interleaved_order(variants, repetitions: int) -> list:
    """[(repetition, variant)] in the order V1..V4 repeated once per repetition (never model-by-model)."""
    return [(rep, variant) for rep in range(1, repetitions + 1) for variant in variants]


def select_rows(ids, x, count: int, split_seed: int, seed_name: str = "efficiency-rows") -> dict:
    """Fixed, deterministic row selection with saved IDs and a hash of the scaled rows."""
    ids, x = np.asarray(ids), np.asarray(x, dtype=np.float32)
    if len(ids) != len(x):
        raise ValueError("ids and x must describe the same rows")
    if len(ids) < count:
        raise ValueError(f"only {len(ids)} rows available, {count} requested")
    pick = np.sort(np.random.default_rng(derived_seed(seed_name, split_seed)).choice(len(ids), count, replace=False))
    rows = np.ascontiguousarray(x[pick])
    return {"ids": [int(i) for i in ids[pick]], "id_sha256": id_hash(ids[pick]), "x": rows,
            "x_sha256": hashlib.sha256(rows.tobytes()).hexdigest(), "seed": derived_seed(seed_name, split_seed),
            "seed_rule": f"derived_seed('{seed_name}', split_seed)", "split_seed": split_seed}


def coverage_from_conversion(conversion: dict | None, total_parameters: int) -> dict:
    """Mixed-precision coverage (protocol F) from conversion.json; None means an all-FP32 model."""
    if conversion is None:
        return {"precision": "FP32", "int8_weight_elements": 0, "int8_weight_share": 0.0,
                "total_parameters": total_parameters, "n_int8_linear_islands": 0, "n_fp32_operation_kinds": None,
                "quant_dequant_boundaries": 0}
    counts = conversion["parameter_counts"]
    return {"precision": "hybrid INT8 Linear + FP32 everything else", "total_parameters": counts["total_parameters"],
            "int8_weight_elements": counts["int8_weight_elements"], "int8_weight_share": counts["int8_weight_share"],
            "fp32_parameters_not_quantized": counts["fp32_parameters_not_quantized"],
            "n_int8_linear_islands": len(conversion["requested"]),
            "classification_path_islands": conversion["classification_path_islands"],
            "anomaly_scorer_islands": conversion["anomaly_scorer_islands"],
            "n_fp32_operation_kinds": len(conversion["fp32_remaining_operations"]),
            "fp32_remaining_operations": conversion["fp32_remaining_operations"],
            "quant_dequant_boundaries": conversion["quant_dequant_boundaries"]["total"]}


# ------------------------------------------------------------------ child process
def peak_memory_bytes(status_path=PROC_STATUS, system=None):
    """(peak bytes, metric name) of this process: peak_wset on Windows, VmHWM on Linux, getrusage as the fallback.

    `status_path` and `system` are injectable so the Linux parser is testable on any platform.
    """
    system = sys.platform if system is None else system
    if system.startswith("win"):
        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
        return (None, "peak_wset") if peak is None else (int(peak), "peak_wset")
    try:
        for line in Path(status_path).read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024, "VmHWM"
    except (OSError, ValueError, IndexError):
        pass
    import resource  # POSIX only; reached when /proc is unavailable
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "ru_maxrss"


class FastMemoryMonitor(MemoryMonitor):
    """provenance.MemoryMonitor with a 10 ms sampling interval (the original samples once per second)."""

    def sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            self.peak = max(self.peak, process.memory_info().rss)
            self.min_available = min(self.min_available, psutil.virtual_memory().available)
            self.stop.wait(0.01)


def _load_checkpoint_model(path, architecture):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(architecture or state["architecture"])
    model.load_state_dict(state["state_dict"], strict=True)
    return model.cpu().eval()


def load_variant(spec: dict):
    """Rebuild the model a spec describes: an FP32 checkpoint or a packed INT8 state."""
    architecture = spec["architecture"]
    if spec["kind"] == "fp32":
        return _load_checkpoint_model(spec["checkpoint"], architecture)
    if spec["kind"] != "int8":
        raise ValueError(f"unknown model kind {spec['kind']!r}")
    parent = _load_checkpoint_model(spec["float_checkpoint"], architecture)
    calibration = torch.load(spec["calibration_path"], map_location="cpu", weights_only=True)
    state = torch.load(spec["state"], map_location="cpu", weights_only=True)
    q, _ = load_int8(parent, spec["scope"], calibration, spec["state"], state["compatibility"], "static", spec["backend"])
    return q.eval()


def child_measure(spec: dict) -> dict:
    """Run one measurement in this (fresh) process and return the result record."""
    started = time.perf_counter()
    threads = int(spec["threads"])
    torch.set_num_threads(threads)
    rows_np = np.load(spec["rows_path"])
    if hashlib.sha256(np.ascontiguousarray(rows_np, dtype=np.float32).tobytes()).hexdigest() != spec["rows_sha256"]:
        raise RuntimeError("fixed input rows differ from the saved hash")
    rows = torch.from_numpy(np.ascontiguousarray(rows_np, dtype=np.float32))
    model = load_variant(spec)
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    n_rows, batch = len(rows), int(spec["batch_size"])

    def forward(batch_rows):  # timed region: rows -> repeat to 50 positions -> forward -> argmax
        return model(batch_rows.unsqueeze(1).repeat(1, SEQUENCE_LENGTH, 1))["logits"].argmax(1)

    counter = {"i": 0}

    def single():
        i = counter["i"] % n_rows
        counter["i"] += 1
        return forward(rows[i:i + 1])

    with torch.inference_mode():
        b1 = summarize(measure(single, spec["batch1"]["warmup"], spec["batch1"]["timed"]), 1)
        b256 = summarize(measure(lambda: forward(rows[:batch]), spec["batch256"]["warmup"], spec["batch256"]["timed"]), batch)
        predictions = forward(rows[:batch])
        with FastMemoryMonitor() as monitor:  # separate untimed pass so the sampler cannot disturb latency
            for _ in range(5):
                forward(rows[:batch])
            for _ in range(20):
                single()
    peak_wset, memory_metric = peak_memory_bytes()
    return {
        "variant": spec["variant"], "kind": spec["kind"], "torch_threads": torch.get_num_threads(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"), "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
        "quantized_engine": torch.backends.quantized.engine, "inference_mode": True, "model_training_flag": bool(model.training),
        "state_dict_bytes": state_bytes(model), "baseline_rss_bytes": int(baseline_rss),
        "peak_wset_bytes": peak_wset, "peak_memory_metric": memory_metric,
        "peak_wset_minus_baseline_bytes": None if peak_wset is None else int(peak_wset - baseline_rss),
        "sampled_peak_rss_bytes": int(monitor.peak), "sampled_peak_minus_baseline_bytes": int(monitor.peak - baseline_rss),
        "sampler_interval_seconds": 0.01, "min_available_ram_bytes": int(monitor.min_available),
        "batch1": b1, "batch256": b256, "batch1_timed_region": "row -> repeat 50 -> forward -> argmax",
        "predictions_sha256": hashlib.sha256(predictions.numpy().astype("<i8").tobytes()).hexdigest(),
        "n_rows": n_rows, "child_seconds": time.perf_counter() - started}


# ------------------------------------------------------------------ parent: scheduling
def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds")


def run_schedule(specs: dict, order: list, run_one, *, max_retries: int = 1, cpu_load=None) -> dict:
    """Execute `order` with `run_one(spec, index)`; failed measurements are retried once, appended and flagged.

    Returns {'run_order': [...], 'measurements': [...]} in executed order; the order is never re-sorted.
    """
    run_order, measurements = [], []

    def execute(rep, variant, retry):
        index = len(run_order) + 1
        started = time.time()
        entry = {"index": index, "repetition": rep, "variant": variant, "retry": retry, "started_unix": started,
                 "started_utc": _utc(started), "cpu_load_percent_before": None if cpu_load is None else cpu_load(),
                 "ram_available_bytes_before": int(psutil.virtual_memory().available)}
        try:
            result = run_one(specs[variant], index)
            entry["status"] = "ok"
            measurements.append({**{k: entry[k] for k in ("index", "repetition", "variant", "retry")}, "result": result})
        except Exception as error:  # noqa: BLE001 - a failed child is recorded, then retried once
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            log.warning("efficiency measurement %s repetition %s failed: %s", variant, rep, entry["error"])
        entry["finished_unix"] = time.time()
        run_order.append(entry)

    for rep, variant in order:
        execute(rep, variant, False)
    if max_retries:
        for entry in [e for e in list(run_order) if e["status"] == "failed" and not e["retry"]]:
            execute(entry["repetition"], entry["variant"], True)
    return {"run_order": run_order, "measurements": measurements}


def subprocess_runner(work_dir: Path, timeout: int, threads: int):
    """run_one(spec, index) that starts a fresh `python -B -m research.m2_efficiency --child` process."""
    work_dir = Path(work_dir)

    def run_one(spec, index):
        spec_path, result_path = work_dir / "specs" / f"{index:02d}.json", work_dir / "results" / f"{index:02d}.json"
        atomic_json(spec_path, spec)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "PYTHONDONTWRITEBYTECODE": "1"}
        command = [sys.executable, "-B", "-m", "research.m2_efficiency", "--child", "--spec", str(spec_path),
                   "--out", str(result_path)]
        try:
            done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"child timed out after {timeout} s") from error
        if done.returncode != 0:
            raise RuntimeError(f"child exited with {done.returncode}: {done.stderr.strip()[-400:]}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    return run_one


def _spread(values):
    values = [v for v in values if v is not None]
    return None if not values else {"n": len(values), "min": min(values), "median": float(np.median(values)), "max": max(values)}


def summarize_runs(measurements: list, variants) -> dict:
    """Per-variant spread over independent processes; ratios to V1 are descriptive, not a claim."""
    out = {}
    for variant in variants:
        runs = [m["result"] for m in measurements if m["variant"] == variant]
        out[variant] = {
            "n_processes": len(runs), "state_dict_bytes": _spread([r["state_dict_bytes"] for r in runs]),
            "batch1_median_ms": _spread([r["batch1"]["median_ms"] for r in runs]),
            "batch1_p95_ms": _spread([r["batch1"]["p95_ms"] for r in runs]),
            "batch256_median_ms": _spread([r["batch256"]["median_ms"] for r in runs]),
            "batch256_throughput_samples_per_s": _spread([r["batch256"]["throughput_samples_per_s"] for r in runs]),
            "baseline_rss_bytes": _spread([r["baseline_rss_bytes"] for r in runs]),
            "peak_wset_minus_baseline_bytes": _spread([r["peak_wset_minus_baseline_bytes"] for r in runs]),
            "sampled_peak_minus_baseline_bytes": _spread([r["sampled_peak_minus_baseline_bytes"] for r in runs]),
            "predictions_sha256": sorted({r["predictions_sha256"] for r in runs})}
    base = out.get("V1", {})
    for variant, entry in out.items():
        b1, b256 = entry["batch1_median_ms"], entry["batch256_median_ms"]
        entry["ratio_to_V1"] = None if not (b1 and base.get("batch1_median_ms")) else {
            "batch1_median": b1["median"] / base["batch1_median_ms"]["median"],
            "batch256_median": b256["median"] / base["batch256_median_ms"]["median"],
            "state_dict_bytes": entry["state_dict_bytes"]["median"] / base["state_dict_bytes"]["median"]}
    return out


# ------------------------------------------------------------------ parent: full run
def run_efficiency(models: dict, rows: dict, cfg: dict, out_dir: Path | str, *, pilot: bool, conversions: dict | None = None,
                   total_parameters: int | None = None, run_one=None, banner: str | None = None,
                   threads: int | None = None) -> dict:
    """Measure V1-V4 and write efficiency.json (never overwrites).

    models: {'V1': {'kind': 'fp32', 'checkpoint', 'architecture'}, 'V2': {'kind': 'int8', 'state',
    'float_checkpoint', 'calibration_path', 'scope', 'backend', 'architecture'}, ...};
    rows: the select_rows() result; conversions: {'V2': conversion dict, 'V4': ...}.
    """
    validate_config(cfg)
    out_dir = Path(out_dir)
    if (out_dir / "efficiency.json").exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir / 'efficiency.json'}")
    base = list(cfg["variants"])
    if list(models) not in (base, base + list(QAT_VARIANTS)):
        raise ValueError(f"models must be {base} in order, optionally followed by {list(QAT_VARIANTS)}")
    variants = list(models)
    mode = "pilot" if pilot else "full"
    b1, b256 = cfg["batch1"], cfg["batch256"]
    prefix = "pilot_" if pilot else ""
    repetitions = cfg["repetitions"][mode]
    threads = cfg["threads"]["primary"] if threads is None else threads  # the campaign adds a second pass (secondary)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "fixed_rows.npy"
    np.save(rows_path, rows["x"])
    atomic_json(out_dir / "fixed_rows.json", {"ids": rows["ids"], "id_sha256": rows["id_sha256"], "x_sha256": rows["x_sha256"],
                                               "seed_rule": rows["seed_rule"], "split_seed": rows["split_seed"],
                                               "partition": cfg["fixed_inputs"]["partition"], "banner": banner})
    specs = {}
    for variant, model in models.items():
        specs[variant] = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in model.items()}, "variant": variant,
                          "threads": threads, "rows_path": str(rows_path), "rows_sha256": rows["x_sha256"],
                          "batch_size": b256["batch_size"],
                          "batch1": {"warmup": b1[f"{prefix}warmup"], "timed": b1[f"{prefix}timed"]},
                          "batch256": {"warmup": b256[f"{prefix}warmup"], "timed": b256[f"{prefix}timed"]}}
    order = interleaved_order(variants, repetitions)
    if run_one is None:
        run_one = subprocess_runner(out_dir, cfg["child"]["timeout_seconds"], threads)
    executed = run_schedule(specs, order, run_one, max_retries=cfg["child"]["max_retries"],
                            cpu_load=lambda: psutil.cpu_percent(interval=0.5))
    coverage = {v: coverage_from_conversion((conversions or {}).get(v), total_parameters or 0) for v in variants}
    record = {
        "banner": banner, "claim": cfg["claim"], "mode": mode, "threads": threads, "repetitions": repetitions,
        "planned_order": [f"{v}" for _, v in order], "run_order": executed["run_order"],
        "interleaved": [e["variant"] for e in executed["run_order"] if not e["retry"]] == [v for _, v in order],
        "any_retry": any(e["retry"] for e in executed["run_order"]),
        "any_failed": any(e["status"] == "failed" for e in executed["run_order"]),
        "protocol": {k: cfg[k] for k in ("timed_region", "excluded_from_timing", "memory", "artifact_size", "inference")},
        "warmup_and_timed": {"batch1": specs["V1"]["batch1"], "batch256": specs["V1"]["batch256"]},
        "fixed_rows": {"count": len(rows["ids"]), "id_sha256": rows["id_sha256"], "x_sha256": rows["x_sha256"]},
        "mixed_precision_coverage": coverage, "measurements": executed["measurements"],
        "summary": summarize_runs(executed["measurements"], variants),
        "peak_memory_metric": sorted({m["result"].get("peak_memory_metric") or "peak_wset" for m in executed["measurements"]}),
        "memory_comparison_note": MEMORY_COMPARISON_NOTE, "variants": variants,
        "no_claim": "Outcomes are descriptive observations on one laptop CPU; no speed or memory advantage is assumed or claimed."}
    atomic_json(out_dir / "efficiency.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", action="store_true", help="internal: run one measurement in this process")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not args.child:
        parser.error("the harness is driven by research.m2_pilot / research.m2_campaign; use --child --spec --out")
    if not (args.spec and args.out):
        parser.error("--child needs --spec and --out")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    atomic_json(args.out, child_measure(spec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
