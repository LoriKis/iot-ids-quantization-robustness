"""Timing estimate command (DESIGN 5): probe, estimate, lock if feasible, print, stop.

    python -m m2_deadline.timing --hours-remaining H [--v4-root P --out P --config P
                                  --prior-evidence P ... --allow-contention --force-relock --max-minutes M]

Uses training and validation data only. No accuracy, F1 or loss is computed or recorded, so the epoch
choice depends on runtime alone. The probe function is injectable (``main(measure_fn=...)``) so the
decision logic is tested with synthetic measurements; ``measure`` itself runs only on the VM.

Exit codes: 0 ok; 2 usage/config/data error or probe time limit (nothing written);
3 guard (smoke path, existing lock/estimate, relock with markers); 4 infeasible; 5 contention or noisy
timings without --allow-contention; 130 interrupted.

Import-time rule: standard library + numpy only (torch, psutil and research.* are imported lazily).
Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import copy
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from . import budget, markers, protocol, sampling
from .common import (
    DeadlineError,
    EXIT_OK,
    EXIT_USAGE,
    ConfigError,
    ContentionError,
    GuardError,
    InfeasibleError,
    N_CLASSES,
    PATHS,
    RUN_ID,
    atomic_write_json,
    exit_code_for,
    load_config,
    out_path,
    read_json,
    setup_logging,
    sha256_file,
    utc_now,
    utc_stamp,
)

OVERLAY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = OVERLAY_ROOT / "artifacts" / RUN_ID
DEFAULT_CONFIG = OVERLAY_ROOT / "configs" / "m2_deadline_v1.json"
PRIOR_DEFAULT_REL = "artifacts/m2a-pilot/pilot_runtime.json"
BUSY_NAMES = ("research.m2b_campaign", "pytest")


class ProbeTimeoutError(DeadlineError):
    """The probe exceeded --max-minutes. Nothing is locked (exit 2)."""

    exit_code = EXIT_USAGE


# ---------------------------------------------------------------- V4 root and arguments
def check_v4_root(v4_root: Any) -> Path:
    root = Path(str(v4_root))
    venv_python = (root / ".venv" / "bin" / "python", root / ".venv" / "Scripts" / "python.exe")
    problems = []
    if not (root / "MANIFEST.sha256").is_file():
        problems.append("MANIFEST.sha256")
    if not (root / "research").is_dir():
        problems.append("research/")
    if not any(p.exists() for p in venv_python):
        problems.append(".venv/bin/python")
    if problems:
        raise ConfigError("V4 root %s lacks: %s" % (root, ", ".join(problems)))
    return root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m m2_deadline.timing",
        description="Measure runtime on train+validation data only, print the plan estimate and stop. "
        "Writes timing_estimate.json and (if feasible) budget_lock.json. Starts no training run.",
    )
    p.add_argument("--hours-remaining", type=float, default=None, help="actual hours left until the deadline (0 < H <= 12); required")
    p.add_argument("--v4-root", default=os.environ.get("V4_ROOT"), help="verified V4 package folder (default: env V4_ROOT)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="run output folder (default: artifacts/m2-deadline-v1 in the overlay)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="overlay config JSON")
    p.add_argument("--prior-evidence", action="append", default=None, help="optional prior timing JSON (repeatable); informational only")
    p.add_argument("--allow-contention", action="store_true", help="accept a contended or noisy probe and record it")
    p.add_argument("--force-relock", action="store_true", help="archive the old estimate and lock and write new ones (refused once a stage marker exists)")
    p.add_argument("--max-minutes", type=float, default=None, help="abort the probe after this many minutes (default from config)")
    return p


# ---------------------------------------------------------------- contention
def default_contention(config: Dict[str, Any]) -> Dict[str, Any]:
    """CPU contention check with psutil (never signals or kills anything)."""
    import psutil

    threshold = float(config["timing_probe"]["contention_cpu_percent"])
    me = os.getpid()
    procs = list(psutil.process_iter(["pid", "name", "cmdline"]))
    for p in procs:
        try:
            p.cpu_percent(None)
        except (psutil.Error, OSError):
            pass
    load = float(psutil.cpu_percent(interval=1.0))
    busy: List[Dict[str, Any]] = []
    for p in procs:
        try:
            if p.info["pid"] == me:
                continue
            pct = float(p.cpu_percent(None))
            cmd = " ".join(p.info.get("cmdline") or [])
            named = any(n in cmd for n in BUSY_NAMES)
            if pct > threshold or named:
                busy.append({"pid": p.info["pid"], "name": p.info.get("name"), "cpu_percent": pct, "named": named,
                             "cmdline": cmd[:200]})
        except (psutil.Error, OSError, KeyError):
            continue
    return {"load": load, "busy_processes": busy, "contended": bool(busy or load > threshold)}


def merge_contention(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    busy = list(before.get("busy_processes", []))
    seen = {(b.get("pid"), b.get("name")) for b in busy}
    busy += [b for b in after.get("busy_processes", []) if (b.get("pid"), b.get("name")) not in seen]
    return {
        "contended": bool(before.get("contended") or after.get("contended")),
        "load_before": float(before.get("load", 0.0)),
        "load_after": float(after.get("load", 0.0)),
        "busy_processes": busy,
    }


# ---------------------------------------------------------------- prior evidence (informational only)
def prior_evidence_record(path: Any, config: Dict[str, Any], measured_ratio: Optional[float], host_system: str) -> Dict[str, Any]:
    path = Path(str(path))
    rec: Dict[str, Any] = {"path": str(path), "used_in_estimate": False}
    if not path.is_file():
        rec.update(sha256=None, error="file not found", compatibility={"compatible": False})
        return rec
    rec["sha256"] = sha256_file(path)
    try:
        doc = read_json(path)
        platform_text = str(doc.get("environment", {}).get("platform", ""))
        compat = {
            "os_match": platform_text.split("-")[0].lower() == host_system.lower(),
            "split_match": doc.get("split_dir") == config["split_dir"],
            "epochs_match": doc.get("epochs") in config["training"]["epoch_candidates"],
            "threads_match": (doc.get("threads") or {}).get("training") == config["training"]["threads"],
        }
        compat["compatible"] = all(compat.values())
        rec["compatibility"] = compat
        pilot = (doc.get("at_cost") or {}).get("measured_wall_clock_ratio")
        if pilot is not None and measured_ratio is not None and float(pilot) > 0:
            rec["adv_over_ord_ratio"] = {
                "measured": float(measured_ratio),
                "pilot": float(pilot),
                "within_factor_2": bool(0.5 <= float(measured_ratio) / float(pilot) <= 2.0),
            }
    except (ValueError, OSError, AttributeError, TypeError) as exc:
        rec.update(error="%s: %s" % (type(exc).__name__, exc), compatibility={"compatible": False})
    return rec


# ---------------------------------------------------------------- probe environment and measurement (VM only)
class ProbeEnv:
    """What ``measure`` needs: config, paths, deps, logger and the time limit."""

    def __init__(self, *, config: Dict[str, Any], v4_root: Path, out: Path, deps: Any, log: Any, max_minutes: float) -> None:
        self.config = config
        self.v4_root = v4_root
        self.out = out
        self.deps = deps
        self.log = log
        self.started = time.monotonic()
        self.limit_s = float(max_minutes) * 60.0
        self.probe_dir = out / PATHS["probe_models_dir"] / utc_stamp()

    @property
    def split_path(self) -> Path:
        return self.v4_root / self.config["split_dir"] / "split_m2.json"

    @property
    def scaler_path(self) -> Path:
        return self.v4_root / self.config["split_dir"] / "scaler_minmax.json"

    def sibling(self, name: str) -> Path:
        return self.v4_root / self.config["siblings"][name]

    def check_deadline(self, where: str) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed > self.limit_s:
            raise ProbeTimeoutError("probe exceeded --max-minutes (%.1f min) during %s; nothing locked" % (self.limit_s / 60.0, where))


def measure(env: ProbeEnv) -> Dict[str, Any]:
    """Probe the real code paths on untrained models. Train and validation arrays only; no test array, no accuracy."""
    deps, cfg, log = env.deps, env.config, env.log
    probe = cfg["timing_probe"]
    seed = int(cfg["seed"])
    threads = int(cfg["training"]["threads"])
    env.probe_dir.mkdir(parents=True, exist_ok=True)

    split = read_json(env.split_path)
    n_test = len(split["row_ids"]["test"])  # ids only; no test feature or label is read
    split_seed = int(split["split_seed"])
    train_cfg = deps.mt.load_config(env.sibling("training"))
    quant_cfg = deps.mq.load_config(env.sibling("quantization"))
    att_cfg = deps.ma.load_config(env.sibling("attacks"))
    eff_cfg_real = deps.me.load_config(env.sibling("efficiency"))
    arch = train_cfg["architecture"]

    # 1. data (train + validation only)
    t0 = time.perf_counter()
    arrays = deps.m2_split.load_arrays(env.split_path, env.scaler_path, ("train", "validation"))
    load_arrays_s = time.perf_counter() - t0
    if "test" in arrays:
        raise GuardError("the timing probe must never hold the test partition")
    train, val = arrays["train"], arrays["validation"]
    n_train, n_val = int(len(train["ids"])), int(len(val["ids"]))
    batch = int(train_cfg["training"]["batch_size"])
    if batch != int(cfg["training"]["batch_size_expected"]):
        raise ConfigError("training batch size %d differs from the configured %d" % (batch, cfg["training"]["batch_size_expected"]))
    steps_o, steps_a = int(probe["train_steps_ordinary"]), int(probe["train_steps_adversarial"])
    if n_train < max(steps_o, steps_a) * batch:
        raise ConfigError("n_train=%d is smaller than the probe needs (%d rows)" % (n_train, max(steps_o, steps_a) * batch))
    env.check_deadline("data load")

    # 2. training batches (out_dir=None: nothing is saved)
    deps.configure_inference(threads=threads)
    probes: Dict[str, Any] = {}
    for kind, adversarial, n_steps in (("ordinary", False, steps_o), ("adversarial", True, steps_a)):
        rows = n_steps * batch
        result = deps.mt.train_unit(
            train_cfg, train["x"][:rows], train["y"][:rows], seed, out_dir=None,
            adversarial=adversarial, epochs=1, threads=threads,
        )
        step_seconds = list(result["log"]["step_seconds"])
        if len(step_seconds) != n_steps:
            raise GuardError("%s probe ran %d updates, expected %d" % (kind, len(step_seconds), n_steps))
        probes[kind] = {
            "model": result["model"],
            "summary": budget.summarize_steps(step_seconds, int(probe["warmup_steps_dropped"]), float(probe["p90_weight"])),
        }
        log.info("probe %s: mean %.3f s, p90 %.3f s, cv %.2f", kind, probes[kind]["summary"]["mean"], probes[kind]["summary"]["p90"], probes[kind]["summary"]["step_cv"])
        env.check_deadline("%s training probe" % kind)
    model_o, model_a = probes["ordinary"]["model"], probes["adversarial"]["model"]

    # 3. checkpoint write
    ckpt_dir = env.probe_dir / "ckpt_probe"
    opt = deps.make_probe_optimizer(train_cfg, model_o)
    t0 = time.perf_counter()
    deps.mt.save_epoch_checkpoint(
        ckpt_dir, epoch=1, model=model_o, opt=opt, updates=steps_o, epoch_losses=[0.0], step_seconds=[], attack_seconds=[],
        init_state_hash="timing-probe", seed=seed, adversarial=False, threads=threads,
        fingerprints={"kind": "timing-probe"}, extras={}, keep_last=int(cfg["training"]["keep_last"]),
    )
    checkpoint_write_s = time.perf_counter() - t0
    env.check_deadline("checkpoint write")

    # 4. PTQ (calibration build + conversion + save + reload check), twice: V2 from the ordinary probe, V4 from the adversarial probe
    per_class_cal = int(quant_cfg["ptq"]["calibration_per_class"])
    val_seq = deps.sequences(val["x"])
    t0 = time.perf_counter()
    cal = deps.mq.calibration_ids({"train": train}, per_class_cal, split_seed)
    cal_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    res2 = deps.mq.ptq_unit(model_o, cal, quant_cfg, env.probe_dir / "V2", variant="V2", eval_x=val_seq[:256])
    ptq_v2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    res4 = deps.mq.ptq_unit(model_a, cal, quant_cfg, env.probe_dir / "V4", variant="V4", eval_x=val_seq[:256])
    ptq_v4 = time.perf_counter() - t0
    ptq_s = max(cal_s + ptq_v2, cal_s + ptq_v4)
    env.check_deadline("PTQ")

    # 5. validation predict paths (batch 64, 1 thread, fbgemm: the same as val_check and test_clean)
    deps.configure_inference(threads=1)
    t0 = time.perf_counter()
    deps.predict(model_o, val["x"], 64)
    val_fp32 = time.perf_counter() - t0
    t0 = time.perf_counter()
    deps.predict(res2["model"], val["x"], 64)
    val_int8 = time.perf_counter() - t0
    env.check_deadline("validation predict")

    # 6. attack cost per example on a k-per-class validation subset with the pgd5 config
    sub = sampling.stratified_subset(val["ids"], val["y"], int(probe["attack_probe_per_class"]), split_seed, N_CLASSES,
                                     cfg["robustness"]["subset_seed_name"])
    index = {int(i): n for n, i in enumerate(val["ids"])}
    rows_idx = np.array([index[int(i)] for i in sub["ids"]], dtype=np.int64)
    x_sub = np.ascontiguousarray(np.asarray(val["x"], dtype=np.float32)[rows_idx])
    pgd_cfg = protocol.build_pgd5_cfg(att_cfg, cfg["robustness"]["attacks"]["pgd5"])
    deps.ma.validate_config(pgd_cfg)  # validation-partition config
    bpda = deps.quantization.bpda_model(res2["model"], res2["conversion"]["requested"], cal["x"][:2])
    targets = {
        "fp32_direct": ({"fp32": model_o}, {"fp32": "V1"}, "V1"),
        "bpda_int8": ({"int8": res2["model"], "bpda": bpda}, {"fp32": "V1", "int8": "V2", "bpda": "V2_bpda"}, "V2"),
    }
    attack_s: Dict[str, Dict[str, float]] = {}
    for source, (models, model_ids, model_id) in targets.items():
        attack_s[source] = {}
        for label, attack, eps in (("clean", "fgsm", 0.0), ("fgsm", "fgsm", 0.05), ("pgd5", "pgd5", 0.05)):
            t0 = time.perf_counter()
            deps.ma.evaluate_unit(
                source=source, attack_name=attack, epsilon=eps, x=x_sub, y=sub["labels"], ids=sub["ids"], models=models,
                cfg=pgd_cfg, train_seed=seed, model_id=model_id, evaluation_partition="validation",
                model_ids=model_ids, raise_on_invalid=False,
            )
            attack_s[source][label] = (time.perf_counter() - t0) / float(sub["n_actual"])
            env.check_deadline("attack probe %s/%s" % (source, label))
    attack_s["probe_n"] = int(sub["n_actual"])  # type: ignore[assignment]

    # 7. one efficiency pass with 1 repetition, scaled by the real repetition count
    eff_probe_cfg = copy.deepcopy(eff_cfg_real)
    eff_probe_cfg["repetitions"] = {"full": 1, "pilot": 1}
    deps.save_fp32_checkpoint(model_o, env.probe_dir / "V1" / "checkpoint.pt", arch)
    deps.save_fp32_checkpoint(model_a, env.probe_dir / "V3" / "checkpoint.pt", arch)
    deps.save_tensor(cal["x"], env.probe_dir / "calibration_x.pt")
    ptq = quant_cfg["ptq"]

    def int8_spec(mid: str, parent: str) -> Dict[str, Any]:
        return {"kind": "int8", "state": env.probe_dir / mid / "state.pt",
                "float_checkpoint": env.probe_dir / parent / "checkpoint.pt",
                "calibration_path": env.probe_dir / "calibration_x.pt", "scope": ptq["scope"],
                "backend": ptq["backend"], "architecture": arch}

    specs = {
        "V1": {"kind": "fp32", "checkpoint": env.probe_dir / "V1" / "checkpoint.pt", "architecture": arch},
        "V2": int8_spec("V2", "V1"),
        "V3": {"kind": "fp32", "checkpoint": env.probe_dir / "V3" / "checkpoint.pt", "architecture": arch},
        "V4": int8_spec("V4", "V3"),
    }
    rows = deps.me.select_rows(val["ids"], val["x"], int(eff_probe_cfg["fixed_inputs"]["count"]), split_seed)
    t0 = time.perf_counter()
    record = deps.me.run_efficiency(
        specs, rows, eff_probe_cfg, env.probe_dir / "efficiency", pilot=False,
        conversions={"V2": res2["conversion"], "V4": res4["conversion"]},
        total_parameters=deps.parameter_count(model_o), banner="TIMING-PROBE", threads=int(cfg["efficiency"]["threads"]),
    )
    eff_wall = time.perf_counter() - t0
    if record.get("any_failed"):
        raise GuardError("the efficiency probe had a failed measurement; see %s" % (env.probe_dir / "efficiency"))
    efficiency_s = eff_wall * int(eff_cfg_real["repetitions"]["full"])
    env.check_deadline("efficiency probe")

    return {
        "data": {"n_train": n_train, "n_val": n_val, "n_test": int(n_test)},
        "train_step_seconds": {"ordinary": probes["ordinary"]["summary"], "adversarial": probes["adversarial"]["summary"]},
        "checkpoint_write_s": checkpoint_write_s,
        "ptq_s": ptq_s,
        "load_arrays_s": load_arrays_s,
        "val_eval_s": {"fp32": val_fp32, "int8": val_int8, "n_rows": n_val},
        "attack_seconds_per_example": attack_s,
        "efficiency_s": efficiency_s,
    }


# ---------------------------------------------------------------- estimate document and printing
def _host_info() -> Dict[str, Any]:
    torch_module = sys.modules.get("torch")
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": getattr(torch_module, "__version__", None),
        "python": platform.python_version(),
    }


def build_estimate(
    *,
    measurements: Dict[str, Any],
    config: Dict[str, Any],
    config_sha256: str,
    v4_manifest_sha256: str,
    hours_remaining: float,
    contention: Dict[str, Any],
    prior_evidence: List[Dict[str, Any]],
    allow_contention: bool,
) -> Dict[str, Any]:
    planned = budget.plan(measurements, config, hours_remaining)
    noisy = budget.is_noisy(measurements["train_step_seconds"], float(config["timing_probe"]["max_step_cv"]))
    meas = {k: v for k, v in measurements.items() if k != "data"}
    meas["noisy"] = bool(noisy)
    meas["noisy_accepted"] = bool(noisy and allow_contention)
    contention = dict(contention)
    contention["contention_accepted"] = bool(contention.get("contended") and allow_contention)
    return {
        "schema": 1,
        "kind": "m2_deadline_timing_estimate",
        "run_id": config["run_id"],
        "created_utc": utc_now(),
        "host": _host_info(),
        "config_sha256": config_sha256,
        "v4_manifest_sha256": v4_manifest_sha256,
        "inputs": planned["inputs"],
        "data": planned["data"],
        "contention": contention,
        "measurements": meas,
        "prior_evidence": prior_evidence,
        "candidates": planned["candidates"],
        "decision": planned["decision"],
        "no_performance_inspected": True,
    }


def format_plan(estimate: Dict[str, Any], locked: bool, out: Path, v4_root: Path) -> str:
    d = estimate["decision"]
    inputs = estimate["inputs"]
    if d["status"] == "feasible":
        key = (d["epochs"], d["per_class"])
    else:
        key = (d["smallest_plan"]["epochs"], d["smallest_plan"]["per_class"])
    cand = next(c for c in estimate["candidates"] if (c["epochs"], c["per_class"]) == key)
    lines = ["", "M2 deadline timing estimate (runtime only; no accuracy or F1 inspected)",
             "plan: %d epochs, %d test rows per class%s" % (key[0], key[1], "" if d["status"] == "feasible" else " (SMALLEST PLAN, does not fit)"),
             "", "%-20s %12s" % ("component", "seconds")]
    for name, seconds in cand["components_s"].items():
        lines.append("%-20s %12.0f" % (name, seconds))
    lines += [
        "%-20s %12.0f" % ("raw total", cand["raw_s"]),
        "",
        "safety multiplier    %.2f" % inputs["safety_multiplier"],
        "planned hours        %.2f" % cand["planned_hours"],
        "cap hours            %.2f  = min(%.1f, %.2f - %.1f)" % (inputs["cap_hours"], inputs["hard_cap_hours"], inputs["hours_remaining"], inputs["reserve_hours"]),
        "decision             %s%s" % (d["status"], "" if d["status"] == "feasible" else " (nothing locked, nothing launched)"),
    ]
    if estimate["contention"].get("contended") or estimate["measurements"].get("noisy"):
        lines.append("warning              contended=%s noisy=%s" % (estimate["contention"].get("contended"), estimate["measurements"].get("noisy")))
    if locked:
        lines += ["", "budget lock written: %s" % out_path(out, "budget_lock"),
                  "launch (separate command): python -m m2_deadline.campaign launch --hours-remaining <HOURS_NOW> --out %s --v4-root %s" % (out, v4_root),
                  "  or: bash scripts/vm_deadline/launch.sh <HOURS_NOW>"]
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def _archive_old(out: Path) -> Optional[Path]:
    """Move an existing estimate and lock to timing/locks_archive/<utc>/ (never deleted)."""
    moved = False
    dest = out_path(out, "locks_archive_dir") / utc_stamp()
    for key in ("timing_estimate", "budget_lock"):
        p = out_path(out, key)
        if p.is_file():
            dest.mkdir(parents=True, exist_ok=True)
            os.replace(str(p), str(dest / p.name))
            moved = True
    return dest if moved else None


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    measure_fn: Optional[Callable[[ProbeEnv], Dict[str, Any]]] = None,
    contention_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    deps: Any = None,
) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return _run(args, measure_fn, contention_fn, deps)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return exit_code_for(KeyboardInterrupt())
    except DeadlineError as exc:
        print("%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return exit_code_for(exc)


def _run(args: argparse.Namespace, measure_fn: Any, contention_fn: Any, deps: Any) -> int:
    # 1. validate everything before anything is written
    hours = budget.validate_hours(args.hours_remaining)
    config_path = Path(args.config)
    config = load_config(config_path)
    config_sha256 = sha256_file(config_path)
    if not args.v4_root:
        raise ConfigError("--v4-root (or env V4_ROOT) is required")
    v4_root = check_v4_root(args.v4_root)
    out = Path(args.out)
    markers.assert_not_smoke(out)
    max_minutes = float(args.max_minutes) if args.max_minutes is not None else float(config["budget"]["max_probe_minutes"])
    if not max_minutes > 0:
        raise ConfigError("--max-minutes must be > 0")
    estimate_path = out_path(out, "timing_estimate")
    lock_path = out_path(out, "budget_lock")
    if args.force_relock:
        if markers.any_marker_exists(out):
            raise GuardError("--force-relock refused: stage markers exist under %s (the budget cannot change after a stage ran)" % out_path(out, "markers_dir"))
    elif lock_path.exists() or estimate_path.exists():
        raise GuardError("a timing estimate or budget lock already exists under %s; use --force-relock to archive and replace it (only before launch)" % out_path(out, "timing_dir"))

    prior_paths = list(args.prior_evidence) if args.prior_evidence else []
    if not prior_paths and (v4_root / PRIOR_DEFAULT_REL).is_file():
        prior_paths = [str(v4_root / PRIOR_DEFAULT_REL)]

    out.mkdir(parents=True, exist_ok=True)
    log, log_path = setup_logging(out, "timing")
    log.info("timing: hours_remaining=%.2f out=%s v4=%s", hours, out, v4_root)

    # 2. contention check before measuring
    check = contention_fn if contention_fn is not None else default_contention
    before = check(config)
    if before.get("contended") and not args.allow_contention:
        raise ContentionError("CPU contention before the probe (load %.0f%%, busy: %s); stop other work or pass --allow-contention"
                              % (before.get("load", 0.0), [b.get("name") for b in before.get("busy_processes", [])]))

    # 3. probe
    if measure_fn is None:
        if deps is None:
            from .stages import DefaultDeps

            deps = DefaultDeps()
        measure_fn = measure
    env = ProbeEnv(config=config, v4_root=v4_root, out=out, deps=deps, log=log, max_minutes=max_minutes)
    measurements = measure_fn(env)
    after = check(config)
    contention = merge_contention(before, after)

    # 4. estimate, prior evidence, decision
    ts = measurements["train_step_seconds"]
    ratio = None
    try:
        ratio = float(ts["adversarial"]["mean"]) / float(ts["ordinary"]["mean"])
    except (KeyError, ZeroDivisionError, TypeError):
        pass
    prior = [prior_evidence_record(p, config, ratio, platform.system()) for p in prior_paths]
    estimate = build_estimate(
        measurements=measurements, config=config, config_sha256=config_sha256,
        v4_manifest_sha256=sha256_file(v4_root / "MANIFEST.sha256"), hours_remaining=hours, contention=contention,
        prior_evidence=prior, allow_contention=bool(args.allow_contention),
    )

    # 5. write (archive first when relocking), lock only if feasible and not noisy/contended
    lock_error: Optional[DeadlineError] = None
    lock: Optional[Dict[str, Any]] = None
    try:
        budget.check_lockable(estimate, bool(args.allow_contention))
    except (InfeasibleError, ContentionError) as exc:
        lock_error = exc
    if args.force_relock:
        archived = _archive_old(out)
        if archived is not None:
            log.info("archived previous estimate/lock to %s", archived)
    estimate_sha = atomic_write_json(estimate_path, estimate)
    if lock_error is None:
        lock = budget.build_lock(estimate, estimate_sha, config_sha256, utc_now(), bool(args.allow_contention))
        atomic_write_json(lock_path, lock)
    print(format_plan(estimate, lock is not None, out, v4_root))
    print("\nestimate written: %s (log: %s)" % (estimate_path, log_path))
    if lock_error is not None:
        log.error("no lock written: %s", lock_error)
        print("NO LOCK WRITTEN: %s" % lock_error, file=sys.stderr)
        raise lock_error
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
