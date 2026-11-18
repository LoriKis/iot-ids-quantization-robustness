"""Runtime budget planning (DESIGN 5.2) and the launch/resume deadline checks (DESIGN 6.3).

Pure module: standard library + numpy. Python 3.9 compatible. No I/O.

Measurement dict layout accepted by :func:`plan` (the D5.3 ``measurements`` block plus ``data``)::

    {"data": {"n_train", "n_val", "n_test"[, "updates_per_epoch"]},
     "train_step_seconds": {"ordinary": {"mean", "p90"[, "step_cv"]}, "adversarial": {...}},
     "checkpoint_write_s", "ptq_s", "load_arrays_s",
     "val_eval_s": {"fp32", "int8"[, "n_rows"]},
     "attack_seconds_per_example": {"fp32_direct": {"clean", "fgsm", "pgd5"}, "bpda_int8": {...}},
     "efficiency_s"}

Nothing here reads accuracy, loss or F1, so the choice of epochs depends on runtime only.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import (
    COMPONENT_NAMES,
    ConfigError,
    ContentionError,
    GuardError,
    InfeasibleError,
    STAGES,
    STAGE_COMPONENT,
)

MAX_HOURS = 12.0
SELECTION_RULE = "first fitting of (25,10),(25,5),(10,10),(10,5),(5,10),(5,5); runtime only"


# ---------------------------------------------------------------- validation helpers
def validate_hours(hours_remaining: Any) -> float:
    """``0 < H <= 12`` and finite. Raises ``ConfigError`` (exit 2) otherwise, including for ``None``."""
    if hours_remaining is None:
        raise ConfigError("--hours-remaining is required")
    if isinstance(hours_remaining, bool) or not isinstance(hours_remaining, (int, float)):
        raise ConfigError("--hours-remaining must be a number, got %r" % (hours_remaining,))
    h = float(hours_remaining)
    if not math.isfinite(h) or h <= 0 or h > MAX_HOURS:
        raise ConfigError("--hours-remaining must satisfy 0 < H <= %g, got %r" % (MAX_HOURS, hours_remaining))
    return h


def cap_hours(hours_remaining: float, hard_cap_hours: float = 9.0, reserve_hours: float = 3.0) -> float:
    """``min(hard_cap, H - reserve)``. Zero or negative means nothing can be planned."""
    return min(float(hard_cap_hours), float(hours_remaining) - float(reserve_hours))


def _nonneg(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ConfigError("%s must be a number, got %r" % (name, value))
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ConfigError("%s must be finite and >= 0, got %r" % (name, value))
    return v


# ---------------------------------------------------------------- step statistics
def step_used(mean: float, p90: float, p90_weight: float = 0.9) -> float:
    """``max(mean, 0.9 * p90)``: the mean because totals are sums, the p90 term guards a lucky quiet probe."""
    return max(_nonneg(mean, "mean"), float(p90_weight) * _nonneg(p90, "p90"))


def summarize_steps(step_seconds: Sequence[float], warmup: int = 2, p90_weight: float = 0.9) -> Dict[str, Any]:
    """Statistics of probe step times after dropping the first ``warmup`` steps."""
    steps = [_nonneg(s, "step_seconds") for s in step_seconds][int(warmup):]
    if len(steps) < 2:
        raise ConfigError("need at least 2 timed steps after dropping %d warm-up steps" % warmup)
    arr = np.asarray(steps, dtype=np.float64)
    mean = float(arr.mean())
    p90 = float(np.percentile(arr, 90))
    std = float(arr.std())  # population standard deviation
    cv = std / mean if mean > 0 else float("inf")
    return {
        "n_timed": int(len(arr)),
        "mean": mean,
        "p90": p90,
        "std": std,
        "step_cv": cv,
        "step_used": step_used(mean, p90, p90_weight),
    }


def is_noisy(train_step_seconds: Dict[str, Dict[str, Any]], max_step_cv: float = 0.5) -> bool:
    """True when any parent's probe step time has ``step_cv > max_step_cv``."""
    return any(float(train_step_seconds[kind]["step_cv"]) > float(max_step_cv) for kind in train_step_seconds)


# ---------------------------------------------------------------- the plan
def candidate_grid(config: Dict[str, Any]) -> List[Tuple[int, int]]:
    """(epochs, per_class) in decision order: epochs descending first, then per-class target."""
    return [
        (int(e), int(k))
        for e in config["training"]["epoch_candidates"]
        for k in config["budget"]["per_class_candidates"]
    ]


def _check_measurements(m: Dict[str, Any]) -> None:
    for key in ("data", "train_step_seconds", "checkpoint_write_s", "ptq_s", "load_arrays_s", "val_eval_s",
                "attack_seconds_per_example", "efficiency_s"):
        if key not in m:
            raise ConfigError("measurements lack %r" % key)
    for key in ("n_train", "n_val", "n_test"):
        if _nonneg(m["data"].get(key), "data." + key) <= 0:
            raise ConfigError("data.%s must be > 0" % key)
    for kind in ("ordinary", "adversarial"):
        s = m["train_step_seconds"][kind]
        if _nonneg(s["mean"], "%s.mean" % kind) <= 0:
            raise ConfigError("train_step_seconds.%s.mean must be > 0" % kind)
        _nonneg(s["p90"], "%s.p90" % kind)
    for key in ("checkpoint_write_s", "ptq_s", "load_arrays_s", "efficiency_s"):
        _nonneg(m[key], key)
    for key in ("fp32", "int8"):
        _nonneg(m["val_eval_s"][key], "val_eval_s." + key)
    for target in ("fp32_direct", "bpda_int8"):
        for key in ("clean", "fgsm", "pgd5"):
            _nonneg(m["attack_seconds_per_example"][target][key], "attack_seconds_per_example.%s.%s" % (target, key))


def components_seconds(m: Dict[str, Any], config: Dict[str, Any], epochs: int, per_class: int) -> Dict[str, float]:
    """Component seconds of one candidate (DESIGN 5.2), purely sequential."""
    b = config["budget"]
    weight = float(config["timing_probe"]["p90_weight"])
    n_train = int(m["data"]["n_train"])
    n_val = int(m["data"]["n_val"])
    n_test = int(m["data"]["n_test"])
    batch = int(config["training"]["batch_size_expected"])
    updates = int(math.ceil(n_train / float(batch)))
    given = m["data"].get("updates_per_epoch")
    if given is not None and int(given) != updates:
        raise ConfigError("data.updates_per_epoch=%r disagrees with ceil(n_train/%d)=%d" % (given, batch, updates))

    ts = m["train_step_seconds"]
    ck = float(m["checkpoint_write_s"])
    step_ord = step_used(ts["ordinary"]["mean"], ts["ordinary"]["p90"], weight)
    step_adv = step_used(ts["adversarial"]["mean"], ts["adversarial"]["p90"], weight)
    val_pair = 2.0 * float(m["val_eval_s"]["fp32"]) + 2.0 * float(m["val_eval_s"]["int8"])
    n_ub = int(b["n_classes"]) * int(per_class)

    def rob_unit(target: str) -> float:
        a = m["attack_seconds_per_example"][target]
        return n_ub * (float(a["clean"]) + 2.0 * float(a["fgsm"]) + 2.0 * float(a["pgd5"]))

    return {
        "train_ordinary": epochs * (updates * step_ord + ck),
        "train_adversarial": epochs * (updates * step_adv + ck),
        "ptq": 2.0 * float(m["ptq_s"]),
        "validation": val_pair,
        "test": val_pair * n_test / float(n_val),
        "robustness": 2.0 * rob_unit("fp32_direct") + 2.0 * rob_unit("bpda_int8"),
        "efficiency": float(m["efficiency_s"]),
        "report": float(b["report_allowance_seconds"]),
        "overhead": int(b["n_stage_processes"]) * (float(m["load_arrays_s"]) + float(b["stage_overhead_seconds"])),
    }


def plan(measurements: Dict[str, Any], config: Dict[str, Any], hours_remaining: Any) -> Dict[str, Any]:
    """Compute every candidate and the runtime-only decision. No I/O, no randomness.

    Returns ``{"inputs", "data", "candidates", "decision"}`` in the D5.3 layout.
    """
    h = validate_hours(hours_remaining)
    _check_measurements(measurements)
    b = config["budget"]
    safety = float(b["safety_multiplier"])
    cap = cap_hours(h, b["hard_cap_hours"], b["reserve_hours"])

    candidates: List[Dict[str, Any]] = []
    for epochs, per_class in candidate_grid(config):
        comp = components_seconds(measurements, config, epochs, per_class)
        raw = float(sum(comp[name] for name in COMPONENT_NAMES))
        planned = raw * safety / 3600.0
        candidates.append(
            {
                "epochs": epochs,
                "per_class": per_class,
                "components_s": comp,
                "raw_s": raw,
                "planned_hours": planned,
                "fits": bool(cap > 0 and planned <= cap),
            }
        )

    chosen = next((c for c in candidates if c["fits"]), None)
    if chosen is not None:
        decision = {
            "status": "feasible",
            "epochs": chosen["epochs"],
            "per_class": chosen["per_class"],
            "robustness_n_upper_bound": int(b["n_classes"]) * chosen["per_class"],
            "planned_hours": chosen["planned_hours"],
            "cap_hours": cap,
            "rule": SELECTION_RULE,
        }
    else:
        smallest = candidates[-1]
        decision = {
            "status": "infeasible",
            "epochs": None,
            "per_class": None,
            "robustness_n_upper_bound": None,
            "planned_hours": smallest["planned_hours"],
            "cap_hours": cap,
            "smallest_plan": {"epochs": smallest["epochs"], "per_class": smallest["per_class"]},
            "rule": SELECTION_RULE,
        }
    return {
        "inputs": {
            "hours_remaining": h,
            "reserve_hours": float(b["reserve_hours"]),
            "hard_cap_hours": float(b["hard_cap_hours"]),
            "safety_multiplier": safety,
            "cap_hours": cap,
        },
        "data": {
            "n_train": int(measurements["data"]["n_train"]),
            "n_val": int(measurements["data"]["n_val"]),
            "n_test": int(measurements["data"]["n_test"]),
            "updates_per_epoch": int(math.ceil(int(measurements["data"]["n_train"]) / float(config["training"]["batch_size_expected"]))),
        },
        "candidates": candidates,
        "decision": decision,
    }


# ---------------------------------------------------------------- lock creation
def check_lockable(estimate: Dict[str, Any], allow_contention: bool = False) -> None:
    """Raise unless a budget lock may be written for this estimate.

    ``InfeasibleError`` (exit 4): no candidate fits. ``ContentionError`` (exit 5): noisy or contended
    probe without ``--allow-contention``.
    """
    decision = estimate["decision"]
    if decision["status"] != "feasible":
        raise InfeasibleError(
            "no plan fits: smallest plan needs %.2f h, cap is %.2f h; nothing is locked or launched"
            % (decision["planned_hours"], decision["cap_hours"])
        )
    meas = estimate.get("measurements", {})
    contended = bool(estimate.get("contention", {}).get("contended"))
    noisy = bool(meas.get("noisy"))
    if (noisy or contended) and not allow_contention:
        raise ContentionError(
            "timings too noisy or CPU contended; wait for other CPU load to stop and rerun, "
            "or pass --allow-contention to accept a conservative lock"
        )


def build_lock(
    estimate: Dict[str, Any],
    estimate_sha256: str,
    config_sha256: str,
    locked_utc: str,
    allow_contention: bool = False,
) -> Dict[str, Any]:
    """The ``budget_lock.json`` content (D5.3). Raises as :func:`check_lockable`."""
    check_lockable(estimate, allow_contention)
    d = estimate["decision"]
    return {
        "schema": 1,
        "kind": "m2_deadline_budget_lock",
        "run_id": estimate["run_id"],
        "locked_utc": locked_utc,
        "epochs": int(d["epochs"]),
        "per_class_target": int(d["per_class"]),
        "planned_hours": float(d["planned_hours"]),
        "cap_hours": float(d["cap_hours"]),
        "hours_remaining_at_timing": float(estimate["inputs"]["hours_remaining"]),
        "estimate_sha256": estimate_sha256,
        "config_sha256": config_sha256,
        "selection_basis": "runtime only",
        "performance_inspected": False,
    }


# ---------------------------------------------------------------- launch and resume checks
def chosen_candidate(estimate: Dict[str, Any], epochs: int, per_class: int) -> Dict[str, Any]:
    for c in estimate["candidates"]:
        if int(c["epochs"]) == int(epochs) and int(c["per_class"]) == int(per_class):
            return c
    raise GuardError("estimate has no candidate for epochs=%s per_class=%s" % (epochs, per_class))


def launch_check(
    lock: Dict[str, Any],
    estimate: Dict[str, Any],
    hours_remaining: Any,
    estimate_sha256: Optional[str] = None,
    config_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Full-plan deadline check at launch (D6.3). Needs the CURRENT hours remaining.

    Raises ``ConfigError`` (2) for a missing/invalid H, ``GuardError`` (3) for an inconsistent lock,
    ``InfeasibleError`` (4) when ``locked.planned_hours > min(9, H - 3)``.
    """
    h = validate_hours(hours_remaining)
    if lock.get("performance_inspected") is not False:
        raise GuardError("budget lock does not state performance_inspected == false")
    if estimate_sha256 is not None and lock.get("estimate_sha256") != estimate_sha256:
        raise GuardError("budget lock estimate_sha256 differs from timing_estimate.json")
    if config_sha256 is not None and lock.get("config_sha256") != config_sha256:
        raise GuardError("budget lock config_sha256 differs from the current config")
    cand = chosen_candidate(estimate, lock["epochs"], lock["per_class_target"])
    if abs(float(cand["planned_hours"]) - float(lock["planned_hours"])) > 1e-9:
        raise GuardError("budget lock planned_hours differs from the estimate's chosen candidate")
    inputs = estimate["inputs"]
    cap = cap_hours(h, inputs["hard_cap_hours"], inputs["reserve_hours"])
    planned = float(lock["planned_hours"])
    if cap <= 0 or planned > cap:
        raise InfeasibleError(
            "locked plan needs %.2f h but min(%.1f, H - %.1f) = %.2f h for H = %.2f; "
            "rerun timing with the current hours (--force-relock is allowed only before any stage marker exists)"
            % (planned, inputs["hard_cap_hours"], inputs["reserve_hours"], cap, h)
        )
    return {"planned_hours": planned, "cap_hours": cap, "hours_remaining": h, "ok": True}


def remaining_seconds(
    estimate: Dict[str, Any],
    epochs: int,
    per_class: int,
    valid_stages: Sequence[str],
    last_completed_epoch: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Per-stage locked seconds still to run. A stage with a valid marker contributes 0.

    For a training stage without a valid marker, only the unfinished epochs count
    (``(E - last_completed_epoch) * seconds per epoch``). ``overhead`` always counts once because every
    run or resume process loads the arrays again.
    """
    comp = chosen_candidate(estimate, epochs, per_class)["components_s"]
    done = set(valid_stages)
    progress = last_completed_epoch or {}
    out: Dict[str, float] = {}
    for stage in STAGES:
        name = STAGE_COMPONENT[stage]
        if name is None or stage in done:
            out[stage] = 0.0
            continue
        seconds = float(comp[name])
        if stage in ("train_ordinary", "train_adversarial"):
            finished = min(max(int(progress.get(stage, 0)), 0), int(epochs))
            seconds = seconds * (int(epochs) - finished) / float(epochs)
        out[stage] = seconds
    out["overhead"] = float(comp["overhead"])
    return out


def resume_check(
    estimate: Dict[str, Any],
    lock: Dict[str, Any],
    valid_stages: Sequence[str],
    hours_remaining: Any,
    last_completed_epoch: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Remaining-plan deadline check at resume (D6.3). The 9 h ceiling is not applied.

    ``remaining_planned_hours = safety * sum(remaining stage seconds) / 3600`` must be ``<= H - reserve``.
    Raises ``InfeasibleError`` otherwise; the message deliberately does not suggest ``--force-relock``.
    """
    h = validate_hours(hours_remaining)
    inputs = estimate["inputs"]
    breakdown = remaining_seconds(
        estimate, lock["epochs"], lock["per_class_target"], valid_stages, last_completed_epoch
    )
    remaining_h = float(inputs["safety_multiplier"]) * sum(breakdown.values()) / 3600.0
    limit = h - float(inputs["reserve_hours"])
    result = {
        "remaining_planned_hours": remaining_h,
        "limit_hours": limit,
        "hours_remaining": h,
        "breakdown_s": breakdown,
        "ok": bool(limit > 0 and remaining_h <= limit),
    }
    if not result["ok"]:
        parts = ", ".join("%s=%.0fs" % (k, v) for k, v in breakdown.items() if v > 0)
        raise InfeasibleError(
            "remaining plan needs %.2f h but H - %.1f = %.2f h (stages left: %s). Decide by hand what to cut; "
            "the budget lock cannot be relocked after stage markers exist."
            % (remaining_h, inputs["reserve_hours"], limit, parts)
        )
    return result
