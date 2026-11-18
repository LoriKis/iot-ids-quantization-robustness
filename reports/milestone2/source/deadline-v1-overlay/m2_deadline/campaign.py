"""Campaign runner (DESIGN 6.3): launch, resume, status.

    python -m m2_deadline.campaign launch --hours-remaining H [--out P --v4-root P --config P --stop-after STAGE]
    python -m m2_deadline.campaign resume --hours-remaining H [same flags]
    python -m m2_deadline.campaign status [--out P --v4-root P --config P]

The epoch count comes only from ``timing/budget_lock.json``; there is no ``--epochs`` flag.
``--hours-remaining`` is required for launch and resume because time passes between timing and launch.

Exit codes: 0 ok; 1 unexpected error; 2 usage/config/data error (including a missing lock or hours);
3 guard failure (inconsistent lock, run identity, protocol, smoke path); 4 infeasible deadline check or
failed correctness gate; 5 contention; 130 interrupted (SIGINT/SIGTERM honoured at an epoch or stage boundary).

Import-time rule: standard library + numpy only. torch and research.* are reached through the deps object.
Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import budget, markers
from .common import (
    DeadlineError,
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_OK,
    STAGES,
    ConfigError,
    GuardError,
    check_required_keys,
    exit_code_for,
    load_config,
    marker_path,
    model_dir,
    out_path,
    read_json,
    setup_logging,
    sha256_file,
)
from .timing import DEFAULT_CONFIG, DEFAULT_OUT, check_v4_root

TRAIN_STAGE_MODEL = {"train_ordinary": "V1", "train_adversarial": "V3"}


# ---------------------------------------------------------------- stop flag
class StopFlag:
    """Set by SIGINT/SIGTERM. A second signal raises KeyboardInterrupt to force an exit."""

    def __init__(self) -> None:
        self.requested = False
        self.count = 0
        self.signal_name: Optional[str] = None

    def __call__(self) -> bool:
        return self.requested

    def handle(self, signum: int, frame: Any) -> None:
        self.count += 1
        self.requested = True
        self.signal_name = signal.Signals(signum).name
        if self.count >= 2:
            raise KeyboardInterrupt()


# ---------------------------------------------------------------- arguments
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=str(DEFAULT_OUT), help="run output folder (default: artifacts/m2-deadline-v1 in the overlay)")
    common.add_argument("--v4-root", default=None, help="verified V4 package folder (default: env V4_ROOT)")
    common.add_argument("--config", default=str(DEFAULT_CONFIG), help="overlay config JSON")

    run = argparse.ArgumentParser(add_help=False)
    run.add_argument("--hours-remaining", type=float, default=None, help="actual hours left until the deadline (0 < H <= 12); required")
    run.add_argument("--stop-after", choices=STAGES, default=None, help="stop cleanly after this stage")

    p = argparse.ArgumentParser(
        prog="python -m m2_deadline.campaign",
        description="Run the locked M2 deadline campaign on the VM. The epoch count comes from budget_lock.json only.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("launch", parents=[common, run], help="start a new run (refuses if one exists)")
    sub.add_parser("resume", parents=[common, run], help="continue at the first missing or invalid stage")
    sub.add_parser("status", parents=[common], help="print stage markers, last log line and elapsed times")
    return p


# ---------------------------------------------------------------- guarded loading
def _resolve_v4_root(arg: Optional[str]) -> Path:
    import os

    value = arg or os.environ.get("V4_ROOT")
    if not value:
        raise ConfigError("--v4-root (or env V4_ROOT) is required")
    return check_v4_root(value)


def load_run_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    """Config, lock and estimate with every consistency guard of DESIGN 6.3 (exit 2 missing, exit 3 inconsistent)."""
    config_path = Path(args.config)
    config = load_config(config_path)
    config_sha256 = sha256_file(config_path)
    v4_root = _resolve_v4_root(args.v4_root)
    out = Path(args.out)
    markers.assert_not_smoke(out)
    try:
        out.resolve().relative_to((v4_root / "output").resolve())
    except ValueError:
        pass
    else:
        raise GuardError("OUT may not live under the V4 output folder: %s" % out)

    lock_path = out_path(out, "budget_lock")
    estimate_path = out_path(out, "timing_estimate")
    if not lock_path.is_file():
        raise ConfigError("budget_lock.json not found under %s: run the timing command first" % lock_path.parent)
    if not estimate_path.is_file():
        raise ConfigError("timing_estimate.json not found under %s: run the timing command first" % estimate_path.parent)
    lock = read_json(lock_path)
    estimate = read_json(estimate_path)
    check_required_keys(lock, "budget_lock", "budget_lock.json")
    check_required_keys(estimate, "timing_estimate", "timing_estimate.json")
    if lock["schema"] != 1:
        raise ConfigError("budget_lock.json schema %r is not supported" % (lock["schema"],))
    if lock["epochs"] not in config["training"]["epoch_candidates"] or lock["per_class_target"] not in config["budget"]["per_class_candidates"]:
        raise ConfigError("budget lock epochs=%r per_class_target=%r are not allowed candidates" % (lock["epochs"], lock["per_class_target"]))
    lock_sha256 = sha256_file(lock_path)
    estimate_sha256 = sha256_file(estimate_path)
    if lock["estimate_sha256"] != estimate_sha256:
        raise GuardError("budget lock estimate_sha256 differs from timing_estimate.json (estimate edited after locking)")
    if lock["config_sha256"] != config_sha256 or estimate["config_sha256"] != config_sha256:
        raise GuardError("config changed since the budget was locked (config sha256 differs)")
    if lock["performance_inspected"] is not False:
        raise GuardError("budget lock does not state performance_inspected == false")
    if lock["run_id"] != config["run_id"]:
        raise GuardError("budget lock run_id differs from the config run_id")
    if estimate["v4_manifest_sha256"] != sha256_file(v4_root / "MANIFEST.sha256"):
        raise GuardError("V4 MANIFEST.sha256 differs from the one the budget was measured against")
    return {
        "config": config, "config_path": config_path, "config_sha256": config_sha256, "v4_root": v4_root, "out": out,
        "lock": lock, "lock_sha256": lock_sha256, "estimate": estimate, "estimate_sha256": estimate_sha256,
    }


def training_progress(out: Path) -> Dict[str, int]:
    """Last completed epoch per training stage, from the epoch checkpoint index (0 if none)."""
    progress: Dict[str, int] = {}
    for stage, model_id in TRAIN_STAGE_MODEL.items():
        latest = model_dir(out, model_id) / "ckpt" / "latest.json"
        try:
            progress[stage] = int(read_json(latest)["epoch"]) if latest.is_file() else 0
        except (ValueError, KeyError, OSError):
            progress[stage] = 0
    return progress


# ---------------------------------------------------------------- engine
def run_stages(
    ctx: Any,
    stage_runner: Callable[[Any, str], Any],
    *,
    stop_after: Optional[str] = None,
    log: Optional[logging.Logger] = None,
) -> int:
    """Run from the first missing or invalid stage to the end (or to ``stop_after``). Returns an exit code."""
    log = log or logging.getLogger("m2_deadline.campaign")
    first = markers.first_invalid_stage(ctx.out, ctx.marker_ctx())
    if first is None:
        log.info("every stage marker is valid; nothing to run")
        return EXIT_OK
    start = STAGES.index(first)
    if stop_after is not None and STAGES.index(stop_after) < start:
        log.info("stage %s is already valid; nothing to run before the stop point", stop_after)
        return EXIT_OK
    for stage in STAGES[start:]:
        if ctx.stop_flag():
            log.warning("stop requested; not starting stage %s (resume later)", stage)
            return EXIT_INTERRUPTED
        stage_runner(ctx, stage)
        if stop_after == stage:
            log.info("stopping after stage %s as requested", stage)
            return EXIT_OK
    return EXIT_OK


# ---------------------------------------------------------------- commands
def cmd_start(args: argparse.Namespace, resume: bool, deps: Any, stage_runner: Optional[Callable[[Any, str], Any]],
              install_signals: bool) -> int:
    hours = budget.validate_hours(args.hours_remaining)
    state = load_run_inputs(args)
    out: Path = state["out"]
    lock, estimate = state["lock"], state["estimate"]
    identity_path = out_path(out, "run_identity")
    has_run = identity_path.is_file() or markers.any_marker_exists(out)

    if identity_path.is_file():
        markers.verify_run_identity(
            out,
            config_sha256=state["config_sha256"],
            lock_sha256=state["lock_sha256"],
            epochs=int(lock["epochs"]),
            v4_root=state["v4_root"],
            v4_manifest_sha256=estimate["v4_manifest_sha256"],
        )
    if resume:
        if not has_run:
            raise ConfigError("nothing to resume under %s; use launch for a new run" % out)
    else:
        if has_run:
            raise ConfigError("a run already exists under %s; use resume" % out)
        budget.launch_check(lock, estimate, hours, state["estimate_sha256"], state["config_sha256"])

    from . import stages  # lazy: keeps --help and the guard paths light

    runner = stage_runner or stages.run_stage
    stop = StopFlag()
    log, log_path = setup_logging(out, "campaign")
    ctx = stages.RunContext(
        out=out, v4_root=state["v4_root"], config=state["config"], config_sha256=state["config_sha256"],
        lock=lock, lock_sha256=state["lock_sha256"], estimate=estimate, estimate_sha256=state["estimate_sha256"],
        deps=deps, log=log, stop_flag=stop, config_path=state["config_path"],
    )
    ctx.refresh_protocol_sha()
    if resume:
        valid = markers.valid_stages(out, ctx.marker_ctx())
        check = budget.resume_check(estimate, lock, valid, hours, training_progress(out))
        log.info("resume check: remaining %.2f h <= limit %.2f h", check["remaining_planned_hours"], check["limit_hours"])
    log.info("%s: epochs=%d per_class=%d hours_remaining=%.2f out=%s log=%s",
             "resume" if resume else "launch", ctx.epochs, ctx.per_class, hours, out, log_path)

    previous: Dict[int, Any] = {}
    if install_signals and threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, stop.handle)
    try:
        code = run_stages(ctx, runner, stop_after=args.stop_after, log=log)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if code == EXIT_INTERRUPTED:
        log.warning("interrupted (%s); resume with the resume command", stop.signal_name or "stop flag")
    return code


def cmd_status(args: argparse.Namespace) -> int:
    out = Path(args.out)
    config = load_config(Path(args.config))
    config_sha256 = sha256_file(Path(args.config))
    lock_path = out_path(out, "budget_lock")
    print("run folder: %s" % out)
    if not lock_path.is_file():
        print("no budget_lock.json yet: run the timing command first")
        return EXIT_OK
    lock = read_json(lock_path)
    protocol_path = out_path(out, "protocol")
    ctx = markers.context(
        config_sha256=config_sha256, lock_sha256=sha256_file(lock_path),
        protocol_sha256=sha256_file(protocol_path) if protocol_path.is_file() else None,
        epochs=int(lock["epochs"]),
    )
    print("locked plan: %d epochs, %d test rows per class, planned %.2f h (cap %.2f h)"
          % (lock["epochs"], lock["per_class_target"], lock["planned_hours"], lock["cap_hours"]))
    print("%-18s %-8s %s" % ("stage", "valid", "detail"))
    for stage in STAGES:
        ok, reason = markers.verify(out, stage, ctx)
        detail = reason
        if marker_path(out, stage).is_file():
            try:
                m = read_json(marker_path(out, stage))
                detail = "%s; %.0f s; finished %s" % (reason, float(m.get("seconds", 0.0)), m.get("finished_utc"))
            except (ValueError, OSError):
                pass
        print("%-18s %-8s %s" % (stage, "yes" if ok else "no", detail))
    progress = training_progress(out)
    print("training checkpoints (last completed epoch): V1=%d V3=%d of %d" % (progress["train_ordinary"], progress["train_adversarial"], lock["epochs"]))
    logs = sorted((out / "logs").glob("campaign-*.log")) if (out / "logs").is_dir() else []
    if logs:
        lines = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
        print("last log line (%s): %s" % (logs[-1].name, lines[-1] if lines else ""))
    return EXIT_OK


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    deps: Any = None,
    stage_runner: Optional[Callable[[Any, str], Any]] = None,
    install_signals: bool = True,
) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "status":
            return cmd_status(args)
        return cmd_start(args, args.command == "resume", deps, stage_runner, install_signals)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except DeadlineError as exc:
        print("%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return exit_code_for(exc)
    except Exception as exc:  # unexpected: log the traceback, exit 1
        print("unexpected error: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
