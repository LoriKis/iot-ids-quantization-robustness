"""M2B campaign CLI and scheduler (DESIGN 4.2, 4.6, 4.8): plan, run, status, estimate, freeze-eval, efficiency, report,
smoke, preflight, fingerprint-sets.

`run` executes the unit graph of research.m2b_plan with durable restart state (research.m2b_state): finished units are
verified and reused (never overwritten), crash debris is archived by rename, a fingerprint mismatch exits 2 before
anything is written. The validation phase loads train and validation rows only. The test phase is reachable only through
`freeze-eval` plus `run --phase test --freeze-name N --confirm-freeze SHA`; the freeze checks 1 to 3 (hex, file hash,
recorded model/calibration/conversion/fingerprint hashes) finish before the dataset is hashed or any array is loaded.

Exit codes: 0 ok; 2 configuration, data or fingerprint error (nothing written); 3 partition or freeze guard; 4 a unit
failed or a worker died (rerun resumes); 5 invalid attack units recorded; 130 interrupted gracefully.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import multiprocessing
import os
import re
import signal
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from . import m2b_plan as mp_plan
from .m2_split import ConfigError, DataIntegrityError, PartitionGuardError
from .m2b_io import LockHeldError, durable_write_bytes, durable_write_json, sha256_file, unit_lock, utc_now
from .m2b_plan import PlanError, Unit, blocked_units, build_plan, external_deps
from .m2b_state import (CampaignState, ConfigIntegrityError, FingerprintMismatchError, FreezeGuardError, TAG_RE,
                        UnitCorruptionError, archive_and_restart_affected, archive_incomplete_unit, compute_fingerprints,
                        make_unit_record, require_matching, verify_eval_freeze, verify_unit_completion,
                        write_fingerprint_sets, runbook_table)
from .m2b_state import ClosureResolutionError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/m2b/campaign_v2.json"
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
NAME_RE = TAG_RE
VALIDATION_STAGES = ("train", "calib", "paircheck", "ptq", "clean", "step0", "attack", "qat")
TEST_STAGES = ("clean", "attack", "qat")
FREEZE_CHECK_COMPONENTS = ("config", "split", "source_train", "source_quant", "source_attack", "env")
ATTACK_COST = {"fgsm": 1, "pgd20": 60, "pgd50x5": 250}
RESUMABLE_KINDS = ("train", "fp32_ft", "qat")
log = logging.getLogger("m2b")

EXIT_OK, EXIT_CONFIG, EXIT_GUARD, EXIT_FAILED, EXIT_INVALID, EXIT_INTERRUPTED = 0, 2, 3, 4, 5, 130


class ExitError(Exception):
    """A command-line level refusal with the exit code the design assigns to it."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class StopFlag:
    """Graceful stop: an in-memory event plus a marker file so spawn workers see the signal too."""

    def __init__(self, path):
        self.path, self.event = Path(path), threading.Event()

    def __call__(self) -> bool:
        return self.event.is_set() or self.path.exists()

    def set(self) -> None:
        self.event.set()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(utc_now(), encoding="utf-8")
        except OSError:
            pass


class Ctx:
    """Everything a run needs that is not an argument: paths, config, state, current fingerprints."""

    def __init__(self, root, run_dir, campaign_path, cfg, state, fingerprints, phase):
        self.root, self.run_dir, self.campaign_path = Path(root), Path(run_dir), Path(campaign_path)
        self.cfg, self.state, self.fingerprints, self.phase = cfg, state, fingerprints, phase


# ------------------------------------------------------------------------------------------ configuration
def load_campaign(path, root=ROOT) -> dict:
    """campaign_v2.json with the structural checks of DESIGN section 6 (sibling hashes are verified by the fingerprints)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"campaign configuration not found: {path}")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    required = {"schema_version", "claim", "supersedes", "seeds", "epochs", "batch_size", "epsilon_grid", "threads",
                "checkpointing", "attack_row_policy", "matrix", "e4", "qat", "step0", "outputs", "siblings"}
    if set(cfg) != required:
        raise ConfigError(f"campaign config keys differ: missing {sorted(required - set(cfg))}, unknown {sorted(set(cfg) - required)}")
    if cfg["schema_version"] != 2:
        raise ConfigError("campaign config schema_version must be 2")
    root = Path(root)
    training = json.loads((root / cfg["siblings"]["training"]["path"]).read_text(encoding="utf-8"))
    attacks = json.loads((root / cfg["siblings"]["attacks"]["path"]).read_text(encoding="utf-8"))
    if cfg["seeds"] != training["seeds"] or cfg["epochs"] != training["training"]["epochs"] or cfg["epochs"] != 25:
        raise ConfigError("campaign seeds/epochs differ from configs/m2/training.json (epochs must be 25)")
    if cfg["batch_size"] != training["training"]["batch_size"] or cfg["batch_size"] != 512:
        raise ConfigError("campaign batch_size must be 512 and equal training.json")
    if cfg["epsilon_grid"] != attacks["epsilon_grid"]:
        raise ConfigError("campaign epsilon_grid differs from configs/m2/attacks.json")
    q = cfg["qat"]
    if q["k"] != 3 or q["lr"] != 1e-4 or q["enabled"] is not False:
        raise ConfigError("qat block must have k=3, lr=1e-4 and enabled=false (QAT is opt-in on the command line)")
    return cfg


def resolve_paths(args, root):
    if not RUN_NAME_RE.match(args.run_name):
        raise ExitError(EXIT_CONFIG, f"--run-name must match {RUN_NAME_RE.pattern}")
    out_root = Path(args.out_root)
    out_root = (out_root if out_root.is_absolute() else Path(root) / out_root)
    if ".." in Path(args.out_root).parts:
        raise ExitError(EXIT_CONFIG, "--out-root must not contain '..'")
    resolved = out_root.resolve()
    if not getattr(args, "allow_outside_root", False):
        try:
            resolved.relative_to(Path(root).resolve())
        except ValueError:
            raise ExitError(EXIT_CONFIG, f"--out-root {args.out_root} is outside the package root (use --allow-outside-root)") from None
    if out_root.is_symlink():
        raise ExitError(EXIT_CONFIG, "--out-root must not be a symlink")
    campaign_path = Path(args.config)
    campaign_path = campaign_path if campaign_path.is_absolute() else Path(root) / campaign_path
    return resolved / args.run_name, campaign_path


def parse_seeds(seeds, cfg):
    if seeds is None:
        return list(cfg["seeds"])
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(cfg["seeds"]):
        raise ExitError(EXIT_CONFIG, f"--seeds must be unique and a subset of {cfg['seeds']}")
    return list(seeds)


def parse_stages(stages, phase):
    if stages is None:
        return None
    flat = [s for item in stages for s in item.split(",") if s]
    allowed = VALIDATION_STAGES if phase == "validation" else TEST_STAGES
    bad = [s for s in flat if s not in allowed]
    if bad:
        raise ExitError(EXIT_CONFIG, f"--stages {bad} not allowed in the {phase} phase; choose from {list(allowed)}")
    return tuple(dict.fromkeys(flat))


def read_decision(run_dir):
    path = Path(run_dir) / "step0" / "budget_decision.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def require_qat_opt_in(args):
    if getattr(args, "opt_in", None) != "qat" or os.environ.get("M2_ENABLE_QAT") != "yes":
        raise ExitError(EXIT_GUARD, "the QAT stage is opt-in: pass --opt-in qat and set M2_ENABLE_QAT=yes")


# ------------------------------------------------------------------------------------------ freeze (DESIGN 4.8)
def model_files(run_dir, seed, arm):
    """Files whose hashes a freeze binds for one model of one seed."""
    folder = Path(run_dir) / "seeds" / f"s{seed}"
    models = folder / "models" / arm
    int8 = arm in ("V2", "V4", "FP32-FT-PTQ", "QAT-FT")
    files = {"checkpoint_or_state": models / ("state.pt" if int8 else "checkpoint.pt")}
    if int8:
        files.update(calibration_ids=folder / "calibration" / "calibration_ids.json",
                     calibration_x=folder / "calibration" / "calibration_x.pt", conversion=models / "conversion.json")
    return files


def freeze_model_entry(run_dir, seed, arm):
    files = {k: sha256_file(p) for k, p in model_files(run_dir, seed, arm).items()}
    int8 = "calibration_x" in files
    return {"checkpoint_or_state_sha256": files["checkpoint_or_state"], "parent": mp_plan.PARENT.get(arm),
            "calibration_ids_sha256": files.get("calibration_ids"), "calibration_x_sha256": files.get("calibration_x"),
            "conversion_json_sha256": files.get("conversion"),
            "surrogate_input_sha256": files.get("calibration_x") if int8 else None}


def compare_freeze(data, run_dir, root, campaign_path) -> None:
    """Check 3 of DESIGN 4.8: recorded hashes versus the files in output/ and the recorded config/split/source components.

    Reads only output/ files and small JSON configs; never the dataset.
    """
    run_dir = Path(run_dir)
    for key in ("schema", "freeze_name", "seeds", "models", "fingerprints", "budget_decision_sha256", "attack_budget",
                "campaign_sha256", "sibling_sha256", "qat_included", "test_partition_id_sha256"):
        if key not in data:
            raise FreezeGuardError(f"freeze file lacks {key!r}")
    problems = []
    for seed, arms in data["models"].items():
        for arm, entry in arms.items():
            try:
                current = freeze_model_entry(run_dir, int(seed), arm)
            except OSError as error:
                problems.append(f"s{seed} {arm}: {error}")
                continue
            problems += [f"s{seed} {arm}: {k} differs from the freeze" for k, v in current.items()
                         if k != "parent" and v != entry.get(k)]
    decision = run_dir / "step0" / "budget_decision.json"
    if not decision.is_file() or sha256_file(decision) != data["budget_decision_sha256"]:
        problems.append("step0/budget_decision.json differs from the freeze")
    if sha256_file(campaign_path) != data["campaign_sha256"]:
        problems.append("campaign_v2.json differs from the freeze")
    campaign = json.loads(Path(campaign_path).read_text(encoding="utf-8"))
    siblings = {n: sha256_file(Path(root) / e["path"]) for n, e in campaign["siblings"].items()}
    if siblings != data["sibling_sha256"]:
        problems.append("a sibling configuration differs from the freeze")
    if not problems:
        current = compute_fingerprints(root, campaign_path, with_data=False)["components"]
        frozen = data["fingerprints"]["components"]
        problems += [f"fingerprint component {c} differs from the freeze" for c in FREEZE_CHECK_COMPONENTS
                     if current.get(c) != frozen.get(c)]
    if problems:
        raise FreezeGuardError("the evaluation freeze does not match the current files: " + "; ".join(problems[:12]))


# ------------------------------------------------------------------------------------------ environment / executors
def get_executors():
    return importlib.import_module("research.m2b_exec").EXECUTORS


def make_env(spec, state, stop_flag, arrays_loader=None):
    """Build the Env of a run from a picklable spec (also used by spawn workers)."""
    from . import m2_attacks as ma
    from . import m2b_exec as ex
    cfg = spec["cfg"]
    freeze, attack_cfg = None, None
    if spec["phase"] == "test":
        # validate the UNMODIFIED attacks.json first; only then copy and switch the partition (DESIGN 4.8)
        validated = ma.load_config(Path(spec["root"]) / cfg["siblings"]["attacks"]["path"])
        attack_cfg = ex.deep_copy_test_attack_cfg(validated)
        freeze = verify_eval_freeze(spec["freeze_path"], spec["freeze_sha"])
    return ex.Env(spec["root"], spec["run_dir"], cfg, phase=spec["phase"], state=state, freeze=freeze, stop_flag=stop_flag,
                  fingerprints=spec["fingerprints"], epochs=spec.get("epochs"), arrays_loader=arrays_loader,
                  attack_cfg=attack_cfg, extra=spec.get("extra"))


_WORKER = {}


def _worker_init():
    flag = {"stop": None}

    def handler(signum, frame):
        if _WORKER.get("stop") is not None:
            _WORKER["stop"].event.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def _worker_entry(spec, unit):
    if "env" not in _WORKER:
        state = CampaignState(Path(spec["run_dir"]) / "state")
        stop = StopFlag(state.dir / "STOP")
        _WORKER.update(stop=stop, state=state, env=make_env(spec, state, stop))
    ctx = Ctx(spec["root"], spec["run_dir"], spec["campaign_path"], spec["cfg"], _WORKER["state"], spec["fingerprints"],
              spec["phase"])
    return execute_unit(ctx, unit, get_executors()[unit.kind], _WORKER["env"], _WORKER["stop"])


# ------------------------------------------------------------------------------------------ one unit
def list_outputs(run_dir, rel_dir):
    """All files below a unit directory (relative posix paths from the run root), temp files excluded."""
    folder = Path(run_dir) / rel_dir
    return sorted(p.relative_to(run_dir).as_posix() for p in folder.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))


def prepare_unit_dir(ctx, unit, rel_dir):
    """Crash debris: archive a unit directory that has no completed marker. A resumable unit keeps its ckpt/."""
    if rel_dir is None:
        return
    record = ctx.state.read_unit(unit.id)
    folder = ctx.run_dir / rel_dir
    if (record is not None and record["status"] == "completed") or not folder.exists():
        return
    stamp = f"incomplete-{utc_now()}"
    if unit.kind in RESUMABLE_KINDS and (folder / "ckpt" / "latest.json").is_file():
        from .m2b_io import archive_move
        for entry in sorted(folder.iterdir()):
            if entry.name != "ckpt":
                archive_move(entry, ctx.run_dir / "archive", stamp)
    else:
        archive_incomplete_unit(ctx.state, ctx.run_dir, unit.id, rel_dir, now=stamp.split("incomplete-")[1])


def _record(ctx, unit, status, rel_dir, started, t0, threads, **kw):
    return make_unit_record(unit.id, unit.kind, unit.phase, unit.seed, status, fingerprints=ctx.fingerprints,
                            started_utc=started, finished_utc=utc_now(), seconds=time.perf_counter() - t0, threads=threads,
                            unit_dir=rel_dir, **kw)


def execute_unit(ctx, unit, executor, env, stop) -> dict:
    """Lock, run, hash outputs, write the completion record last. Never raises for unit-level errors."""
    from . import m2b_exec as ex
    state = ctx.state
    try:
        lock = unit_lock(state.locks_dir / f"{unit.id}.lock")
        lock.__enter__()
    except LockHeldError as error:
        log.warning("unit %s skipped: %s", unit.id, error)
        return {"id": unit.id, "status": "locked"}
    try:
        rel_dir = ex.unit_rel_dir(unit)
        prepare_unit_dir(ctx, unit, rel_dir)
        started, t0 = utc_now(), time.perf_counter()
        threads = ex.threads_for(unit.kind, ctx.cfg)
        try:
            import torch
            torch.set_num_threads(threads)
        except ImportError:  # pragma: no cover
            pass
        os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(threads)
        state.write_unit(make_unit_record(unit.id, unit.kind, unit.phase, unit.seed, "running", fingerprints=ctx.fingerprints,
                                          started_utc=started, threads=threads, unit_dir=rel_dir))
        log.info("unit %s: start (%s, %d thread(s))", unit.id, unit.kind, threads)
        died = False
        try:
            result = executor(unit, env) or {}
            status = result.get("status", "completed")
            outputs = result.get("outputs")
            if outputs is None:
                outputs = list_outputs(ctx.run_dir, rel_dir) if rel_dir else []
            from .m2b_state import describe_outputs
            hashes, sizes = describe_outputs(ctx.run_dir, outputs)
            record = _record(ctx, unit, status, rel_dir, started, t0, threads, outputs=hashes, output_sizes=sizes,
                             extra=result.get("extra"))
        except ex.UnitBlocked as blocked:
            record = _record(ctx, unit, "blocked", rel_dir, started, t0, threads,
                             error={"type": "UnitBlocked", "message": str(blocked), **blocked.error},
                             extra={"own_block": True, "documented": blocked.documented})
            log.warning("unit %s blocked: %s", unit.id, blocked)
        except Exception as error:  # noqa: BLE001 - classified below
            from .m2_training import TrainingInterrupted
            info = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            if isinstance(error, TrainingInterrupted):
                record = _record(ctx, unit, "interrupted", rel_dir, started, t0, threads, error=info, extra={"resumable": True})
            elif isinstance(error, (PartitionGuardError, FreezeGuardError)):
                state.write_unit(_record(ctx, unit, "failed", rel_dir, started, t0, threads, error=info,
                                         extra={"retryable": False}))
                raise
            else:
                retry = isinstance(error, (OSError, MemoryError))
                record = _record(ctx, unit, "failed", rel_dir, started, t0, threads, error=info, extra={"retryable": retry})
                log.error("unit %s failed: %s: %s", unit.id, type(error).__name__, error)
        except (KeyboardInterrupt, SystemExit) as error:
            died = isinstance(error, SystemExit)
            if isinstance(error, KeyboardInterrupt):
                stop.event.set()
            record = _record(ctx, unit, "interrupted", rel_dir, started, t0, threads,
                             error={"type": type(error).__name__, "message": str(error), "traceback": None},
                             extra={"worker_died": died})
        state.write_unit(record)
        log.info("unit %s: %s in %.1f s", unit.id, record["status"], record["seconds"])
        return {"id": unit.id, "status": record["status"], "died": died, "extra": record["extra"]}
    finally:
        lock.__exit__(None, None, None)


# ------------------------------------------------------------------------------------------ scheduler
def dispatch_key(unit, index):
    """Non-attack units first in plan order; attack units by estimated cost, long pgd budgets first."""
    if unit.kind != "attack":
        return (0, 0, index)
    return (1, -ATTACK_COST.get(unit.params.get("attack"), 1), index)


class _InlinePool:
    """workers=1: run in this process (also used by tests with stub executors) behind the Future interface."""

    def __init__(self, fn):
        self.fn = fn

    def submit(self, *args):
        future = Future()
        try:
            future.set_result(self.fn(*args))
        except BaseException as error:  # noqa: BLE001 - e.g. PartitionGuardError propagates via result()
            future.set_exception(error)
        return future

    def shutdown(self, *a, **k):
        pass


def _classify_existing(ctx, units, retry_failed):
    """Reuse verified completed units; return (status map, units still to run, reused count)."""
    status, todo, reused = {}, [], []
    for u in units:
        rec = ctx.state.read_unit(u.id)
        if rec is not None and rec["status"] == "completed":
            verify_unit_completion(rec, ctx.run_dir)
            status[u.id] = "completed"
            reused.append(u)
        elif rec is not None and rec["status"] == "invalid":
            verify_unit_completion({**rec, "status": "completed"}, ctx.run_dir)
            status[u.id] = "invalid"
        elif rec is not None and rec["status"] == "failed" and not (retry_failed or rec["extra"].get("retryable")):
            status[u.id] = "failed"
        elif rec is not None and rec["status"] == "blocked" and rec["extra"].get("own_block") and not retry_failed:
            status[u.id] = "blocked"
        else:
            todo.append(u)
    return status, todo, reused


def run_units(ctx, units, *, executors, env, stop, workers=1, retry_failed=False, worker_spec=None) -> dict:
    """Execute `units` in dependency order. Returns {status counts, exit_code, reused, ...}; writes unit records."""
    state = ctx.state
    status, todo, reused = _classify_existing(ctx, units, retry_failed)
    for dep in external_deps(units):
        rec = state.read_unit(dep)
        if rec is None or rec["status"] not in ("completed", "invalid"):
            raise ExitError(EXIT_CONFIG, f"dependency {dep} is not completed (it is outside the selected stages or seeds); "
                                         "run it first, dependencies are never added silently")
        if rec["status"] == "completed":
            verify_unit_completion(rec, ctx.run_dir)
        status[dep] = rec["status"]
    index = {u.id: i for i, u in enumerate(units)}
    pending = {u.id: u for u in todo}
    documented, own_blocked = set(), {u.id for u in units if status.get(u.id) == "blocked"}
    for uid in own_blocked:
        if (state.read_unit(uid)["extra"] or {}).get("documented"):
            documented.add(uid)
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_worker_init)
        submit = lambda u: pool.submit(_worker_entry, worker_spec, u)  # noqa: E731
    else:
        pool = _InlinePool(lambda u: execute_unit(ctx, u, executors[u.kind], env, stop))
        submit = pool.submit
    in_flight, interrupted, died, fatal = {}, False, False, None
    try:
        while True:
            for uid, reason in blocked_units(units, status).items():
                if uid in pending:
                    unit = pending.pop(uid)
                    inherited = any(d in documented for d in unit.deps)
                    from .m2b_exec import unit_rel_dir
                    state.write_unit(make_unit_record(
                        unit.id, unit.kind, unit.phase, unit.seed, "blocked", fingerprints=ctx.fingerprints,
                        finished_utc=utc_now(), unit_dir=unit_rel_dir(unit),
                        error={"type": "DependencyBlocked", "message": reason},
                        extra={"own_block": False, "documented": inherited}))
                    status[uid] = "blocked"
                    if inherited:
                        documented.add(uid)
            if not (stop() or interrupted or died or fatal):
                ready = sorted((u for u in pending.values() if all(status.get(d) in ("completed", "invalid") for d in u.deps)),
                               key=lambda u: dispatch_key(u, index[u.id]))
                for unit in ready:
                    if len(in_flight) >= workers:
                        break
                    pending.pop(unit.id)
                    in_flight[submit(unit)] = unit
            if not in_flight:
                break
            done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                unit = in_flight.pop(future)
                try:
                    result = future.result()
                except BrokenProcessPool:
                    died = True
                    log.error("a worker process died while running %s", unit.id)
                    _mark_interrupted(ctx, unit, "worker process died")
                    status[unit.id] = "interrupted"
                    continue
                except BaseException as error:  # noqa: BLE001 - guard errors are re-raised after in-flight units finish
                    fatal = fatal or error
                    status[unit.id] = "failed"
                    continue
                status[unit.id] = result["status"]
                if result["status"] == "interrupted":
                    interrupted = True
                    died = died or bool(result.get("died"))
                if result["status"] == "blocked":
                    rec = state.read_unit(unit.id)
                    if (rec["extra"] or {}).get("documented"):
                        documented.add(unit.id)
    finally:
        pool.shutdown(wait=True, cancel_futures=True) if workers > 1 else pool.shutdown()
    if fatal is not None:
        raise fatal
    counts = {}
    for u in units:
        counts[status.get(u.id, "not_run")] = counts.get(status.get(u.id, "not_run"), 0) + 1
    undocumented_blocked = [u.id for u in units if status.get(u.id) == "blocked" and u.id not in documented]
    if died:
        code = EXIT_FAILED
    elif interrupted or stop():
        code = EXIT_INTERRUPTED
    elif counts.get("failed") or undocumented_blocked or counts.get("locked"):
        code = EXIT_FAILED
    elif counts.get("invalid"):
        code = EXIT_INVALID
    else:
        code = EXIT_OK
    reused_by_seed = {}
    for u in reused:
        reused_by_seed[u.seed] = reused_by_seed.get(u.seed, 0) + 1
    return {"counts": counts, "exit_code": code, "reused": len(reused), "reused_by_seed": reused_by_seed, "status": status, "documented_blocked": sorted(documented)}


def _mark_interrupted(ctx, unit, message):
    rec = ctx.state.read_unit(unit.id)
    if rec is not None and rec["status"] == "running":
        rec = dict(rec, status="interrupted", finished_utc=utc_now(), error={"type": "WorkerDied", "message": message, "traceback": None})
        ctx.state.write_unit(rec)


# ------------------------------------------------------------------------------------------ run command
def setup_logging(run_dir, stage):
    path = Path(run_dir) / "logs" / f"{stage}-{utc_now()}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(process)d %(message)s")
    root = logging.getLogger("m2b")
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stderr)):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    return path


def install_signal_handlers(stop):
    previous = {}

    def handler(signum, frame):
        log.warning("signal %s: finishing the current epoch/unit, then stopping", signum)
        stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):  # not the main thread
            pass
    return previous


def restore_signal_handlers(previous):
    for sig, old in previous.items():
        try:
            signal.signal(sig, old)
        except (ValueError, OSError):
            pass


def check_fingerprints(state, current, archive_tag, phase, run_dir):
    """Compare with the recorded fingerprints. Mismatch: FingerprintMismatchError (exit 2) unless archiving (validation only)."""
    if not state.exists():
        return
    recorded = state.read_fingerprints()
    try:
        require_matching(recorded, current, log)
    except FingerprintMismatchError:
        if not archive_tag or phase != "validation":
            raise
        moved = archive_and_restart_affected(state, run_dir, archive_tag, current)
        print(f"archived {len(moved)} item(s) of affected units under {Path(run_dir) / 'archive'}; they will be re-run")


def cmd_run(args, root, executors=None) -> int:
    phase = args.phase
    run_dir, campaign_path = resolve_paths(args, root)
    cfg = load_campaign(campaign_path, root)
    seeds = parse_seeds(args.seeds, cfg)
    stages = parse_stages(args.stages, phase)
    if args.workers < 1 or args.workers > 64:
        raise ExitError(EXIT_CONFIG, "--workers must be 1..64")
    if args.archive_tag is not None and not TAG_RE.match(args.archive_tag):
        raise ExitError(EXIT_CONFIG, f"--archive-and-restart-affected TAG must match {TAG_RE.pattern}")
    trains = phase == "validation" and (stages is None or bool(set(stages) & {"train", "qat"}))
    threads = cfg["threads"]["training"] if trains else cfg["threads"]["attack_and_evaluation"]
    if args.workers * threads > (os.cpu_count() or 1) and not args.oversubscribe:
        raise ExitError(EXIT_CONFIG, f"--workers {args.workers} x {threads} threads exceeds {os.cpu_count()} cores "
                                     "(use --oversubscribe to override)")
    freeze = None
    if phase == "test":
        if args.archive_tag:
            raise ExitError(EXIT_CONFIG, "the test phase has no archive override; re-freeze instead")
        if not NAME_RE.match(args.freeze_name):
            raise ExitError(EXIT_CONFIG, f"--freeze-name must match {NAME_RE.pattern}")
        freeze_path = run_dir / "freezes" / f"{args.freeze_name}.json"
        # checks 1 to 3 of DESIGN 4.8; nothing below touches the dataset before they pass
        freeze = verify_eval_freeze(freeze_path, args.confirm_freeze,
                                    compare=lambda data: compare_freeze(data, run_dir, root, campaign_path))
        if not set(seeds) <= {int(s) for s in freeze.data["seeds"]}:
            raise FreezeGuardError(f"--seeds {seeds} are not all frozen: {freeze.data['seeds']}")
        if stages is not None and "qat" in stages and not freeze.data["qat_included"]:
            raise ExitError(EXIT_GUARD, "the freeze was created without --include-qat; QAT arms are not evaluated in the test phase")
        stages = stages or (("clean", "attack", "qat") if freeze.data["qat_included"] else ("clean", "attack"))
    elif stages is not None and "qat" in stages:
        require_qat_opt_in(args)
    current = compute_fingerprints(root, campaign_path)  # includes the data fingerprint (check 4 for the test phase)
    if freeze is not None and current["components"]["data"] != freeze.data["fingerprints"]["components"]["data"]:
        raise FingerprintMismatchError("data fingerprint differs from the freeze (dataset or PAVE model source changed)")
    state = CampaignState(run_dir / "state")
    check_fingerprints(state, current, args.archive_tag, phase, run_dir)
    state.init(args.run_name, "real", seeds, opt_ins=["qat"] if stages and "qat" in stages else (), fingerprints=current)
    try:
        runner_lock = unit_lock(state.dir / "runner.lock")
        runner_lock.__enter__()
    except LockHeldError as error:
        raise ExitError(EXIT_CONFIG, f"another run holds the output root: {error}") from error
    stop = StopFlag(state.dir / "STOP")
    previous = install_signal_handlers(stop)
    try:
        if stop.path.exists():
            stop.path.unlink()
        setup_logging(run_dir, f"run-{phase}")
        spec = {"root": str(root), "run_dir": str(run_dir), "campaign_path": str(campaign_path), "cfg": cfg, "phase": phase,
                "fingerprints": current, "freeze_path": str(freeze.path) if freeze else None,
                "freeze_sha": freeze.sha256 if freeze else None, "extra": {}}
        ctx = Ctx(root, run_dir, campaign_path, cfg, state, current, phase)
        executors = executors if executors is not None else get_executors()
        env = make_env(spec, state, stop) if args.workers == 1 else None
        return _run_passes(ctx, args, seeds, stages, freeze, executors, env, stop, spec)
    finally:
        restore_signal_handlers(previous)
        runner_lock.__exit__(None, None, None)


def select_units(cfg, seeds, phase, stages, pgd_budget):
    """build_plan plus the one documented exception: the step0 stage pulls in the first seed's diag units (prerequisites of E4)."""
    if phase == "validation" and stages and "step0" in stages and "attack" not in stages:
        first = cfg["seeds"][0]
        full = build_plan(cfg, seeds, phase, tuple(stages) + ("attack",), pgd_budget=pgd_budget)
        return [u for u in full if u.stage in stages or (u.kind == "diag" and u.seed == first)]
    return build_plan(cfg, seeds, phase, stages, pgd_budget=pgd_budget)


def _run_passes(ctx, args, seeds, stages, freeze, executors, env, stop, spec) -> int:
    cfg, phase = ctx.cfg, ctx.phase
    decision_id = mp_plan._uid(cfg["seeds"][0], "validation", "step0", "decision")
    budget = freeze.data["attack_budget"] if freeze else ((read_decision(ctx.run_dir) or {}).get("decision") or "pgd20")

    def plan(pgd_budget):
        units = select_units(cfg, seeds, phase, stages, pgd_budget)
        if freeze is not None:
            allowed = {a for arms in freeze.data["models"].values() for a in arms}
            units = [u for u in units if u.params.get("model") in allowed or u.params.get("model") is None]
        return units

    units = plan(budget)
    deferred = False
    if phase == "validation" and read_decision(ctx.run_dir) is None and decision_id in {u.id for u in units}:
        gated, changed = {decision_id}, True
        while changed:
            changed = False
            for u in units:
                if u.id not in gated and gated & set(u.deps):
                    gated.add(u.id)
                    changed = True
        gated.discard(decision_id)
        deferred = bool(gated)
        units = [u for u in units if u.id not in gated]
    reused_msg(ctx, seeds, cfg)
    result = run_units(ctx, units, executors=executors, env=env, stop=stop, workers=args.workers,
                       retry_failed=args.retry_failed, worker_spec=spec)
    if deferred and result["exit_code"] in (EXIT_OK, EXIT_INVALID) and not stop():
        decision = read_decision(ctx.run_dir)
        if decision is not None:
            print(f"step 0 decision frozen: {decision['decision']} (output/step0/budget_decision.json); continuing with the remaining units")
            result = run_units(ctx, plan(decision["decision"]), executors=executors, env=env, stop=stop,
                               workers=args.workers, retry_failed=args.retry_failed, worker_spec=spec)
    for seed, count in sorted(result["reused_by_seed"].items()):
        print(f"reused {count} completed units (seed {seed})")
    print(json.dumps({"phase": phase, "counts": result["counts"], "exit_code": result["exit_code"]}))
    return result["exit_code"]


def reused_msg(ctx, seeds, cfg):
    """Seed 1 is not part of this run but has completed units: they are verified by the dependency check and reused."""
    first = cfg["seeds"][0]
    if first in seeds:
        return
    done = [r for r in ctx.state.list_units() if r["seed"] == first and r["status"] == "completed" and r["phase"] == ctx.phase]
    if done:
        for record in done:
            verify_unit_completion(record, ctx.run_dir)
        print(f"reused {len(done)} completed units (seed {first})")


# ------------------------------------------------------------------------------------------ plan / status / estimate
def cmd_plan(args, root) -> int:
    campaign_path = Path(args.config) if Path(args.config).is_absolute() else Path(root) / args.config
    cfg = load_campaign(campaign_path, root)
    seeds = parse_seeds(args.seeds, cfg)
    stages = parse_stages(args.stages, args.phase)
    units = build_plan(cfg, seeds, args.phase, stages, pgd_budget=args.pgd_budget)
    for u in units:
        print(f"{u.id}\t{u.kind}\tdeps={len(u.deps)}")
    kinds = mp_plan.counts_by_kind(units)
    print(f"{len(units)} units for seeds {seeds}, phase {args.phase}, pgd budget {args.pgd_budget}: "
          + ", ".join(f"{k}={v}" for k, v in kinds.items()))
    return EXIT_OK


def unit_summary(state, out_root, verify=False):
    rows, problems = [], []
    for rec in state.list_units():
        row = {"id": rec["id"], "status": rec["status"], "phase": rec["phase"], "seed": rec["seed"], "kind": rec["kind"],
               "seconds": rec["seconds"]}
        if verify and rec["status"] == "completed":
            try:
                verify_unit_completion(rec, out_root)
            except UnitCorruptionError as error:
                row["verify"] = f"CORRUPT: {error}"
                problems.append(row["id"])
        rows.append(row)
    return rows, problems


def cmd_status(args, root) -> int:
    run_dir, _ = resolve_paths(args, root)
    state = CampaignState(run_dir / "state")
    if not state.exists():
        print(f"no state under {run_dir}: nothing has run yet")
        return EXIT_OK
    rows, problems = unit_summary(state, run_dir, args.verify)
    rows = [r for r in rows if (args.phase is None or r["phase"] == args.phase) and (args.seed is None or r["seed"] == args.seed)]
    counts = {}
    for r in rows:
        counts.setdefault(r["phase"], {}).setdefault(r["kind"], {}).setdefault(r["status"], 0)
        counts[r["phase"]][r["kind"]][r["status"]] += 1
    decision = read_decision(run_dir)
    info = state.read_state()
    if args.json:
        print(json.dumps({"run": info, "counts": counts, "decision": decision and decision["decision"], "units": rows,
                          "corrupt": problems}, indent=2))
    else:
        print(f"run {info['run_name']} ({info['kind']}), seeds requested {info['seeds_requested']}; step 0 decision: "
              f"{decision['decision'] if decision else 'not made'}")
        for phase, kinds in counts.items():
            for kind, by_status in sorted(kinds.items()):
                print(f"  {phase:<10} {kind:<10} " + ", ".join(f"{s}={n}" for s, n in sorted(by_status.items())))
        for r in rows:
            if r["status"] != "completed":
                print(f"  NOT COMPLETED {r['id']}: {r['status']}")
        for p in problems:
            print(f"  CORRUPT {p}")
        print(f"{sum(r['status'] == 'completed' for r in rows)} of {len(rows)} recorded units completed")
    return EXIT_CONFIG if problems else EXIT_OK


def attack_key(unit):
    return (unit.kind, unit.params.get("attack") if unit.kind in ("attack",) else None)


def cmd_estimate(args, root) -> int:
    run_dir, campaign_path = resolve_paths(args, root)
    cfg = load_campaign(campaign_path, root)
    state = CampaignState(run_dir / "state")
    if not state.exists():
        raise ExitError(EXIT_CONFIG, "no completed units yet; run the seed-1 stages first")
    records = {r["id"]: r for r in state.list_units()}
    all_units = build_plan(cfg, cfg["seeds"], "validation", None, pgd_budget=(read_decision(run_dir) or {}).get("decision", "pgd20"))
    samples = {}
    for u in all_units:
        rec = records.get(u.id)
        if rec and rec["status"] in ("completed", "invalid") and rec["seconds"]:
            samples.setdefault(attack_key(u), []).append(rec["seconds"])
    medians = {k: statistics.median(v) for k, v in samples.items()}
    cost = lambda key: ATTACK_COST.get(key, 1)  # noqa: E731

    def estimate_for(unit):
        key = attack_key(unit)
        if key in medians:
            return medians[key]
        if unit.kind == "attack":  # scale another measured attack by steps x restarts
            for other, value in medians.items():
                if other[0] == "attack":
                    return value * cost(unit.params["attack"]) / cost(other[1])
        return None
    remaining = [u for u in all_units if not (records.get(u.id) and records[u.id]["status"] in ("completed", "invalid"))]
    per_stage, unknown = {}, []
    for u in remaining:
        value = estimate_for(u)
        if value is None:
            unknown.append(u.id)
            continue
        per_stage[u.stage] = per_stage.get(u.stage, 0.0) + value
    workers = max(args.workers, 1)
    report = {"workers": workers, "samples": {f"{k[0]}:{k[1]}": {"n": len(v), "median_seconds": medians[k]}
                                              for k, v in samples.items()},
              "remaining_units": len(remaining), "units_without_estimate": len(unknown),
              "hours_by_stage_validation": {s: v / workers / 3600 for s, v in sorted(per_stage.items())},
              "hours_validation_total": sum(per_stage.values()) / workers / 3600,
              "test_phase_note": "test attack and clean cost scale with row count (about the validation figure x test rows / validation rows)",
              "caveats": ["attack time depends on the step 0 decision: pgd50x5 multiplies pgd20 time by the measured 3.2-5.2x ratio",
                          "estimates use medians of completed units of this run; kinds with no sample are listed as without estimate"],
              "created_utc": utc_now()}
    durable_write_json(run_dir / "estimate.json", report)
    print(json.dumps(report, indent=2))
    return EXIT_OK


# ------------------------------------------------------------------------------------------ freeze-eval
def cmd_freeze_eval(args, root) -> int:
    run_dir, campaign_path = resolve_paths(args, root)
    cfg = load_campaign(campaign_path, root)
    seeds = parse_seeds(args.seeds, cfg)
    if not NAME_RE.match(args.freeze_name):
        raise ExitError(EXIT_CONFIG, f"--freeze-name must match {NAME_RE.pattern}")
    target = run_dir / "freezes" / f"{args.freeze_name}.json"
    if target.exists():
        raise ExitError(EXIT_CONFIG, f"freeze {args.freeze_name} exists; refusing to overwrite (use a new --freeze-name)")
    state = CampaignState(run_dir / "state")
    if not state.exists():
        raise ExitError(EXIT_GUARD, "no run state: nothing to freeze")
    current = compute_fingerprints(root, campaign_path)
    check_fingerprints(state, current, None, "test", run_dir)
    decision_path = run_dir / "step0" / "budget_decision.json"
    if not decision_path.is_file():
        raise ExitError(EXIT_GUARD, "step0/budget_decision.json is missing: finish seed-1 validation (stage step0) first")
    decision = read_decision(run_dir)
    stages = ("train", "calib", "paircheck", "ptq", "clean", "attack") + (("qat",) if args.include_qat else ())
    units = build_plan(cfg, seeds, "validation", stages, pgd_budget=decision["decision"])
    missing, blocked_qat = [], {}
    for u in units:
        rec = state.read_unit(u.id)
        ok = rec is not None and rec["status"] in ("completed", "invalid")
        if u.stage == "qat":
            if rec is not None and rec["status"] == "blocked" and rec["extra"].get("documented"):
                blocked_qat[u.id] = rec["error"]
                continue
            if rec is not None and rec["status"] == "blocked":
                ok = False
        if not ok:
            missing.append(f"{u.id} ({rec['status'] if rec else 'not run'})")
        elif rec["status"] == "completed":
            verify_unit_completion(rec, run_dir)
    if missing:
        raise ExitError(EXIT_GUARD, f"cannot freeze: {len(missing)} unit(s) not finished, e.g. " + "; ".join(missing[:8]))
    split_dir = Path(root) / json.loads((Path(root) / cfg["siblings"]["data_v2"]["path"]).read_text(encoding="utf-8"))["split"]["output_dir"]
    split_doc = json.loads((split_dir / "split_m2.json").read_text(encoding="utf-8"))
    arms = ["V1", "V2", "V3", "V4"]
    models = {}
    for seed in seeds:
        usable = list(arms)
        if args.include_qat:
            for arm in cfg["qat"]["arms"]:
                rec = state.read_unit(f"s{seed}.val.qat.{arm}")
                if rec is not None and rec["status"] == "completed":
                    usable.append(arm)
        models[str(seed)] = {a: freeze_model_entry(run_dir, seed, a) for a in usable}
    doc = {"schema": 1, "created_utc": utc_now(), "run_name": args.run_name, "freeze_name": args.freeze_name, "seeds": seeds,
           "fingerprints": {"components": current["components"], "pave_tree_id": current["pave_tree_id"]},
           "campaign_sha256": sha256_file(campaign_path),
           "sibling_sha256": {n: sha256_file(Path(root) / e["path"]) for n, e in cfg["siblings"].items()},
           "budget_decision_sha256": sha256_file(decision_path), "attack_budget": decision["decision"],
           "attack_row_policy_test": cfg["attack_row_policy"]["test"], "qat_included": bool(args.include_qat),
           "qat_blocked": blocked_qat, "models": models,
           "split": {"manifest_sha256": split_doc["manifest_sha256"], "id_sha256": split_doc["id_sha256"]},
           "test_partition_id_sha256": split_doc["id_sha256"]["test"]}
    durable_write_json(target, doc)
    digest = sha256_file(target)
    durable_write_bytes(target.with_name(target.name + ".sha256"), (digest + "\n").encode())
    print(f"freeze written: {target}\nsha256: {digest}\nlaunch the final test with:\n  python -m research.m2b_campaign run "
          f"--phase test --seeds {' '.join(map(str, seeds))} --freeze-name {args.freeze_name} --confirm-freeze {digest}")
    return EXIT_OK


# ------------------------------------------------------------------------------------------ efficiency
def busy_reason():
    """Why the machine is not quiet enough for timing, or None."""
    import psutil
    me = os.getpid()
    others = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(proc.info.get("cmdline") or [])
        if proc.info["pid"] != me and "research.m2b_campaign" in cmd and "efficiency" not in cmd:
            others.append(proc.info["pid"])
    if others:
        return f"other campaign processes are running: {others}"
    if hasattr(os, "getloadavg"):
        load, cores = os.getloadavg()[0], os.cpu_count() or 1
        if load >= 0.25 * cores:
            return f"1-minute load average {load:.2f} is not below {0.25 * cores:.2f} (0.25 x {cores} cores)"
    return None


def cmd_efficiency(args, root, executors=None) -> int:
    run_dir, campaign_path = resolve_paths(args, root)
    cfg = load_campaign(campaign_path, root)
    seeds = parse_seeds(args.seeds, cfg)
    state = CampaignState(run_dir / "state")
    if not state.exists():
        raise ExitError(EXIT_CONFIG, "no run state: train and quantize first")
    reason = busy_reason()
    if reason and not args.allow_busy:
        raise ExitError(EXIT_CONFIG, f"machine is busy ({reason}); rerun when quiet or pass --allow-busy (recorded)")
    current = compute_fingerprints(root, campaign_path)
    check_fingerprints(state, current, None, "validation", run_dir)
    runner_lock = unit_lock(state.dir / "runner.lock")
    try:
        runner_lock.__enter__()
    except LockHeldError as error:
        raise ExitError(EXIT_CONFIG, f"another run holds the output root: {error}") from error
    stop = StopFlag(state.dir / "STOP")
    try:
        setup_logging(run_dir, "efficiency")
        units = [u for u in build_plan(cfg, seeds, "validation", ("efficiency", "qat") if args.include_qat else ("efficiency",))
                 if u.kind == "efficiency"]
        for u in units:
            rec = state.read_unit(u.id)
            if rec and rec["status"] == "completed" and args.include_qat and not any(
                    "QAT-FT" in v.get("variants", []) for v in (rec["extra"] or {}).values()):
                raise ExitError(EXIT_GUARD, f"{u.id} completed without the QAT arms; use a new --run-name or archive it first")
        spec = {"root": str(root), "run_dir": str(run_dir), "campaign_path": str(campaign_path), "cfg": cfg,
                "phase": "validation", "fingerprints": current, "freeze_path": None, "freeze_sha": None,
                "extra": {"busy_override": bool(reason and args.allow_busy)}}
        ctx = Ctx(root, run_dir, campaign_path, cfg, state, current, "validation")
        env = make_env(spec, state, stop)
        result = run_units(ctx, units, executors=executors if executors is not None else get_executors(), env=env, stop=stop,
                           workers=1)
        print(json.dumps({"counts": result["counts"], "exit_code": result["exit_code"]}))
        return result["exit_code"]
    finally:
        runner_lock.__exit__(None, None, None)


# ------------------------------------------------------------------------------------------ other subcommands
def cmd_fingerprint_sets(args, root) -> int:
    from .m2b_state import fingerprint_sets
    target = Path(args.output) if args.output else Path(root) / "fingerprint_sets.json"
    write_fingerprint_sets(target, Path(root) / "research")
    print(runbook_table(fingerprint_sets(Path(root) / "research")))
    print(f"written: {target}")
    return EXIT_OK


def _delegate(module_name, function, args, root) -> int:
    """report, preflight and smoke live in their own modules (FEAT-004); a missing module is a refusal, not a crash."""
    try:
        module = importlib.import_module(f"research.{module_name}")
    except ImportError as error:
        raise ExitError(EXIT_CONFIG, f"`{args.command}` needs research/{module_name}.py, which is not part of this build: {error}") from error
    return getattr(module, function)(args, root)


# ------------------------------------------------------------------------------------------ CLI
def build_parser():
    parser = argparse.ArgumentParser(prog="python -m research.m2b_campaign", description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DEFAULT_CONFIG)
    common.add_argument("--run-name", default="m2b-v1")
    common.add_argument("--out-root", default="output")
    common.add_argument("--allow-outside-root", action="store_true", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", parents=[common], help="non-training environment, data and split checks")
    p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("plan", parents=[common], help="print the unit graph (no data is loaded)")
    p.add_argument("--phase", choices=("validation", "test"), default="validation")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--stages", nargs="+")
    p.add_argument("--pgd-budget", choices=("pgd20", "pgd50x5"), default="pgd20")

    p = sub.add_parser("run", parents=[common], help="execute units (resumes; never overwrites finished units)")
    p.add_argument("--phase", choices=("validation", "test"), required=True)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--stages", nargs="+")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--oversubscribe", action="store_true")
    p.add_argument("--freeze-name", default="eval_freeze")
    p.add_argument("--confirm-freeze", default=None, help="SHA-256 printed by freeze-eval (test phase only)")
    p.add_argument("--archive-and-restart-affected", dest="archive_tag", default=None, metavar="TAG")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--opt-in", choices=("qat",), default=None)

    p = sub.add_parser("status", parents=[common], help="which units completed")
    p.add_argument("--phase", choices=("validation", "test"))
    p.add_argument("--seed", type=int)
    p.add_argument("--json", action="store_true")
    p.add_argument("--verify", action="store_true", help="re-hash every completed unit's outputs")

    p = sub.add_parser("estimate", parents=[common], help="project the remaining time from completed units")
    p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("freeze-eval", parents=[common], help="freeze models and budget before the final test")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--freeze-name", default="eval_freeze")
    p.add_argument("--include-qat", action="store_true")

    p = sub.add_parser("efficiency", parents=[common], help="size, latency and memory on a quiet machine")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--include-qat", action="store_true")
    p.add_argument("--allow-busy", action="store_true")

    p = sub.add_parser("report", parents=[common], help="tables, plots and efficiency report")
    p.add_argument("--phase", choices=("validation", "test", "efficiency"), required=True)
    p.add_argument("--freeze-name", default="eval_freeze")
    p.add_argument("--include-qat", action="store_true")

    p = sub.add_parser("smoke", parents=[common], help="bounded smoke test (train and validation subsets only)")
    p.add_argument("--max-minutes", type=int, default=45)
    p.add_argument("--with-qat", action="store_true")

    p = sub.add_parser("fingerprint-sets", parents=[common], help="print and write the computed fingerprint sets")
    p.add_argument("--output", default=None)
    return parser


def main(argv=None, *, root=None, executors=None) -> int:
    root = Path(root) if root is not None else ROOT
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return int(exit_.code or 0)
    if args.command == "run" and args.phase == "validation" and args.confirm_freeze:
        print("ERROR --confirm-freeze belongs to --phase test", file=sys.stderr)
        return EXIT_CONFIG
    try:
        if args.command == "plan":
            return cmd_plan(args, root)
        if args.command == "run":
            return cmd_run(args, root, executors)
        if args.command == "status":
            return cmd_status(args, root)
        if args.command == "estimate":
            return cmd_estimate(args, root)
        if args.command == "freeze-eval":
            return cmd_freeze_eval(args, root)
        if args.command == "efficiency":
            return cmd_efficiency(args, root, executors)
        if args.command == "fingerprint-sets":
            return cmd_fingerprint_sets(args, root)
        module, function = {"preflight": ("m2b_preflight", "run"), "report": ("m2b_report", "run"),
                            "smoke": ("m2b_preflight", "run_smoke")}[args.command]
        return _delegate(module, function, args, root)
    except ExitError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return error.code
    except (FreezeGuardError, PartitionGuardError) as error:
        print(f"ERROR {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_GUARD
    except (ConfigError, ConfigIntegrityError, DataIntegrityError, FingerprintMismatchError, UnitCorruptionError, PlanError,
            ClosureResolutionError) as error:
        print(f"ERROR {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
