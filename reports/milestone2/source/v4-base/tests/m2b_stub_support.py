"""Stub executors and CLI helpers for the M2B scheduler and gating tests. No model, no data, no torch compute."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research import m2b_campaign as mc
from research import m2b_exec as ex
from research.m2b_io import durable_write_json

ALL_KINDS = ("train", "calib", "paircheck", "ptq", "clean", "e4", "step0", "attack", "diag", "fp32_ft", "qat", "ft_ptq",
             "efficiency")
FILES = {"train": ("checkpoint.pt", "train_log.json"), "fp32_ft": ("checkpoint.pt", "train_log.json"),
         "ptq": ("state.pt", "conversion.json"), "ft_ptq": ("state.pt", "conversion.json"),
         "qat": ("state.pt", "conversion.json"), "calib": ("calibration_ids.json", "calibration_x.pt")}
BASE_SEED1_STAGES = ["train", "calib", "paircheck", "ptq", "clean", "step0", "attack"]


def stub_executors(calls=None, fail=(), invalid=(), die=(), blocked=(), hook=None):
    """One stub for every kind. It records the unit id, then writes small deterministic files like the real executors."""
    calls = [] if calls is None else calls

    def run(unit, env):
        calls.append(unit.id)
        if hook is not None:
            hook(unit, env)
        if unit.id in die:
            raise SystemExit(1)
        if unit.id in fail:
            raise RuntimeError("stub failure")
        if unit.id in blocked:
            raise ex.UnitBlocked({"stage": "finetune", "module": None, "exception": "RuntimeError: stub QAT limitation"})
        if unit.kind == "step0":
            durable_write_json(env.run_dir / "step0" / "budget_decision.json", {"decision": "pgd20", "stub": True})
            return {"status": "completed", "outputs": ["step0/budget_decision.json"]}
        folder = env.run_dir / ex.unit_rel_dir(unit)
        folder.mkdir(parents=True, exist_ok=True)
        for name in FILES.get(unit.kind, ("out.txt",)):
            (folder / name).write_bytes(hashlib.sha256((unit.id + name).encode()).digest())
        return {"status": "invalid" if unit.id in invalid else "completed"}
    run.calls = calls
    return {kind: run for kind in ALL_KINDS}


def cli(tmp_path, *args, executors=None, run_name="t"):
    """Invoke m2b_campaign.main with the run directory under tmp_path (outside the package root, allowed explicitly)."""
    extra = ["--out-root", str(tmp_path), "--allow-outside-root", "--run-name", run_name]
    if args[0] == "run":
        extra.append("--oversubscribe")
    return mc.main([*args, *extra], executors=executors)


def run_dir(tmp_path, run_name="t") -> Path:
    return Path(tmp_path) / run_name


def unit_record(tmp_path, unit_id, run_name="t"):
    path = run_dir(tmp_path, run_name) / "state" / "units" / f"{unit_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def snapshot(folder) -> dict:
    """{relative path: sha256} of every file under folder (to prove that a refused command wrote nothing)."""
    folder = Path(folder)
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}
