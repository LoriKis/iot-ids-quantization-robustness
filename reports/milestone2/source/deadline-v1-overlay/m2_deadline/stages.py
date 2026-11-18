"""Campaign stages 0-9 (DESIGN 6.3 to 6.6): one function per stage plus the single marker call site.

Import-time rule: standard library + numpy only. torch and ``research.*`` are reached through a ``deps``
object whose attributes are imported lazily (``DefaultDeps``), so tests inject stubs and ``--help`` never
loads torch. Python 3.9 compatible.

Each ``stage_<name>(ctx)`` does its work and returns ``{"outputs": [...], "validate": callable}``.
:func:`run_stage` is the only caller of ``markers.write_after_validate``; nothing else writes a marker.
"""
from __future__ import annotations

import csv
import importlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import markers, metrics, protocol, sampling
from .common import (
    ADVERSARIAL,
    ATTACK_UNITS,
    CLEAN_ATTACK,
    CSV_COLUMNS,
    DeadlineError,
    EXIT_INFEASIBLE,
    EXIT_INTERRUPTED,
    GuardError,
    INT8_MODELS,
    KIND,
    MODEL_IDS,
    N_CLASSES,
    PARENTS,
    DataIntegrityError,
    RUN_ID,
    STAGES,
    UNIT_ARRAY_KEYS,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    checkpoint_path,
    deadline_meta_path,
    eps_str,
    model_dir,
    out_path,
    read_json,
    sha256_file,
    unit_dir,
    unit_id,
    utc_now,
    utc_stamp,
)

MIN_FREE_BYTES = 2 * 1024 ** 3
INT8_NOTE = "APPROXIMATE INT8 GRADIENT (BPDA)"
FULL_TEST_BATCH = 64


class GateFailedError(DeadlineError):
    """A blocking correctness gate or an attack unit failed (exit 4). No marker is written."""

    exit_code = EXIT_INFEASIBLE


class StageInterrupted(DeadlineError):
    """A stop request was honoured at an epoch boundary (exit 130)."""

    exit_code = EXIT_INTERRUPTED


# ---------------------------------------------------------------- lazy dependency handles
class DefaultDeps:
    """Lazy handles to the V4 library modules and the few torch-level helpers the stages need.

    Attribute access imports on first use (``deps.mt`` -> ``research.m2_training``). Tests pass any object
    with the same attributes instead.
    """

    _MODULES = {
        "torch": "torch",
        "mt": "research.m2_training",
        "mq": "research.m2_quant",
        "ma": "research.m2_attacks",
        "me": "research.m2_efficiency",
        "m2_split": "research.m2_split",
        "m2b_state": "research.m2b_state",
        "m2b_exec": "research.m2b_exec",
        "m2b_io": "research.m2b_io",
        "quantization": "research.quantization",
        "pda": "research.paper_described_adam",
    }

    def __getattr__(self, name: str) -> Any:
        target = self._MODULES.get(name)
        if target is None:
            raise AttributeError(name)
        module = importlib.import_module(target)
        self.__dict__[name] = module
        return module

    # ---- inference helpers
    def configure_inference(self, threads: int = 1, engine: str = "fbgemm") -> None:
        self.torch.set_num_threads(int(threads))
        self.torch.backends.quantized.engine = engine

    def sequences(self, x: Any):
        rows = np.asarray(x, dtype=np.float32)
        return self.torch.from_numpy(self.pda.to_sequence(rows))

    def logits(self, model: Any, x: Any, batch_size: int = FULL_TEST_BATCH) -> np.ndarray:
        torch = self.torch
        model.eval()
        seq = self.sequences(x)
        with torch.inference_mode():
            return torch.cat([model(b)["logits"] for b in seq.split(int(batch_size))]).numpy()

    def predict(self, model: Any, x: Any, batch_size: int = FULL_TEST_BATCH) -> np.ndarray:
        return self.m2b_exec._predict(model, self.sequences(x), int(batch_size))

    # ---- model and file helpers
    def load_fp32(self, checkpoint: Any, architecture: Dict[str, Any]):
        return self.me._load_checkpoint_model(checkpoint, architecture)

    def load_int8(self, spec: Dict[str, Any]):
        self.configure_inference(threads=1, engine=spec["backend"])
        return self.me.load_variant(spec)

    def parameter_count(self, model: Any) -> int:
        return int(sum(p.numel() for p in model.parameters()))

    def save_tensor(self, value: Any, path: Any) -> None:
        self.m2b_io.durable_write_torch(path, value)

    def load_tensor(self, path: Any):
        return self.torch.load(str(path), map_location="cpu", weights_only=True)

    def save_fp32_checkpoint(self, model: Any, path: Any, architecture: Dict[str, Any]) -> None:
        Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
        self.m2b_io.durable_write_torch(
            path, {"state_dict": model.state_dict(), "epoch": 0, "architecture": architecture}
        )

    def make_probe_optimizer(self, train_cfg: Dict[str, Any], model: Any):
        """Adam with populated state (one zero-gradient step) so a probe checkpoint has a realistic size."""
        opt = self.pda.build_optimizer(train_cfg, model)
        for p in model.parameters():
            p.grad = self.torch.zeros_like(p)
        opt.step()
        opt.zero_grad()
        return opt

    # ---- environment checks
    def import_check(self) -> None:
        for name in ("torch", "numpy", "pandas", "sklearn", "psutil"):
            importlib.import_module(name)
        for attr in ("mt", "mq", "ma", "me", "m2_split", "m2b_state", "m2b_exec", "quantization"):
            getattr(self, attr)

    def run_verify_overlay(self, overlay_root: Any, v4_root: Any) -> Tuple[int, str]:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([str(overlay_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        cmd = [sys.executable, "-B", "-m", "m2_deadline.verify_overlay", "--overlay", str(overlay_root),
               "--v4-root", str(v4_root)]
        done = subprocess.run(cmd, cwd=str(overlay_root), env=env, capture_output=True, text=True, timeout=1800)
        return done.returncode, (done.stdout or "") + (done.stderr or "")


# ---------------------------------------------------------------- run context
class RunContext:
    """Everything a stage needs: paths, locked values, lazy configs/arrays, the deps object and the stop flag."""

    def __init__(
        self,
        *,
        out: Any,
        v4_root: Any,
        config: Dict[str, Any],
        config_sha256: str,
        lock: Dict[str, Any],
        lock_sha256: str,
        estimate: Dict[str, Any],
        estimate_sha256: str,
        deps: Any = None,
        log: Optional[logging.Logger] = None,
        stop_flag: Optional[Callable[[], bool]] = None,
        overlay_root: Any = None,
        config_path: Any = None,
    ) -> None:
        self.config_path = config_path
        self.out = Path(str(out))
        self.v4_root = Path(str(v4_root))
        self.config = config
        self.config_sha256 = config_sha256
        self.lock = lock
        self.lock_sha256 = lock_sha256
        self.estimate = estimate
        self.estimate_sha256 = estimate_sha256
        self.deps = deps if deps is not None else DefaultDeps()
        self.log = log or logging.getLogger("m2_deadline.stages")
        self.stop_flag = stop_flag or (lambda: False)
        self.overlay_root = Path(str(overlay_root)) if overlay_root is not None else Path(__file__).resolve().parents[1]
        self.protocol_sha256: Optional[str] = None
        self.current_stage: Optional[str] = None
        self._cache: Dict[str, Any] = {}

    # ---- values
    @property
    def epochs(self) -> int:
        return int(self.lock["epochs"])

    @property
    def per_class(self) -> int:
        return int(self.lock["per_class_target"])

    @property
    def seed(self) -> int:
        return int(self.config["seed"])

    @property
    def split_dir(self) -> Path:
        return self.v4_root / self.config["split_dir"]

    @property
    def split_path(self) -> Path:
        return self.split_dir / "split_m2.json"

    @property
    def scaler_path(self) -> Path:
        return self.split_dir / "scaler_minmax.json"

    def sibling(self, name: str) -> Path:
        return self.v4_root / self.config["siblings"][name]

    def marker_ctx(self) -> Dict[str, Any]:
        return markers.context(
            config_sha256=self.config_sha256,
            lock_sha256=self.lock_sha256,
            protocol_sha256=self.protocol_sha256,
            epochs=self.epochs,
        )

    def refresh_protocol_sha(self) -> Optional[str]:
        path = out_path(self.out, "protocol")
        self.protocol_sha256 = sha256_file(path) if path.is_file() else None
        return self.protocol_sha256

    def check_stop(self) -> None:
        if self.stop_flag():
            raise StageInterrupted("stop requested")

    # ---- lazy configs (V4 loaders, so they validate exactly like V4 does)
    def _cached(self, key: str, make: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def train_cfg(self) -> Dict[str, Any]:
        return self._cached("train_cfg", lambda: self.deps.mt.load_config(self.sibling("training")))

    def quant_cfg(self) -> Dict[str, Any]:
        return self._cached("quant_cfg", lambda: self.deps.mq.load_config(self.sibling("quantization")))

    def att_cfg(self) -> Dict[str, Any]:
        return self._cached("att_cfg", lambda: self.deps.ma.load_config(self.sibling("attacks")))

    def eff_cfg(self) -> Dict[str, Any]:
        return self._cached("eff_cfg", lambda: self.deps.me.load_config(self.sibling("efficiency")))

    def split_doc(self) -> Dict[str, Any]:
        return self._cached("split_doc", lambda: read_json(self.split_path))

    @property
    def split_seed(self) -> int:
        return int(self.split_doc()["split_seed"])

    def architecture(self) -> Dict[str, Any]:
        return self.train_cfg()["architecture"]

    # ---- data (train + validation only; test arrays are loaded by stage test_clean alone)
    def trainval(self) -> Dict[str, Any]:
        def load() -> Dict[str, Any]:
            return self.deps.m2_split.load_arrays(self.split_path, self.scaler_path, ("train", "validation"))

        return self._cached("trainval", load)

    # ---- models
    def calibration_x(self):
        return self._cached("calibration_x", lambda: self.deps.load_tensor(out_path(self.out, "calibration_x")))

    def conversion(self, model_id: str) -> Dict[str, Any]:
        return read_json(model_dir(self.out, model_id) / "conversion.json")

    def int8_spec(self, model_id: str) -> Dict[str, Any]:
        ptq = self.quant_cfg()["ptq"]
        return {
            "kind": "int8",
            "state": checkpoint_path(self.out, model_id),
            "float_checkpoint": checkpoint_path(self.out, PARENTS[model_id]),
            "calibration_path": out_path(self.out, "calibration_x"),
            "scope": ptq["scope"],
            "backend": ptq["backend"],
            "architecture": self.architecture(),
        }

    def load_model(self, model_id: str):
        """Load a model only after its ``deadline_meta.json`` verifies for this run (DESIGN 6.6)."""

        def load() -> Any:
            markers.verify_deadline_meta(self.out, model_id, config_sha256=self.config_sha256, epochs=self.epochs)
            if model_id in INT8_MODELS:
                return self.deps.load_int8(self.int8_spec(model_id))
            return self.deps.load_fp32(checkpoint_path(self.out, model_id), self.architecture())

        return self._cached("model_" + model_id, load)

    def fingerprints(self, model_id: str) -> Dict[str, Any]:
        return {
            "kind": KIND,
            "run_id": RUN_ID,
            "config_sha256": self.config_sha256,
            "epochs": self.epochs,
            "adversarial": bool(ADVERSARIAL[model_id]),
            "lock_sha256": self.lock_sha256,
        }


# ---------------------------------------------------------------- small helpers
def _quarantine(path: Path, root: Path, tag: str) -> Path:
    """Move ``path`` to ``root/<tag>/<name>`` (never deletes)."""
    dest_dir = root / tag
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    n = 1
    while dest.exists():
        dest = dest_dir / ("%s.%d" % (path.name, n))
        n += 1
    os.replace(str(path), str(dest))
    return dest


def _quarantine_partial(directory: Path, names: Sequence[str], tag: str) -> List[Path]:
    """If only some of the output files exist, move the existing ones to ``_incomplete/<tag>/``."""
    existing = [directory / n for n in names if (directory / n).exists()]
    if not existing or len(existing) == len(names):
        return []
    return [_quarantine(p, directory / "_incomplete", tag) for p in existing]


def _csv_bytes(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(list(columns))
    for row in rows:
        writer.writerow(list(row))
    return buf.getvalue().encode("utf-8")


def _read_csv_rows(path: Path, columns: Sequence[str]) -> List[Dict[str, str]]:
    with open(str(path), "r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise DataIntegrityError("%s: header %s differs from %s" % (path.name, reader.fieldnames, list(columns)))
        return list(reader)


def _npz_bytes(**arrays: Any) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _subset(out: Path) -> Tuple[np.ndarray, np.ndarray]:
    return protocol.parse_ids_csv(out_path(out, "subset_ids").read_bytes())


def _protocol_doc(out: Path) -> Dict[str, Any]:
    return read_json(out_path(out, "protocol"))


def _full_test_predictions(out: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rows = _read_csv_rows(out_path(out, "predictions_full_test"), CSV_COLUMNS["predictions_full_test"])
    ids = np.array([int(r["example_id"]) for r in rows], dtype=np.int64)
    preds = {m: np.array([int(r["%s_pred" % m]) for r in rows], dtype=np.int64) for m in MODEL_IDS}
    return ids, preds


def _finish(outputs: Sequence[Any], validate: Callable[[], Any]) -> Dict[str, Any]:
    return {"outputs": list(outputs), "validate": validate}


# ---------------------------------------------------------------- stage 0: preflight
def stage_preflight(ctx: RunContext) -> Dict[str, Any]:
    out = ctx.out
    markers.assert_not_smoke(out)
    for base in (ctx.v4_root / "output",):
        try:
            out.resolve().relative_to(base.resolve())
        except ValueError:
            continue
        raise GuardError("OUT may not live under the V4 output folder: %s" % out)
    lock_path = out_path(out, "budget_lock")
    estimate_path = out_path(out, "timing_estimate")
    if not lock_path.is_file() or not estimate_path.is_file():
        raise GuardError("budget lock or timing estimate missing under %s" % out_path(out, "timing_dir"))
    if sha256_file(lock_path) != ctx.lock_sha256 or sha256_file(estimate_path) != ctx.estimate_sha256:
        raise GuardError("lock or estimate changed since this campaign was started")
    if ctx.lock.get("estimate_sha256") != ctx.estimate_sha256:
        raise GuardError("budget lock estimate_sha256 differs from timing_estimate.json")
    if ctx.lock.get("config_sha256") != ctx.config_sha256 or ctx.estimate.get("config_sha256") != ctx.config_sha256:
        raise GuardError("config hash differs from the one the budget was locked with")
    if ctx.lock.get("performance_inspected") is not False:
        raise GuardError("budget lock does not state performance_inspected == false")
    v4_manifest = sha256_file(ctx.v4_root / "MANIFEST.sha256")
    if ctx.estimate.get("v4_manifest_sha256") != v4_manifest:
        raise GuardError("V4 MANIFEST.sha256 differs from the one the budget was measured against")
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(str(out)).free
    if free < MIN_FREE_BYTES:
        raise GuardError("only %.2f GB free under %s, need at least 2 GB" % (free / 1024 ** 3, out))
    ctx.deps.import_check()
    rc, text = ctx.deps.run_verify_overlay(ctx.overlay_root, ctx.v4_root)
    if rc != 0:
        raise GuardError("verify_overlay failed (exit %s): %s" % (rc, text.strip()[-600:]))
    return _finish([lock_path, estimate_path], lambda: True)


# ---------------------------------------------------------------- stage 1: freeze
def stage_freeze(ctx: RunContext) -> Dict[str, Any]:
    """Freeze protocol and subset from labels and ids only. Never calls ``load_arrays``."""
    out = ctx.out
    split = ctx.split_doc()
    data_cfg = read_json(ctx.sibling("data"))
    dataset_csv = ctx.v4_root / split["dataset"]["path"]
    test_ids, y_test = protocol.read_test_labels(split, data_cfg.get("dataset"), dataset_csv)
    eff_json = read_json(ctx.sibling("efficiency"))
    freeze = protocol.build_freeze(
        config=ctx.config,
        config_sha256=ctx.config_sha256,
        lock=ctx.lock,
        lock_sha256=ctx.lock_sha256,
        split=split,
        split_m2_sha256=sha256_file(ctx.split_path),
        scaler_sha256=sha256_file(ctx.scaler_path),
        efficiency_cfg=eff_json,
        test_ids=test_ids,
        y_test=y_test,
    )
    written = protocol.write_protocol(out, freeze, utc_now())  # raises GuardError if different content exists
    ctx.protocol_sha256 = written["protocol_sha256"]
    identity = markers.build_run_identity(
        config_sha256=ctx.config_sha256,
        lock_sha256=ctx.lock_sha256,
        epochs=ctx.epochs,
        protocol_sha256=ctx.protocol_sha256,
        v4_manifest_sha256=sha256_file(ctx.v4_root / "MANIFEST.sha256"),
        source_sha256=markers.source_hashes(ctx.v4_root),
    )
    markers.write_run_identity(out, identity)
    ctx.log.info("freeze: protocol %s, subset n=%d (written=%s)", ctx.protocol_sha256, freeze["subset"]["n_actual"], written["written"])

    def validate() -> bool:
        sha = protocol.assert_frozen(out)
        if sha != ctx.protocol_sha256 or sha256_file(out_path(out, "protocol")) != freeze["protocol_sha256"]:
            raise GuardError("protocol.json hash differs from the freshly computed freeze")
        ids, labels = _subset(out)
        if len(ids) != freeze["subset"]["n_actual"] or len(ids) > N_CLASSES * ctx.per_class:
            raise GuardError("subset size %d is inconsistent (target %d per class)" % (len(ids), ctx.per_class))
        if len(np.unique(ids)) != len(ids) or not set(int(i) for i in ids) <= set(int(i) for i in test_ids):
            raise GuardError("subset ids are not unique members of the test partition")
        counts = read_json(out_path(out, "subset_counts"))
        if sum(int(v["selected"]) for v in counts["counts"].values()) != len(ids):
            raise GuardError("subset counts do not sum to the subset size")
        if sampling.ids_sha256(ids) != _protocol_doc(out)["robustness"]["subset_ids_sha256"]:
            raise GuardError("subset ids hash differs from protocol.json")
        return True

    outputs = [out_path(out, k) for k in ("subset_ids", "subset_counts", "protocol", "protocol_sha256", "protocol_frozen", "run_identity")]
    return _finish(outputs, validate)


# ---------------------------------------------------------------- stages 2 and 3: training
def _train_stage(ctx: RunContext, model_id: str) -> Dict[str, Any]:
    out = ctx.out
    adversarial = bool(ADVERSARIAL[model_id])
    mdir = model_dir(out, model_id)
    markers.assert_not_smoke(mdir)
    ckpt = mdir / "checkpoint.pt"
    log_path = mdir / "train_log.json"
    mdir.mkdir(parents=True, exist_ok=True)
    if ckpt.is_file() and log_path.is_file():
        ctx.log.info("%s: checkpoint.pt and train_log.json exist; skipping train_unit and validating", model_id)
    else:
        _quarantine_partial(mdir, ("checkpoint.pt", "train_log.json"), utc_stamp())
        train = ctx.trainval()["train"]
        mt = ctx.deps.mt
        try:
            mt.train_unit(
                ctx.train_cfg(),
                train["x"],
                train["y"],
                ctx.seed,
                mdir,
                adversarial=adversarial,
                epochs=ctx.epochs,
                threads=int(ctx.config["training"]["threads"]),
                checkpoint_dir=mdir / "ckpt",
                keep_last=int(ctx.config["training"]["keep_last"]),
                stop_flag=ctx.stop_flag,
                fingerprints=ctx.fingerprints(model_id),
            )
        except mt.TrainingInterrupted as exc:
            raise StageInterrupted(str(exc)) from exc

    log = _check_train_log(ctx, model_id)
    meta_path = deadline_meta_path(out, model_id)
    if meta_path.is_file():
        markers.verify_deadline_meta(out, model_id, config_sha256=ctx.config_sha256, epochs=ctx.epochs)
    else:
        markers.write_deadline_meta(
            out,
            model_id,
            config_sha256=ctx.config_sha256,
            epochs=ctx.epochs,
            extra={
                "init_state_hash": log["init_state_hash"],
                "final_state_hash": log["final_state_hash"],
                "optimizer_updates": log["optimizer_updates"],
            },
        )

    def validate() -> bool:
        _check_train_log(ctx, model_id)
        markers.verify_deadline_meta(out, model_id, config_sha256=ctx.config_sha256, epochs=ctx.epochs)
        return True

    return _finish([ckpt, log_path, meta_path], validate)


def _check_train_log(ctx: RunContext, model_id: str) -> Dict[str, Any]:
    mdir = model_dir(ctx.out, model_id)
    log = read_json(mdir / "train_log.json")
    checks = (
        ("epochs", log.get("epochs"), ctx.epochs),
        ("adversarial", log.get("adversarial"), bool(ADVERSARIAL[model_id])),
        ("seed", log.get("seed"), ctx.seed),
        ("train_rows", log.get("train_rows"), int(ctx.estimate["data"]["n_train"])),
        ("checkpoint_sha256", log.get("checkpoint_sha256"), sha256_file(mdir / "checkpoint.pt")),
    )
    for name, got, want in checks:
        if got != want:
            raise GuardError("%s train_log.%s is %r, expected %r" % (model_id, name, got, want))
    if model_id == "V3":
        v1 = read_json(model_dir(ctx.out, "V1") / "train_log.json")
        for name in ("init_state_hash", "optimizer_updates", "epochs", "train_rows"):
            if v1.get(name) != log.get(name):
                raise GuardError("pair check failed: V1 and V3 differ in %s (%r vs %r)" % (name, v1.get(name), log.get(name)))
    return log


def stage_train_ordinary(ctx: RunContext) -> Dict[str, Any]:
    return _train_stage(ctx, "V1")


def stage_train_adversarial(ctx: RunContext) -> Dict[str, Any]:
    return _train_stage(ctx, "V3")


# ---------------------------------------------------------------- stage 4: PTQ
def stage_ptq(ctx: RunContext) -> Dict[str, Any]:
    out = ctx.out
    quant_cfg = ctx.quant_cfg()
    per_class = int(quant_cfg["ptq"]["calibration_per_class"])
    if per_class != 20:
        raise GuardError("calibration_per_class is %r, the design requires 20" % per_class)
    arrays = ctx.trainval()
    train = arrays["train"]
    mq = ctx.deps.mq
    cal = mq.calibration_ids({"train": train}, per_class, ctx.split_seed)
    if not set(int(i) for i in cal["ids"]) <= set(int(i) for i in train["ids"]):
        raise GuardError("calibration ids are not drawn from the train partition")
    cal_ids_path = out_path(out, "calibration_ids")
    cal_x_path = out_path(out, "calibration_x")
    cal_ids_path.parent.mkdir(parents=True, exist_ok=True)
    if cal_ids_path.is_file() and cal_x_path.is_file():
        if read_json(cal_ids_path).get("id_sha256") != cal["id_sha256"]:
            raise GuardError("existing calibration_ids.json differs from the train-only calibration rows")
    else:
        _quarantine_partial(cal_ids_path.parent, ("calibration_ids.json", "calibration_x.pt"), utc_stamp())
        mq.save_calibration(cal, cal_ids_path)
        ctx.deps.save_tensor(cal["x"], cal_x_path)

    val_seq = None
    outputs: List[Path] = [cal_ids_path, cal_x_path]
    for model_id in ("V2", "V4"):
        parent_id = PARENTS[model_id]
        mdir = model_dir(out, model_id)
        markers.assert_not_smoke(mdir)
        mdir.mkdir(parents=True, exist_ok=True)
        state, conv = mdir / "state.pt", mdir / "conversion.json"
        parent_sha = sha256_file(checkpoint_path(out, parent_id))
        if state.is_file() and conv.is_file():
            ctx.log.info("%s: state.pt and conversion.json exist; skipping ptq_unit and validating", model_id)
        else:
            _quarantine_partial(mdir, ("state.pt", "conversion.json"), utc_stamp())
            parent = ctx.load_model(parent_id)
            if val_seq is None:
                val_seq = ctx.deps.sequences(arrays["validation"]["x"])
            started = time.perf_counter()
            mq.ptq_unit(parent, cal, quant_cfg, mdir, variant=model_id, eval_x=val_seq)
            ctx.log.info("%s: ptq_unit %.1f s", model_id, time.perf_counter() - started)
        meta_path = mdir / "deadline_meta.json"
        if meta_path.is_file():
            meta = markers.verify_deadline_meta(out, model_id, config_sha256=ctx.config_sha256, epochs=ctx.epochs)
            if (meta.get("extra") or {}).get("parent_checkpoint_sha256") != parent_sha:
                raise GuardError("%s was quantized from a different %s checkpoint" % (model_id, parent_id))
        else:
            markers.write_deadline_meta(
                out, model_id, config_sha256=ctx.config_sha256, epochs=ctx.epochs,
                extra={"parent": parent_id, "parent_checkpoint_sha256": parent_sha,
                       "calibration_id_sha256": cal["id_sha256"]},
            )
        outputs += [state, conv, meta_path]

    expected = int(quant_cfg["ptq"]["expected_quantized_linear"])

    def validate() -> bool:
        for model_id in ("V2", "V4"):
            conversion = ctx.conversion(model_id)
            if len(conversion["requested"]) != expected:
                raise GuardError("%s has %d quantized linears, expected %d" % (model_id, len(conversion["requested"]), expected))
            reload_check = conversion.get("reload_check") or {}
            if reload_check.get("logits_bitwise_equal") is not True or reload_check.get("argmax_identical") is not True:
                raise GuardError("%s INT8 reload check is not bit-exact" % model_id)
            if conversion["calibration"]["id_sha256"] != cal["id_sha256"]:
                raise GuardError("%s conversion.json names another calibration set" % model_id)
            markers.verify_deadline_meta(out, model_id, config_sha256=ctx.config_sha256, epochs=ctx.epochs)
        return True

    return _finish(outputs, validate)


# ---------------------------------------------------------------- stage 5: validation checks
def stage_val_check(ctx: RunContext) -> Dict[str, Any]:
    """Correctness gates on validation data only. No accuracy or F1 is computed here."""
    out = ctx.out
    ctx.deps.configure_inference(threads=1)
    val = ctx.trainval()["validation"]
    n_val = int(len(val["ids"]))
    quant_cfg = ctx.quant_cfg()
    expected_islands = int(quant_cfg["ptq"]["expected_quantized_linear"])
    preds: Dict[str, np.ndarray] = {}
    models: Dict[str, Any] = {}
    gates: Dict[str, Dict[str, bool]] = {
        "finite_logits": {}, "pred_range_0_26": {}, "ptq_reload_bitexact": {}, "n_quantized_linears_23": {},
    }
    if sorted(gates) != sorted(ctx.config["val_check"]["blocking"]):
        raise GuardError("config val_check.blocking differs from the implemented gates")
    for model_id in MODEL_IDS:
        logits = np.asarray(ctx.deps.logits(ctx.load_model(model_id), val["x"], FULL_TEST_BATCH))
        pred = logits.argmax(1).astype(np.int64)
        preds[model_id] = pred
        gates["finite_logits"][model_id] = bool(np.isfinite(logits).all() and logits.shape == (n_val, N_CLASSES))
        gates["pred_range_0_26"][model_id] = bool(len(pred) == n_val and pred.min() >= 0 and pred.max() < N_CLASSES)
        models[model_id] = {"n": int(len(pred)), "pred_min": int(pred.min()), "pred_max": int(pred.max())}
        if model_id in INT8_MODELS:
            conversion = ctx.conversion(model_id)
            reload_check = conversion.get("reload_check") or {}
            gates["ptq_reload_bitexact"][model_id] = bool(
                reload_check.get("logits_bitwise_equal") is True and reload_check.get("argmax_identical") is True
            )
            gates["n_quantized_linears_23"][model_id] = bool(len(conversion["requested"]) == expected_islands)
    agreement = {m: float(np.mean(preds[m] == preds[PARENTS[m]])) for m in INT8_MODELS}
    passed = all(all(per_model.values()) for per_model in gates.values())
    doc = {
        "schema": 1,
        "kind": "m2_deadline_val_check",
        "run_id": RUN_ID,
        "n_val": n_val,
        "models": models,
        "gates": gates,
        "informational": {"int8_parent_argmax_agreement": agreement},
        "passed": passed,
        "note": "validation data only; no accuracy or F1 was computed and nothing here selects any setting",
    }
    path = out_path(out, "val_check")
    atomic_write_json(path, doc)
    if not passed:
        failed = sorted("%s/%s" % (g, m) for g, per in gates.items() for m, ok in per.items() if not ok)
        raise GateFailedError(
            "blocking validation gates failed: %s. Do not rerun with changed settings; inspect %s and the log." % (failed, path)
        )

    def validate() -> bool:
        saved = read_json(path)
        return bool(saved.get("passed") is True and int(saved["n_val"]) == n_val)

    return _finish([path], validate)


# ---------------------------------------------------------------- stage 6: full-test clean evaluation
def stage_test_clean(ctx: RunContext) -> Dict[str, Any]:
    """The only stage that obtains test arrays (``verify_eval_freeze`` then ``load_arrays(final_test)``)."""
    out = ctx.out
    deps = ctx.deps
    sha = protocol.assert_frozen(out)
    if ctx.protocol_sha256 != sha:
        raise GuardError("protocol hash in memory differs from protocol.json")
    proto = _protocol_doc(out)
    freeze = deps.m2b_state.verify_eval_freeze(out_path(out, "protocol"), sha)
    protocol.assert_frozen(out)  # again, immediately before the guarded load
    test = deps.m2_split.load_arrays(
        ctx.split_path, ctx.scaler_path, ("test",), purpose="final_test", freeze=freeze
    )["test"]
    test_ids = np.asarray(test["ids"], dtype=np.int64)
    y_test = np.asarray(test["y"], dtype=np.int64)
    if sampling.ids_sha256(test_ids) != proto["test_partition_id_sha256"]:
        raise GuardError("loaded test ids differ from the frozen test partition")
    subset_ids, subset_y = _subset(out)
    position = {int(i): n for n, i in enumerate(test_ids)}
    if any(int(i) not in position for i in subset_ids):
        raise GuardError("a frozen subset id is not in the test partition")
    rows = np.array([position[int(i)] for i in subset_ids], dtype=np.int64)
    if not np.array_equal(y_test[rows], subset_y):
        raise GuardError("frozen subset labels differ from the test labels")
    subset_rows_path = out_path(out, "subset_rows")
    atomic_write_bytes(
        subset_rows_path,
        _npz_bytes(ids=subset_ids, x=np.ascontiguousarray(test["x"], dtype=np.float32)[rows], y=subset_y),
    )

    deps.configure_inference(threads=1)
    preds: Dict[str, np.ndarray] = {}
    for model_id in MODEL_IDS:
        started = time.perf_counter()
        preds[model_id] = np.asarray(deps.predict(ctx.load_model(model_id), test["x"], FULL_TEST_BATCH), dtype=np.int64)
        ctx.log.info("test_clean: %s predicted %d rows in %.1f s", model_id, len(test_ids), time.perf_counter() - started)
    flagged = set(int(i) for i in subset_ids)
    csv_rows = [
        [int(test_ids[n]), int(y_test[n]), "true" if int(test_ids[n]) in flagged else "false"]
        + [int(preds[m][n]) for m in MODEL_IDS]
        for n in range(len(test_ids))
    ]
    pred_path = out_path(out, "predictions_full_test")
    atomic_write_bytes(pred_path, _csv_bytes(CSV_COLUMNS["predictions_full_test"], csv_rows))
    results = {
        "schema": 1,
        "kind": "m2_deadline_clean_results",
        "run_id": RUN_ID,
        "n_test": int(len(test_ids)),
        "labels": list(range(N_CLASSES)),
        "batch_size": FULL_TEST_BATCH,
        "models": {
            m: {
                "accuracy": metrics.accuracy(y_test, preds[m]),
                "macro_f1": metrics.macro_f1(y_test, preds[m], labels=range(N_CLASSES), zero_division=0),
                "n": int(len(test_ids)),
            }
            for m in MODEL_IDS
        },
    }
    clean_path = out_path(out, "clean_results")
    atomic_write_json(clean_path, results)

    def validate() -> bool:
        ids, saved = _full_test_predictions(out)
        if len(ids) != int(results["n_test"]) or sampling.ids_sha256(ids) != proto["test_partition_id_sha256"]:
            raise GuardError("predictions_full_test.csv ids differ from the split's test ids")
        for m in MODEL_IDS:
            if len(saved[m]) != len(ids) or saved[m].min() < 0 or saved[m].max() >= N_CLASSES:
                raise GuardError("predictions for %s are out of range" % m)
        s_ids, s_x, s_y = _load_subset_rows(out)
        if not np.array_equal(s_ids, subset_ids) or not np.array_equal(s_y, subset_y):
            raise GuardError("robustness_subset_rows.npz differs from the frozen subset")
        return True

    return _finish([pred_path, clean_path, subset_rows_path], validate)


def _load_subset_rows(out: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(str(out_path(out, "subset_rows"))) as data:
        return (
            np.asarray(data["ids"], dtype=np.int64),
            np.asarray(data["x"], dtype=np.float32),
            np.asarray(data["y"], dtype=np.int64),
        )


# ---------------------------------------------------------------- stage 7: robustness
def _unit_done_ok(out: Path, model_id: str, attack: str, eps: float, subset_hash: str) -> Optional[Dict[str, Any]]:
    """The unit's ``unit_done.json`` if it validates (keys, valid flag, ids hash, file hashes), else None."""
    udir = unit_dir(out, model_id, attack, eps)
    path = udir / "unit_done.json"
    if not path.is_file():
        return None
    try:
        doc = read_json(path)
    except ValueError:
        return None
    needed = ("model_id", "attack", "epsilon", "unit_id", "ids_sha256", "valid", "files", "n", "accuracy", "macro_f1", "robust_accuracy")
    if any(k not in doc for k in needed):
        return None
    if doc["valid"] is not True or doc["ids_sha256"] != subset_hash or doc["unit_id"] != unit_id(attack, eps):
        return None
    for name, sha in doc["files"].items():
        f = udir / name
        if not f.is_file() or sha256_file(f) != sha:
            return None
    return doc


def _attack_name(attack: str) -> str:
    return "fgsm" if attack in (CLEAN_ATTACK, "fgsm") else "pgd5"


def stage_robustness(ctx: RunContext) -> Dict[str, Any]:
    out = ctx.out
    deps = ctx.deps
    sha = protocol.assert_frozen(out)
    if ctx.protocol_sha256 != sha:
        raise GuardError("protocol hash in memory differs from protocol.json")
    proto = _protocol_doc(out)
    rob_proto = proto["robustness"]
    subset_ids, subset_y = _subset(out)
    subset_hash = rob_proto["subset_ids_sha256"]
    if sampling.ids_sha256(subset_ids) != deps.m2_split.id_hash(subset_ids) or sampling.ids_sha256(subset_ids) != subset_hash:
        raise GuardError("sampling.ids_sha256 differs from m2_split.id_hash or from the frozen subset hash")
    label_hash = sampling.label_sha256(subset_y)
    if label_hash != rob_proto["subset_label_sha256"]:
        raise GuardError("subset label hash differs from protocol.json")
    s_ids, s_x, s_y = _load_subset_rows(out)
    if not np.array_equal(s_ids, subset_ids) or not np.array_equal(s_y, subset_y):
        raise GuardError("robustness_subset_rows.npz differs from the frozen subset")

    # G1 + G2: pgd5 added to a validated deep copy, then the partition switched on a deep copy (never re-validated)
    pgd_spec = ctx.config["robustness"]["attacks"]["pgd5"]
    cfg = protocol.build_pgd5_cfg(ctx.att_cfg(), pgd_spec)
    deps.ma.validate_config(cfg)
    test_cfg = deps.m2b_exec.deep_copy_test_attack_cfg(cfg)
    if int(test_cfg["evaluation"]["batch_size"]) != int(rob_proto["batch_size"]):
        raise GuardError("attack batch size differs from the frozen protocol")
    deps.configure_inference(threads=1)

    full_ids, full_preds = _full_test_predictions(out)
    started_all = time.perf_counter()
    invalid: List[str] = []
    incomplete_tag = utc_stamp()
    for model_id in MODEL_IDS:
        is_int8 = model_id in INT8_MODELS
        source = ctx.config["robustness"]["methods"]["int8" if is_int8 else "fp32"]
        models: Optional[Dict[str, Any]] = None
        model_ids: Optional[Dict[str, str]] = None
        for attack, eps in ATTACK_UNITS:
            ctx.check_stop()
            udir = unit_dir(out, model_id, attack, eps)
            if _unit_done_ok(out, model_id, attack, eps, subset_hash) is not None:
                ctx.log.info("robustness: %s/%s already done", model_id, unit_id(attack, eps))
                continue
            if udir.exists():
                _quarantine(udir, out_path(out, "robustness_dir") / "_incomplete", incomplete_tag + "/" + model_id)
            if models is None:
                if is_int8:
                    q = ctx.load_model(model_id)
                    bpda = deps.quantization.bpda_model(q, ctx.conversion(model_id)["requested"], ctx.calibration_x()[:2])
                    models = {"int8": q, "bpda": bpda}
                    model_ids = {"fp32": PARENTS[model_id], "int8": model_id, "bpda": model_id + "_bpda"}
                else:
                    models = {"fp32": ctx.load_model(model_id)}
                    model_ids = {"fp32": model_id}
            udir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            result = deps.ma.evaluate_unit(
                source=source,
                attack_name=_attack_name(attack),
                epsilon=float(eps),
                x=s_x,
                y=s_y,
                ids=s_ids,
                models=models,
                cfg=test_cfg,
                train_seed=ctx.seed,
                model_id=model_id,
                evaluation_partition="test",
                model_ids=model_ids,
                notes=[INT8_NOTE] if is_int8 else None,
                out_dir=udir,
                expected_ids_sha256=subset_hash,
                expected_label_sha256=label_hash,
                raise_on_invalid=False,
            )
            seconds = time.perf_counter() - started
            checks = result["checks"]
            if result.get("row") is None or result.get("arrays") is None or not checks.get("valid"):
                invalid.append("%s/%s: %s" % (model_id, unit_id(attack, eps), checks.get("failures")))
                ctx.log.error("robustness: %s/%s INVALID %s (kept in %s)", model_id, unit_id(attack, eps), checks.get("failures"), udir)
                continue
            _write_unit_done(ctx, udir, model_id, attack, eps, source, result, subset_hash, full_ids, full_preds[model_id])
            ctx.log.info("robustness: %s/%s done in %.1f s", model_id, unit_id(attack, eps), seconds)
    if invalid:
        raise GateFailedError("attack units invalid (no marker written; rerun resume to redo them): " + "; ".join(invalid))
    ctx.log.info("robustness: all units finished in %.1f s", time.perf_counter() - started_all)

    results_path = out_path(out, "robustness_results")
    pred_path = out_path(out, "predictions_subset")
    units: Dict[str, Dict[str, Any]] = {}
    eps0: Dict[str, Any] = {}
    csv_rows: List[List[Any]] = []
    unit_files: List[Path] = []
    for model_id in MODEL_IDS:
        units[model_id] = {}
        for attack, eps in ATTACK_UNITS:
            doc = _unit_done_ok(out, model_id, attack, eps, subset_hash)
            if doc is None:
                raise GateFailedError("unit %s/%s has no valid unit_done.json" % (model_id, unit_id(attack, eps)))
            units[model_id][unit_id(attack, eps)] = {
                k: doc[k] for k in ("n", "accuracy", "macro_f1", "robust_accuracy")
            }
            if attack == CLEAN_ATTACK:
                eps0[model_id] = doc.get("eps0_consistency") or {}
            udir = unit_dir(out, model_id, attack, eps)
            unit_files.append(udir / "unit_done.json")
            with np.load(str(udir / "arrays.npz")) as arrays:
                a_ids, a_y, a_pred = arrays["ids"], arrays["y"], arrays["pred_adv"]
            if not np.array_equal(a_ids, subset_ids) or not np.array_equal(a_y, subset_y):
                raise GuardError("%s/%s arrays differ from the frozen subset" % (model_id, unit_id(attack, eps)))
            csv_rows += [
                [model_id, attack, eps_str(eps), int(i), int(y), int(p)] for i, y, p in zip(a_ids, a_y, a_pred)
            ]
    atomic_write_bytes(pred_path, _csv_bytes(CSV_COLUMNS["predictions_subset"], csv_rows))
    atomic_write_json(
        results_path,
        {
            "schema": 1,
            "kind": "m2_deadline_robustness_results",
            "run_id": RUN_ID,
            "n": int(len(subset_ids)),
            "per_class_target": ctx.per_class,
            "subset_ids_sha256": subset_hash,
            "epsilons": [float(e) for e in rob_proto["epsilons"]],
            "units": units,
            "eps0_consistency": eps0,
        },
    )

    def validate() -> bool:
        doc = read_json(results_path)
        if int(doc["n"]) != len(subset_ids) or doc["subset_ids_sha256"] != subset_hash:
            raise GuardError("robustness_results.json differs from the frozen subset")
        for m in MODEL_IDS:
            if sorted(doc["units"][m]) != sorted(unit_id(a, e) for a, e in ATTACK_UNITS):
                raise GuardError("robustness_results.json lacks units for %s" % m)
            if any(k not in doc["eps0_consistency"].get(m, {}) for k in ("eps0_vs_full_test_mismatches", "eps0_max_abs_logit_diff", "n_compared")):
                raise GuardError("eps0 consistency record missing for %s" % m)
        n_rows = len(_read_csv_rows(pred_path, CSV_COLUMNS["predictions_subset"]))
        if n_rows != len(MODEL_IDS) * len(ATTACK_UNITS) * len(subset_ids):
            raise GuardError("predictions_subset.csv has %d rows" % n_rows)
        return True

    return _finish([results_path, pred_path] + unit_files, validate)


def _write_unit_done(
    ctx: RunContext,
    udir: Path,
    model_id: str,
    attack: str,
    eps: float,
    source: str,
    result: Dict[str, Any],
    subset_hash: str,
    full_ids: np.ndarray,
    full_pred: np.ndarray,
) -> None:
    arrays = result["arrays"]
    missing = [k for k in UNIT_ARRAY_KEYS if k not in arrays]
    if missing:
        raise GuardError("evaluate_unit returned no arrays %s" % missing)
    y = np.asarray(arrays["y"], dtype=np.int64)
    pred_adv = np.asarray(arrays["pred_adv"], dtype=np.int64)
    atomic_write_bytes(udir / "arrays.npz", _npz_bytes(**{k: np.asarray(arrays[k]) for k in UNIT_ARRAY_KEYS}))
    row = result["row"]
    doc: Dict[str, Any] = {
        "model_id": model_id,
        "attack": attack,
        "epsilon": float(eps),
        "unit_id": unit_id(attack, eps),
        "source": source,
        "ids_sha256": result["checks"]["example_id_sha256"],
        "valid": True,
        "n": int(len(y)),
        "accuracy": metrics.accuracy(y, pred_adv),
        "macro_f1": metrics.macro_f1(y, pred_adv, labels=range(N_CLASSES), zero_division=0),
        "robust_accuracy": float(row["robust_accuracy"]),
        "attack_seconds": float(row["attack_seconds"]),
    }
    if attack == CLEAN_ATTACK:
        # record-only (pre-registered): never fails the unit or the run
        doc["eps0_consistency"] = metrics.eps0_consistency(arrays["ids"], pred_adv, full_ids, full_pred)
    files = {name: sha256_file(udir / name) for name in ("arrays.npz", "attack_checks.json", "attack_result.json")}
    doc["files"] = files
    if doc["ids_sha256"] != subset_hash:
        raise GuardError("unit ids hash differs from the frozen subset hash")
    atomic_write_json(udir / "unit_done.json", doc)


# ---------------------------------------------------------------- stage 8: efficiency
def stage_efficiency(ctx: RunContext) -> Dict[str, Any]:
    out = ctx.out
    deps = ctx.deps
    protocol.assert_frozen(out)
    proto = _protocol_doc(out)
    eff_dir = out_path(out, "efficiency_dir")
    eff_path = out_path(out, "efficiency")
    reps = int(proto["efficiency"]["repetitions"])

    def valid_record(path: Path) -> bool:
        try:
            rec = read_json(path)
        except (OSError, ValueError):
            return False
        if rec.get("any_failed") is not False or rec.get("interleaved") is not True:
            return False
        for m in MODEL_IDS:
            if len([e for e in rec.get("measurements", []) if e.get("variant") == m]) != reps:
                return False
        return True

    if eff_path.is_file() and valid_record(eff_path):
        ctx.log.info("efficiency: a valid efficiency.json exists; not measuring again")
    else:
        tag = utc_stamp()
        for name in ("efficiency.json", "fixed_rows.json", "fixed_rows.npy", "specs", "results"):
            p = eff_dir / name
            if p.exists():
                _quarantine(p, eff_dir / "_incomplete", tag)
        eff_cfg = ctx.eff_cfg()
        val = ctx.trainval()["validation"]
        me = deps.me
        rows = me.select_rows(val["ids"], val["x"], int(eff_cfg["fixed_inputs"]["count"]), ctx.split_seed)
        specs: Dict[str, Any] = {}
        conversions: Dict[str, Any] = {}
        for model_id in MODEL_IDS:
            markers.verify_deadline_meta(out, model_id, config_sha256=ctx.config_sha256, epochs=ctx.epochs)
            if model_id in INT8_MODELS:
                specs[model_id] = ctx.int8_spec(model_id)
                conversions[model_id] = ctx.conversion(model_id)
            else:
                specs[model_id] = {"kind": "fp32", "checkpoint": checkpoint_path(out, model_id), "architecture": ctx.architecture()}
        total = deps.parameter_count(ctx.load_model("V1"))
        deps.configure_inference(threads=int(ctx.config["efficiency"]["threads"]))
        protocol.assert_frozen(out)
        me.run_efficiency(
            specs, rows, eff_cfg, eff_dir, pilot=False, conversions=conversions, total_parameters=total,
            banner="DEADLINE-V1", threads=int(ctx.config["efficiency"]["threads"]),
        )

    def validate() -> bool:
        if not valid_record(eff_path):
            raise GateFailedError("efficiency.json is missing, has a failed measurement or lacks %d runs per model" % reps)
        return True

    return _finish([eff_path, eff_dir / "fixed_rows.json"], validate)


# ---------------------------------------------------------------- stage 9: report
REPORT_OUTPUT_KEYS = (
    "model_comparison_csv", "model_comparison_md", "robustness_curves_csv", "pp_differences",
    "report_md", "fgsm_figure", "pgd5_figure",
)


def stage_report(ctx: RunContext) -> Dict[str, Any]:
    from . import report  # lazy: matplotlib is imported inside report only when drawing

    out = ctx.out
    report.generate(out, ctx.v4_root, ctx.config_path)
    rc, text = ctx.deps.run_verify_overlay(ctx.overlay_root, ctx.v4_root)
    if rc != 0:
        raise GuardError("overlay/V4 verification failed after the run (exit %s): %s" % (rc, text.strip()[-600:]))

    def validate() -> bool:
        for key in REPORT_OUTPUT_KEYS:
            p = out_path(out, key)
            if not p.is_file() or p.stat().st_size == 0:
                raise GuardError("report output missing or empty: %s" % p)
        read_json(out_path(out, "pp_differences"))
        _read_csv_rows(out_path(out, "robustness_curves_csv"), CSV_COLUMNS["robustness_curves_csv"])
        _read_csv_rows(out_path(out, "model_comparison_csv"), CSV_COLUMNS["model_comparison_csv"])
        return True

    return _finish([out_path(out, k) for k in REPORT_OUTPUT_KEYS], validate)


# ---------------------------------------------------------------- the single marker call site
STAGE_FUNCS: Dict[str, Callable[[RunContext], Dict[str, Any]]] = {
    "preflight": stage_preflight,
    "freeze": stage_freeze,
    "train_ordinary": stage_train_ordinary,
    "train_adversarial": stage_train_adversarial,
    "ptq": stage_ptq,
    "val_check": stage_val_check,
    "test_clean": stage_test_clean,
    "robustness": stage_robustness,
    "efficiency": stage_efficiency,
    "report": stage_report,
}
assert tuple(STAGE_FUNCS) == STAGES


def run_stage(ctx: RunContext, stage: str) -> Dict[str, Any]:
    """Run one stage and write its marker through ``markers.write_after_validate`` (the only call site)."""
    if stage not in STAGE_FUNCS:
        raise GuardError("unknown stage %r" % stage)
    markers.assert_not_smoke(ctx.out)
    markers.clear(ctx.out, stage)
    ctx.current_stage = stage
    started = utc_now()
    t0 = time.monotonic()
    ctx.log.info("stage %s: start", stage)
    work = STAGE_FUNCS[stage](ctx)
    seconds = time.monotonic() - t0
    marker = markers.write_after_validate(
        ctx.out, stage, work["validate"], work["outputs"], ctx.marker_ctx(), started_utc=started, seconds=seconds
    )
    ctx.log.info("stage %s: marker written after %.1f s", stage, seconds)
    ctx.current_stage = None
    return marker
