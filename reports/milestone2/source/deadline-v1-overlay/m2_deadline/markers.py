"""Stage markers, per-model ``deadline_meta.json`` and ``run_identity.json`` (DESIGN 6.3, 6.6).

Pure module: standard library only. Python 3.9 compatible.

``write_after_validate`` is the ONLY function that creates a stage marker. A marker is written last,
after ``validate_fn`` succeeded, and records the SHA-256 of every output file it vouches for.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .common import (
    ADVERSARIAL,
    GuardError,
    KIND,
    PATHS,
    PROTOCOL_STAGES,
    RUN_ID,
    STAGES,
    atomic_write_json,
    checkpoint_path,
    deadline_meta_path,
    marker_path,
    out_path,
    read_json,
    sha256_file,
    utc_now,
)

SOURCE_FILES = (
    "m2_training.py",
    "m2_quant.py",
    "m2_attacks.py",
    "m2_efficiency.py",
    "quantization.py",
    "m2_split.py",
    "m2b_exec.py",
)


# ---------------------------------------------------------------- smoke path guard
def is_smoke_path(path: Any) -> bool:
    """True if any path component equals ``smoke`` or ends with ``-smoke`` (``/home/smokey`` is fine)."""
    parts = Path(str(path)).resolve().parts
    return any(p.lower() == "smoke" or p.lower().endswith("-smoke") for p in parts)


def assert_not_smoke(path: Any) -> None:
    if is_smoke_path(path):
        raise GuardError("refusing smoke path: %s" % path)


# ---------------------------------------------------------------- markers
def context(
    run_id: str = RUN_ID,
    *,
    config_sha256: str,
    lock_sha256: str,
    protocol_sha256: Optional[str],
    epochs: int,
) -> Dict[str, Any]:
    """Values every marker must agree with."""
    return {
        "run_id": run_id,
        "config_sha256": config_sha256,
        "lock_sha256": lock_sha256,
        "protocol_sha256": protocol_sha256,
        "epochs": int(epochs),
    }


def _relpath(out: Path, p: Any) -> str:
    p = Path(str(p))
    if not p.is_absolute():
        p = out / p
    return p.resolve().relative_to(out.resolve()).as_posix()


def clear(out: Any, stage: str) -> bool:
    """Delete a stage marker (done before a stage starts). Returns True if one existed."""
    path = marker_path(out, stage)
    if path.exists():
        path.unlink()
        return True
    return False


def write_after_validate(
    out: Any,
    stage: str,
    validate_fn: Callable[[], Any],
    outputs: Iterable[Any],
    ctx: Dict[str, Any],
    started_utc: Optional[str] = None,
    seconds: float = 0.0,
) -> Dict[str, Any]:
    """Validate, then write the marker. The sole marker writer.

    ``validate_fn`` signals failure by raising or returning ``False``. On any failure no marker exists afterwards
    (a stale one is removed first). Every listed output must exist; its SHA-256 is recorded.
    """
    if stage not in STAGES:
        raise GuardError("unknown stage %r" % stage)
    out = Path(str(out))
    clear(out, stage)
    result = validate_fn()
    if result is False:  # None counts as success: validators signal failure by raising or returning False
        raise GuardError("validation of stage %s failed" % stage)
    hashes: Dict[str, str] = {}
    for p in outputs:
        full = Path(str(p)) if Path(str(p)).is_absolute() else out / str(p)
        if not full.is_file():
            raise GuardError("stage %s output missing: %s" % (stage, full))
        hashes[_relpath(out, full)] = sha256_file(full)
    if stage in PROTOCOL_STAGES and ctx.get("protocol_sha256") is None:
        raise GuardError("stage %s requires a protocol hash" % stage)
    marker = {
        "stage": stage,
        "run_id": ctx["run_id"],
        "config_sha256": ctx["config_sha256"],
        "lock_sha256": ctx["lock_sha256"],
        "protocol_sha256": ctx.get("protocol_sha256"),
        "epochs": int(ctx["epochs"]),
        "started_utc": started_utc or utc_now(),
        "finished_utc": utc_now(),
        "seconds": float(seconds),
        "outputs": hashes,
    }
    atomic_write_json(marker_path(out, stage), marker)
    return marker


def verify(out: Any, stage: str, ctx: Dict[str, Any]) -> Tuple[bool, str]:
    """``(ok, reason)``. A marker is valid only if its context matches and all outputs still hash equal."""
    out = Path(str(out))
    path = marker_path(out, stage)
    if not path.is_file():
        return False, "no marker"
    try:
        m = read_json(path)
    except ValueError:
        return False, "marker unreadable"
    if m.get("stage") != stage:
        return False, "marker names another stage"
    for key in ("run_id", "config_sha256", "lock_sha256", "epochs"):
        if m.get(key) != ctx.get(key):
            return False, "%s differs from the current run" % key
    if stage in PROTOCOL_STAGES:
        if m.get("protocol_sha256") is None or m.get("protocol_sha256") != ctx.get("protocol_sha256"):
            return False, "protocol_sha256 differs from the current protocol"
    elif m.get("protocol_sha256") not in (None, ctx.get("protocol_sha256")):
        return False, "protocol_sha256 differs from the current protocol"
    for rel, expected in sorted((m.get("outputs") or {}).items()):
        target = out / rel
        if not target.is_file():
            return False, "output missing: %s" % rel
        if sha256_file(target) != expected:
            return False, "output changed: %s" % rel
    return True, "ok"


def valid_stages(out: Any, ctx: Dict[str, Any], stages: Sequence[str] = STAGES) -> List[str]:
    return [s for s in stages if verify(out, s, ctx)[0]]


def first_invalid_stage(out: Any, ctx: Dict[str, Any], stages: Sequence[str] = STAGES) -> Optional[str]:
    """Resume plan: the first stage whose marker is missing or invalid, or ``None`` if all are valid."""
    for s in stages:
        if not verify(out, s, ctx)[0]:
            return s
    return None


def any_marker_exists(out: Any) -> bool:
    d = Path(str(out)) / PATHS["markers_dir"]
    return d.is_dir() and any(d.glob("*.ok"))


# ---------------------------------------------------------------- per-model deadline_meta.json
def write_deadline_meta(
    out: Any,
    model_id: str,
    *,
    config_sha256: str,
    epochs: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ckpt = checkpoint_path(out, model_id)
    if not ckpt.is_file():
        raise GuardError("cannot write deadline_meta: %s missing" % ckpt)
    meta = {
        "run_id": RUN_ID,
        "kind": KIND,
        "config_sha256": config_sha256,
        "model_id": model_id,
        "epochs": int(epochs),
        "adversarial": bool(ADVERSARIAL[model_id]),
        "checkpoint_file": ckpt.name,
        "checkpoint_sha256": sha256_file(ckpt),
        "created_utc": utc_now(),
    }
    if extra:
        meta["extra"] = dict(extra)
    atomic_write_json(deadline_meta_path(out, model_id), meta)
    return meta


def verify_deadline_meta(out: Any, model_id: str, *, config_sha256: str, epochs: int) -> Dict[str, Any]:
    """Raise ``GuardError`` unless the model files belong to this run (smoke or other-run files fail)."""
    path = deadline_meta_path(out, model_id)
    if not path.is_file():
        raise GuardError("deadline_meta.json missing for %s" % model_id)
    meta = read_json(path)
    expected = {
        "run_id": RUN_ID,
        "kind": KIND,
        "config_sha256": config_sha256,
        "model_id": model_id,
        "epochs": int(epochs),
        "adversarial": bool(ADVERSARIAL[model_id]),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise GuardError("deadline_meta %s: %s is %r, expected %r" % (model_id, key, meta.get(key), value))
    ckpt = checkpoint_path(out, model_id)
    if not ckpt.is_file() or sha256_file(ckpt) != meta.get("checkpoint_sha256"):
        raise GuardError("checkpoint of %s does not match deadline_meta.json" % model_id)
    return meta


# ---------------------------------------------------------------- run_identity.json
def source_hashes(v4_root: Any) -> Dict[str, str]:
    """SHA-256 of the V4 ``research`` modules this overlay calls (read-only)."""
    base = Path(str(v4_root)) / "research"
    return {name: sha256_file(base / name) for name in SOURCE_FILES}


def build_run_identity(
    *,
    config_sha256: str,
    lock_sha256: str,
    epochs: int,
    protocol_sha256: Optional[str],
    v4_manifest_sha256: str,
    source_sha256: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "kind": KIND,
        "config_sha256": config_sha256,
        "lock_sha256": lock_sha256,
        "epochs": int(epochs),
        "protocol_sha256": protocol_sha256,
        "v4_manifest_sha256": v4_manifest_sha256,
        "source_sha256": dict(source_sha256),
        "created_utc": utc_now(),
    }


def write_run_identity(out: Any, identity: Dict[str, Any]) -> bool:
    """Write once. An existing file with the same content (ignoring ``created_utc``) is kept.

    Returns True if written. A differing existing file raises ``GuardError``.
    """
    path = out_path(out, "run_identity")
    if path.exists():
        old = read_json(path)
        a = {k: v for k, v in old.items() if k != "created_utc"}
        b = {k: v for k, v in identity.items() if k != "created_utc"}
        if a != b:
            raise GuardError("run_identity.json exists with different content; not overwriting")
        return False
    atomic_write_json(path, identity)
    return True


def verify_run_identity(
    out: Any,
    *,
    config_sha256: str,
    lock_sha256: str,
    epochs: int,
    v4_root: Optional[Any] = None,
    v4_manifest_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Raise ``GuardError`` (exit 3) if the run on disk is not the run the caller means to continue."""
    path = out_path(out, "run_identity")
    if not path.is_file():
        raise GuardError("run_identity.json missing")
    ident = read_json(path)
    for key, value in (
        ("run_id", RUN_ID),
        ("kind", KIND),
        ("config_sha256", config_sha256),
        ("lock_sha256", lock_sha256),
        ("epochs", int(epochs)),
    ):
        if ident.get(key) != value:
            raise GuardError("run_identity.%s is %r, expected %r" % (key, ident.get(key), value))
    if v4_manifest_sha256 is not None and ident.get("v4_manifest_sha256") != v4_manifest_sha256:
        raise GuardError("V4 MANIFEST.sha256 changed since the run started")
    if v4_root is not None:
        now = source_hashes(v4_root)
        changed = sorted(k for k, v in now.items() if ident.get("source_sha256", {}).get(k) != v)
        if changed:
            raise GuardError("V4 source files changed since the run started: %s" % changed)
    return ident
