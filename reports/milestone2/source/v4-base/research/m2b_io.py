"""Durable file I/O for the M2B campaign: atomic writes, unit locks, rename-only archiving.

Every write goes to a temp file in the target directory, is flushed and fsynced, then os.replace()d; the directory
is fsynced afterwards where the platform allows it (best effort on Windows). Nothing here ever deletes evidence:
archive_move renames. This module is part of the training source closure because train_unit writes checkpoints with it.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

CREATE_TIME_TOLERANCE_SECONDS = 1.0


class LockHeldError(RuntimeError):
    """Another live process holds the lock."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fsync_dir(path) -> None:
    """fsync a directory so a rename is durable. Best effort: Windows cannot open directories this way."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _tmp_name(path: Path) -> Path:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    return Path(tmp)


def _durable(path, writer) -> Path:
    """writer(file_object_opened_wb) writes the content; the result replaces `path` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(path)
    try:
        with open(tmp, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    fsync_dir(path.parent)
    return path


def durable_write_bytes(path, data: bytes) -> Path:
    return _durable(path, lambda stream: stream.write(data))


def durable_write_json(path, value, indent=2) -> Path:
    text = json.dumps(value, indent=indent, allow_nan=False, sort_keys=False) + "\n"
    return durable_write_bytes(path, text.encode("utf-8"))


def durable_write_npz(path, **arrays) -> Path:
    return _durable(path, lambda stream: np.savez(stream, **arrays))


def durable_write_torch(path, value) -> Path:
    return _durable(path, lambda stream: torch.save(value, stream))


# Short aliases used by the rest of the campaign code.
durable_json = durable_write_json
durable_npz = durable_write_npz
durable_torch = durable_write_torch


def _pid_start(pid: int):
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OSError):
        return None


def _lock_is_live(info: dict) -> bool:
    """Live only if same host, the pid exists and its create time equals the stored one (covers pid reuse)."""
    if info.get("host") != socket.gethostname():
        return True  # cannot prove a foreign lock dead: treat as live
    started = _pid_start(int(info.get("pid", -1)))
    if started is None:
        return False
    return abs(started - float(info.get("create_time", -1e9))) <= CREATE_TIME_TOLERANCE_SECONDS


@contextlib.contextmanager
def unit_lock(path):
    """Exclusive lock file (O_CREAT|O_EXCL) holding pid, host, start time and process create time.

    A lock whose owner is dead (or whose pid was reused by another process) is stale: it is renamed aside and retaken.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"pid": os.getpid(), "host": socket.gethostname(), "start": utc_now(), "create_time": _pid_start(os.getpid())}
    for _attempt in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                held = {}
            if held and _lock_is_live(held):
                raise LockHeldError(f"{path} is held by pid {held.get('pid')} on {held.get('host')}")
            stale = path.with_name(path.name + f".stale-{utc_now()}-{os.getpid()}")
            os.replace(path, stale)  # rename, never delete
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(info, stream)
            stream.flush()
            os.fsync(stream.fileno())
        break
    else:  # pragma: no cover - two consecutive races
        raise LockHeldError(f"could not take {path}")
    try:
        yield info
    finally:
        with contextlib.suppress(OSError):
            path.unlink()


def archive_move(path, archive_root, tag: str) -> Path:
    """Move `path` (file or directory) into archive_root/<tag>/ by rename. Never deletes. Returns the new location."""
    path, archive_root = Path(path), Path(archive_root)
    destination_dir = archive_root / tag
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / path.name
    counter = 1
    while destination.exists():
        destination = destination_dir / f"{path.name}.{counter}"
        counter += 1
    os.replace(path, destination)
    fsync_dir(destination_dir)
    return destination


def stamp_tag(prefix: str) -> str:
    return f"{prefix}-{utc_now()}"


def monotonic() -> float:
    return time.perf_counter()
