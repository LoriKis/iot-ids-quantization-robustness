"""Overlay manifest helpers. Standard library only (runs under the system python3 before any venv exists).

Manifest line format, identical to V4's: ``<sha256>  <posix relpath>`` sorted, LF only. Nothing here writes
under the V4 folder; V4 is only read.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_GUARD = 3

OVERLAY_MANIFEST = "OVERLAY_MANIFEST.sha256"
OVERLAY_IDENTITY = "OVERLAY_IDENTITY.json"
CONFIG_REL = "configs/m2_deadline_v1.json"
V4_MANIFEST = "MANIFEST.sha256"
V4_VERIFY_SCRIPT = "scripts/vm/verify_package.py"
IDENTITY_KIND = "m2-deadline-overlay"

SKIP_TOP_OVERLAY = ("artifacts",)
CACHE_DIRS = ("__pycache__", ".pytest_cache")
V4_SKIP_TOP = ("output", ".venv")
TEXT_SUFFIXES = (".py", ".sh", ".json", ".md", ".txt", ".ini", ".cfg", ".csv", ".yaml", ".yml", ".sha256")
_HEX = set("0123456789abcdef")


def sha256_file(path: Any) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_link(path: Any) -> bool:
    if os.path.islink(str(path)):
        return True
    attrs = getattr(os.lstat(str(path)), "st_file_attributes", 0)
    return bool(attrs & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT (Windows junctions)


def walk_overlay(root: Any) -> Tuple[List[str], List[str], List[str]]:
    """Return (regular files, symlinks, stray cache dirs) as sorted posix paths relative to ``root``.

    Skipped: top-level ``artifacts/``, ``OVERLAY_MANIFEST.sha256`` and ``*.pyc``. Cache directories are
    reported separately and not descended into.
    """
    root = Path(str(root))
    files: List[str] = []
    links: List[str] = []
    strays: List[str] = []
    for dirpath, dirnames, filenames in os.walk(str(root), followlinks=False):
        here = Path(dirpath)
        keep = []
        for name in dirnames:
            full = here / name
            rel = full.relative_to(root)
            if len(rel.parts) == 1 and name in SKIP_TOP_OVERLAY:
                continue
            if name in CACHE_DIRS:
                strays.append(rel.as_posix())
                continue
            if is_link(full):
                links.append(rel.as_posix())
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in filenames:
            full = here / name
            rel = full.relative_to(root).as_posix()
            if name.endswith(".pyc") or rel == OVERLAY_MANIFEST:
                continue
            if is_link(full):
                links.append(rel)
            else:
                files.append(rel)
    return sorted(files), sorted(links), sorted(strays)


def crlf_files(root: Any, files: Sequence[str]) -> List[str]:
    root = Path(str(root))
    bad = []
    for rel in files:
        if rel.endswith(TEXT_SUFFIXES) and b"\r" in (root / rel).read_bytes():
            bad.append(rel)
    return sorted(bad)


def manifest_text(entries: Dict[str, str]) -> str:
    return "".join("%s  %s\n" % (entries[rel], rel) for rel in sorted(entries))


def parse_manifest(data: bytes) -> Dict[str, str]:
    """Strict parser; raises ValueError on CR, malformed lines or duplicates."""
    if b"\r" in data:
        raise ValueError("manifest contains CR (LF only is allowed)")
    entries: Dict[str, str] = {}
    for number, line in enumerate(data.decode("utf-8").split("\n"), 1):
        if not line:
            continue
        digest, sep, rel = line[:64], line[64:66], line[66:]
        if len(line) < 67 or sep != "  " or any(c not in _HEX for c in digest):
            raise ValueError("manifest line %d is malformed" % number)
        if rel in entries:
            raise ValueError("manifest line %d repeats %s" % (number, rel))
        entries[rel] = digest
    return entries


def check_overlay(root: Any, entries: Dict[str, str]) -> Dict[str, List[str]]:
    """Part 1: every listed file exists and hashes equal; nothing unlisted; no symlinks; no CR."""
    root = Path(str(root))
    files, links, _strays = walk_overlay(root)
    present = set(files)
    missing = sorted(set(entries) - present)
    unlisted = sorted(present - set(entries))
    mismatched = sorted(rel for rel in entries if rel in present and sha256_file(root / rel) != entries[rel])
    return {"missing": missing, "mismatched": mismatched, "unlisted": unlisted, "symlinks": links,
            "crlf": crlf_files(root, files)}


def walk_v4(root: Any) -> Tuple[List[str], List[str]]:
    """Regular files and links under V4, skipping top-level ``output/`` and ``.venv/`` and its manifest."""
    root = Path(str(root))
    files: List[str] = []
    links: List[str] = []
    for dirpath, dirnames, filenames in os.walk(str(root), followlinks=False):
        here = Path(dirpath)
        keep = []
        for name in dirnames:
            full = here / name
            rel = full.relative_to(root)
            if len(rel.parts) == 1 and name in V4_SKIP_TOP:
                continue
            if is_link(full):
                links.append(rel.as_posix())
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in filenames:
            full = here / name
            rel = full.relative_to(root).as_posix()
            if is_link(full):
                links.append(rel)
            elif rel != V4_MANIFEST:
                files.append(rel)
    return sorted(files), sorted(links)


def check_v4(v4_root: Any) -> Dict[str, List[str]]:
    """Part 3 (stdlib, read-only): V4's own manifest against V4's files."""
    v4_root = Path(str(v4_root))
    entries = parse_manifest((v4_root / V4_MANIFEST).read_bytes())
    files, links = walk_v4(v4_root)
    present = set(files)
    return {
        "missing": sorted(set(entries) - present),
        "mismatched": sorted(rel for rel in entries if rel in present and sha256_file(v4_root / rel) != entries[rel]),
        "unlisted": sorted(present - set(entries)),
        "symlinks": links,
    }


_PROBLEM_LINE = re.compile(r"^\s+(missing|mismatched|unlisted|symlinks): (.+)$")


def second_check(v4_root: Any, runner: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Independent check through V4's own ``verify_package.py --manifest-only``.

    Returns ``{"status": "ok"|"warn"|"fail"|"skipped", "unlisted": [...], "problems": {...}, "detail": str}``.
    """
    v4_root = Path(str(v4_root))
    script = v4_root / V4_VERIFY_SCRIPT
    run = runner or subprocess.run
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, "-B", str(script), "--root", str(v4_root), "--manifest-only"]
    try:
        if runner is None and not script.is_file():
            raise FileNotFoundError(str(script))
        done = run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    except (FileNotFoundError, ImportError) as exc:
        return {"status": "skipped", "unlisted": [], "problems": {}, "detail": "%s: %s" % (type(exc).__name__, exc)}
    out = (done.stdout or "")
    err = (done.stderr or "")
    if done.returncode == 0:
        return {"status": "ok", "unlisted": [], "problems": {}, "detail": out.strip()}
    if "ModuleNotFoundError" in err or "ImportError" in err:
        return {"status": "skipped", "unlisted": [], "problems": {}, "detail": "import error in the child process"}
    problems: Dict[str, List[str]] = {}
    for line in out.splitlines():
        m = _PROBLEM_LINE.match(line)
        if m:
            problems.setdefault(m.group(1), []).append(m.group(2).strip())
    if problems and set(problems) == {"unlisted"}:
        return {"status": "warn", "unlisted": problems["unlisted"], "problems": problems, "detail": out.strip()}
    return {"status": "fail", "unlisted": problems.get("unlisted", []), "problems": problems,
            "detail": (out.strip() + "\n" + err.strip()).strip()[-800:]}
