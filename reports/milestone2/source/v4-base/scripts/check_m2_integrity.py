"""Protected-path integrity snapshot for Milestone 2A.

--write PATH            hash every protected Milestone 1 path and save PATH
                        (refuses to overwrite an existing file).
--compare BEFORE        re-hash and compare with BEFORE; exit 1 on any change
        [--write AFTER] inside a protected path, optionally saving the new snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

# Directories, files and glob patterns relative to the repository root.
PROTECTED_GLOBS = (
    "reports/milestone1-*",
    "configs/milestone1_*.json",
    "configs/smoke.json",
    "configs/pilot.json",
    "configs/full.json",
    "runs",
    "artifacts/smoke-verified",
    "artifacts/pilot-verified",
    "artifacts/pilot-estimate",
    "docs/MILESTONE1_*",
    "docs/VERIFIED_RESULTS.md",
    "docs/PROTOCOL.md",
    "upstream",
)
SKIP_DIRS = {"__pycache__", ".git"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _files():
    found = {}
    for pattern in PROTECTED_GLOBS:
        for top in sorted(ROOT.glob(pattern)):
            if top.is_file():
                found[top.relative_to(ROOT).as_posix()] = top
                continue
            for base, dirs, names in os.walk(top):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for name in sorted(names):
                    p = Path(base) / name
                    found[p.relative_to(ROOT).as_posix()] = p
    return found


def snapshot():
    files = _files()
    return {
        "schema": 1,
        "head": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain").splitlines(),
        "protected_globs": list(PROTECTED_GLOBS),
        "file_count": len(files),
        "files": {rel: _sha256(p) for rel, p in sorted(files.items())},
    }


def _write(path, value, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)
            f.write("\n")
        if not overwrite:
            # os.link fails when another writer created the target meanwhile.
            os.link(tmp, path)
            os.unlink(tmp)
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def compare(before, after):
    old, new = before["files"], after["files"]
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "modified": sorted(p for p in set(old) & set(new) if old[p] != new[p]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", help="path of the snapshot to write (never overwritten)")
    parser.add_argument("--compare", help="earlier snapshot to compare against")
    args = parser.parse_args(argv)
    if not args.write and not args.compare:
        parser.error("give --write and/or --compare")
    if args.write and Path(args.write).exists():
        raise SystemExit(f"Refusing to overwrite {args.write}")
    current = snapshot()
    status = 0
    if args.compare:
        before = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        diff = compare(before, current)
        current["compared_with"] = str(args.compare)
        current["diff"] = diff
        current["head_unchanged"] = before["head"] == current["head"]
        if any(diff.values()):
            status = 1
            print("PROTECTED PATHS CHANGED:", json.dumps(diff, indent=2))
        else:
            print(f"OK: {current['file_count']} protected files unchanged")
        if not current["head_unchanged"]:
            print("note: git HEAD differs from the before snapshot")
    if args.write:
        _write(args.write, current)
        print(f"wrote {args.write} ({current['file_count']} protected files)")
    return status


if __name__ == "__main__":
    sys.exit(main())
