"""Protected-path integrity snapshot for the Milestone 2B preparation (stdlib only).

--write PATH                 record {relpath: sha256} for every protected file (never overwrites).
--compare BEFORE [--write AFTER]
                             re-hash, compare with BEFORE, exit 1 on any missing, changed or added
                             protected file; the snapshot is saved to AFTER when given.
--output PATH (repeatable)   also fail if PATH (a path created by this work) matches a protected glob.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROTECTED_GLOBS = (
    "reports/milestone2a",
    "configs/m2",
    "artifacts/m2a-pilot",
    "artifacts/m2a-integrity",
    "docs/MILESTONE2*",
    "reports/milestone1-*",
    "configs/milestone1_*.json",
)
SKIP_DIRS = {"__pycache__", ".git"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_protected(rel):
    """True when the repo-relative posix path ``rel`` lies at or under any protected glob."""
    rel = rel.replace("\\", "/").strip("/")
    for pattern in PROTECTED_GLOBS:
        parts = pattern.split("/")
        head = "/".join(rel.split("/")[:len(parts)])
        if fnmatch.fnmatchcase(head, pattern):
            return True
    return False


def _files(root=ROOT):
    root = Path(root)
    found = {}
    for pattern in PROTECTED_GLOBS:
        for top in sorted(root.glob(pattern)):
            if top.is_file():
                found[top.relative_to(root).as_posix()] = top
                continue
            for base, dirs, names in os.walk(top):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for name in sorted(names):
                    p = Path(base) / name
                    found[p.relative_to(root).as_posix()] = p
    return found


def snapshot(root=ROOT):
    files = _files(root)
    return {"schema": 1, "protected_globs": list(PROTECTED_GLOBS), "file_count": len(files),
            "files": {rel: _sha256(p) for rel, p in sorted(files.items())}}


def compare(before, after):
    old, new = before["files"], after["files"]
    return {"added": sorted(set(new) - set(old)),
            "missing": sorted(set(old) - set(new)),
            "changed": sorted(p for p in set(old) & set(new) if old[p] != new[p])}


def protected_outputs(before, after, outputs=()):
    """Paths created by this work (after minus before, plus declared outputs) that are protected."""
    created = set(after["files"]) - set(before["files"]) | {str(o) for o in outputs}
    return sorted(p for p in created if is_protected(p))


def _write(path, value):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, indent=2)
            f.write("\n")
        os.link(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", help="snapshot path to write (never overwritten)")
    parser.add_argument("--compare", help="earlier snapshot to compare against")
    parser.add_argument("--output", action="append", default=[], help="path created by this work")
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
        offending = protected_outputs(before, current, args.output)
        current["compared_with"] = str(args.compare)
        current["diff"] = diff
        current["protected_outputs"] = offending
        if any(diff.values()) or offending:
            status = 1
            print("PROTECTED PATHS CHANGED:", json.dumps({**diff, "protected_outputs": offending}, indent=2))
        else:
            print(f"OK: {current['file_count']} protected files unchanged")
    if args.write:
        _write(args.write, current)
        print(f"wrote {args.write} ({current['file_count']} protected files)")
    return status


if __name__ == "__main__":
    sys.exit(main())
