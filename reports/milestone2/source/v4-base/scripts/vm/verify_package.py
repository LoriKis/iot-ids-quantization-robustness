#!/usr/bin/env python3
"""Verify the transferred M2 VM package. Stdlib only, so it runs before the virtual environment exists.

    python3 scripts/vm/verify_package.py                  # all checks
    python3 scripts/vm/verify_package.py --manifest-only  # file set and hashes only (used again after the tests)

Checks: manifest hashes and file set (missing / mismatched / unlisted), no symlinks, LF line endings of VM-authored
files, dataset SHA-256 and size against configs/m2/data.json, the 19 recorded PAVE source hashes, the PAVE tree id
against PACKAGE_IDENTITY.json, and the runbook fingerprint table (only when the project imports are available).
Exit 0 when everything passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

MANIFEST = "MANIFEST.sha256"
IDENTITY = "PACKAGE_IDENTITY.json"
SKIP_TOP = ("output", ".venv")
DATASET_REL = "upstream/PAVE_artifact/data/balanced_CICIoT.csv"
DATASET_SIZE = 5444531
LF_FILES = ("VM_RUNBOOK.md", "requirements-linux-cpu.txt", MANIFEST, IDENTITY)
PAVE_PIN = "e73911bbdb241f3fb6c5de346139010c4aadefd4"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def walk_files(root: Path):
    """(regular files as posix paths, symlinks and junctions) under root, skipping output/ and .venv/ at the top."""
    files, links = [], []
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        keep = []
        for name in dirnames:
            full = here / name
            rel = full.relative_to(root)
            if len(rel.parts) == 1 and name in SKIP_TOP:
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
            elif rel != MANIFEST:
                files.append(rel)
    return sorted(files), sorted(links)


def is_link(path: Path) -> bool:
    if os.path.islink(path):
        return True
    attrs = getattr(os.lstat(path), "st_file_attributes", 0)
    return bool(attrs & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT (Windows junctions)


def read_manifest(root: Path) -> dict:
    entries = {}
    text = (Path(root) / MANIFEST).read_bytes().decode("utf-8")
    for number, line in enumerate(text.split("\n"), 1):
        if not line:
            continue
        digest, sep, rel = line[:64], line[64:66], line[66:]
        if len(line) < 67 or sep != "  " or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{MANIFEST} line {number} is malformed")
        entries[rel] = digest
    return entries


def check_manifest(root: Path) -> dict:
    root = Path(root)
    entries = read_manifest(root)
    files, links = walk_files(root)
    present = set(files)
    missing = sorted(set(entries) - present)
    unlisted = sorted(present - set(entries))
    mismatched = sorted(rel for rel in entries if rel in present and sha256_file(root / rel) != entries[rel])
    return {"listed": len(entries), "present": len(files), "missing": missing, "mismatched": mismatched,
            "unlisted": unlisted, "symlinks": links}


def check_line_endings(root: Path) -> list:
    root = Path(root)
    targets = [root / name for name in LF_FILES if (root / name).is_file()]
    targets += sorted((root / "scripts" / "vm").glob("*.sh"))
    return [p.relative_to(root).as_posix() for p in targets if b"\r" in p.read_bytes()]


def check_dataset(root: Path) -> list:
    root = Path(root)
    problems = []
    recorded = json.loads((root / "configs" / "m2" / "data.json").read_text(encoding="utf-8"))["dataset"]["sha256"]
    path = root / DATASET_REL
    if not path.is_file():
        return [f"dataset missing: {DATASET_REL}"]
    if path.stat().st_size != DATASET_SIZE:
        problems.append(f"dataset size {path.stat().st_size} != {DATASET_SIZE}")
    if sha256_file(path) != recorded:
        problems.append("dataset SHA-256 differs from configs/m2/data.json")
    return problems


def check_upstream_hashes(root: Path) -> list:
    root = Path(root)
    cfg = json.loads((root / "configs" / "milestone1_artifact_full.json").read_text(encoding="utf-8"))
    problems = []
    for rel, expected in cfg["upstream_source_sha256"].items():
        path = root / "upstream" / "PAVE_artifact" / rel
        if not path.is_file():
            problems.append(f"missing PAVE source {rel}")
        elif sha256_file(path) != expected:
            problems.append(f"PAVE source SHA-256 differs: {rel}")
    return problems


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def git_tree_id(entries: dict) -> str:
    def build(items):
        children, files = {}, {}
        for path, (mode, blob) in items.items():
            head, _, rest = path.partition("/")
            if rest:
                children.setdefault(head, {})[rest] = (mode, blob)
            else:
                files[head] = (mode, blob)
        records = [(n, m.encode(), bytes.fromhex(b)) for n, (m, b) in files.items()]
        records += [(n, b"40000", bytes.fromhex(build(sub))) for n, sub in children.items()]
        records.sort(key=lambda r: r[0].encode() + (b"/" if r[0] in children else b""))
        body = b"".join(mode + b" " + name.encode() + b"\0" + raw for name, mode, raw in records)
        return hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()
    return build(entries)


def check_pave(root: Path) -> list:
    """Recompute every PAVE file hash, git blob id and the root tree id; compare with PACKAGE_IDENTITY.json."""
    root = Path(root)
    identity = json.loads((root / IDENTITY).read_text(encoding="utf-8"))
    pave = identity["pave"]
    upstream = root / "upstream" / "PAVE_artifact"
    problems = []
    if pave["commit"] != PAVE_PIN:
        problems.append("recorded PAVE commit is not the pin")
    entries = {}
    for rel, rec in sorted(pave["files"].items()):
        path = upstream / rel
        if not path.is_file():
            problems.append(f"PAVE file missing: {rel}")
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != rec["sha256"]:
            problems.append(f"PAVE file SHA-256 differs: {rel}")
        if git_blob_id(data) != rec["git_blob_id"] and git_blob_id(data.replace(b"\r\n", b"\n")) != rec["git_blob_id"]:
            problems.append(f"PAVE git blob id differs: {rel}")
        entries[rel] = (rec["mode"], rec["git_blob_id"])
    if not problems and git_tree_id(entries) != pave["tree_id"]:
        problems.append("PAVE tree id differs from PACKAGE_IDENTITY.json")
    present = {p.relative_to(upstream).as_posix() for p in upstream.rglob("*") if p.is_file()}
    extra = sorted(present - set(pave["files"]))
    if extra:
        problems.append(f"PAVE directory has unrecorded files: {extra[:5]}")
    return problems


def check_runbook_table(root: Path):
    """Regenerate the fingerprint table from research/ and compare with VM_RUNBOOK.md. None when imports are missing."""
    root = Path(root)
    sys.path.insert(0, str(root))
    try:
        from research import m2b_state  # needs torch and psutil: available after stage 2, not before
    except ImportError as exc:
        return None, f"skipped (project imports unavailable: {exc.name})"
    finally:
        sys.path.pop(0)
    text = (root / "VM_RUNBOOK.md").read_text(encoding="utf-8")
    sets = m2b_state.fingerprint_sets(root / "research")
    if not m2b_state.runbook_table_matches(text, sets):
        return False, "runbook fingerprint table differs from the regenerated table"
    return True, "runbook fingerprint table matches research/ imports"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root)
    failed = False

    result = check_manifest(root)
    print(f"manifest: {result['listed']} listed, {result['present']} present, "
          f"{len(result['missing'])} missing, {len(result['mismatched'])} mismatched, "
          f"{len(result['unlisted'])} unlisted, {len(result['symlinks'])} symlinks")
    for key in ("missing", "mismatched", "unlisted", "symlinks"):
        for rel in result[key][:20]:
            print(f"  {key}: {rel}")
    failed |= any(result[k] for k in ("missing", "mismatched", "unlisted", "symlinks"))
    if args.manifest_only:
        print("RESULT:", "FAIL" if failed else "PASS (manifest only)")
        return 1 if failed else 0

    for label, problems in (("line endings (LF)", check_line_endings(root)), ("dataset identity", check_dataset(root)),
                            ("PAVE source hashes", check_upstream_hashes(root)), ("PAVE tree id", check_pave(root))):
        print(f"{label}: {'PASS' if not problems else 'FAIL'}")
        for problem in problems:
            print(f"  {problem}")
        failed |= bool(problems)
    ok, message = check_runbook_table(root)
    print(f"runbook table: {message}")
    failed |= ok is False
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
