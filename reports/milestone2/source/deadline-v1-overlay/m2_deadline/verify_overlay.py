"""Verify the overlay folder and the V4 package it runs on. Standard library only; read-only on both.

    python -m m2_deadline.verify_overlay --overlay <O> --v4-root <V4>

1. Overlay files match OVERLAY_MANIFEST.sha256 (no missing, mismatched, unlisted, symlink or CR).
2. sha256(V4/MANIFEST.sha256) equals expects_v4_manifest_sha256 in OVERLAY_IDENTITY.json.
3. V4's manifest against V4's files (stdlib). Unlisted V4 files are a WARN; missing, mismatched or symlinks fail.
4. V4's own verify_package.py --manifest-only as a second check (SKIPPED when it cannot start).

Prints OK and exits 0 on success. Exit 3 on any failure, 2 for usage errors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from m2_deadline import manifest as mf

UNLISTED_WARNING = ("V4 has unlisted files; V4's own stage 01 would reject them; delete them "
                    "(they are not part of the verified package)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m m2_deadline.verify_overlay",
        description="Check the overlay manifest, the expected V4 manifest hash and V4's own manifest (read-only).")
    p.add_argument("--overlay", required=True, help="overlay folder")
    p.add_argument("--v4-root", required=True, help="V4 package folder")
    return p


def _report(label: str, result: dict) -> List[str]:
    failed = []
    for key in ("missing", "mismatched", "unlisted", "symlinks", "crlf"):
        names = result.get(key) or []
        for rel in names[:20]:
            print("  %s %s: %s" % (label, key, rel))
        if names:
            failed.append(key)
    return failed


def main(argv: Optional[Sequence[str]] = None, *, runner: Optional[Callable[..., Any]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    overlay = Path(args.overlay).resolve()
    v4_root = Path(args.v4_root).resolve()
    if not overlay.is_dir():
        print("ERROR: overlay folder not found: %s" % overlay, file=sys.stderr)
        return mf.EXIT_USAGE
    for needed in (overlay / mf.OVERLAY_MANIFEST, overlay / mf.OVERLAY_IDENTITY):
        if not needed.is_file():
            print("ERROR: %s is missing; run m2_deadline.make_manifest" % needed.name, file=sys.stderr)
            return mf.EXIT_USAGE
    if not (v4_root / mf.V4_MANIFEST).is_file():
        print("ERROR: V4 %s not found under %s" % (mf.V4_MANIFEST, v4_root), file=sys.stderr)
        return mf.EXIT_USAGE

    failed = False

    # 1. overlay files
    try:
        entries = mf.parse_manifest((overlay / mf.OVERLAY_MANIFEST).read_bytes())
    except ValueError as exc:
        print("FAIL overlay manifest: %s" % exc)
        return mf.EXIT_GUARD
    result = mf.check_overlay(overlay, entries)
    bad = _report("overlay", result)
    print("overlay: %d listed, %s" % (len(entries), "FAIL (%s)" % ", ".join(bad) if bad else "PASS"))
    failed |= bool(bad)

    # 2. identity
    try:
        identity = json.loads((overlay / mf.OVERLAY_IDENTITY).read_text(encoding="utf-8"))
    except ValueError as exc:
        print("FAIL identity file is not valid JSON: %s" % exc)
        return mf.EXIT_GUARD
    actual_v4 = mf.sha256_file(v4_root / mf.V4_MANIFEST)
    if identity.get("expects_v4_manifest_sha256") != actual_v4:
        print("FAIL V4 MANIFEST.sha256 hash %s differs from the expected %s"
              % (actual_v4, identity.get("expects_v4_manifest_sha256")))
        failed = True
    else:
        print("V4 manifest hash: PASS")
    cfg = overlay / mf.CONFIG_REL
    if not cfg.is_file() or identity.get("config_sha256") != mf.sha256_file(cfg):
        print("FAIL %s differs from OVERLAY_IDENTITY.json" % mf.CONFIG_REL)
        failed = True

    # 3. V4 files, stdlib
    try:
        v4_result = mf.check_v4(v4_root)
    except ValueError as exc:
        print("FAIL V4 manifest: %s" % exc)
        return mf.EXIT_GUARD
    hard = {k: v for k, v in v4_result.items() if k != "unlisted" and v}
    for key, names in hard.items():
        for rel in names[:20]:
            print("  V4 %s: %s" % (key, rel))
    if hard:
        print("V4 files: FAIL (%s)" % ", ".join(sorted(hard)))
        failed = True
    else:
        print("V4 files: PASS")
    if v4_result["unlisted"]:
        print("WARN %s" % UNLISTED_WARNING)
        for rel in v4_result["unlisted"][:20]:
            print("  V4 unlisted: %s" % rel)

    # 4. second check
    second = mf.second_check(v4_root, runner)
    if second["status"] == "ok":
        print("V4 verify_package --manifest-only: PASS")
    elif second["status"] == "warn":
        print("WARN V4 verify_package --manifest-only reports only unlisted files")
        print("WARN %s" % UNLISTED_WARNING)
    elif second["status"] == "skipped":
        print("V4 verify_package --manifest-only: SKIPPED (%s)" % second["detail"])
    else:
        print("V4 verify_package --manifest-only: FAIL")
        print(second["detail"])
        failed = True

    if failed:
        print("FAILED")
        return mf.EXIT_GUARD
    print("OK")
    return mf.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
