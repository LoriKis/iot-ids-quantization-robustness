"""Write OVERLAY_IDENTITY.json and OVERLAY_MANIFEST.sha256 (run on the laptop once all overlay files are final).

    python -m m2_deadline.make_manifest --overlay <O> --v4-root <local V4>

Refuses (exit 3) when a __pycache__ or .pytest_cache folder, a symlink or a CR byte in a text file exists under
the overlay. Exit 2 for usage errors. V4 is only read. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from m2_deadline import manifest as mf


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(text.replace("\r\n", "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def build_identity(overlay: Path, v4_root: Path) -> Dict[str, Any]:
    return {
        "schema": 1,
        "kind": mf.IDENTITY_KIND,
        "config_path": mf.CONFIG_REL,
        "config_sha256": mf.sha256_file(overlay / mf.CONFIG_REL),
        "expects_v4_manifest_sha256": mf.sha256_file(v4_root / mf.V4_MANIFEST),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m m2_deadline.make_manifest",
        description="Write OVERLAY_IDENTITY.json and OVERLAY_MANIFEST.sha256 for the overlay folder (read-only on V4).")
    p.add_argument("--overlay", required=True, help="overlay folder")
    p.add_argument("--v4-root", required=True, help="local verified V4 folder (only its MANIFEST.sha256 is read)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    overlay = Path(args.overlay).resolve()
    v4_root = Path(args.v4_root).resolve()
    if not overlay.is_dir():
        print("ERROR: overlay folder not found: %s" % overlay, file=sys.stderr)
        return mf.EXIT_USAGE
    if not (v4_root / mf.V4_MANIFEST).is_file():
        print("ERROR: V4 %s not found under %s" % (mf.V4_MANIFEST, v4_root), file=sys.stderr)
        return mf.EXIT_USAGE
    if not (overlay / mf.CONFIG_REL).is_file():
        print("ERROR: %s not found under the overlay" % mf.CONFIG_REL, file=sys.stderr)
        return mf.EXIT_USAGE

    files, links, strays = mf.walk_overlay(overlay)
    if strays:
        print("ERROR: delete these generated folders first: %s" % ", ".join(strays), file=sys.stderr)
        return mf.EXIT_GUARD
    if links:
        print("ERROR: symlinks are not allowed: %s" % ", ".join(links), file=sys.stderr)
        return mf.EXIT_GUARD
    bad = mf.crlf_files(overlay, files)
    if bad:
        print("ERROR: CR bytes found (LF only): %s" % ", ".join(bad), file=sys.stderr)
        return mf.EXIT_GUARD

    identity = build_identity(overlay, v4_root)
    _atomic_write(overlay / mf.OVERLAY_IDENTITY, json.dumps(identity, sort_keys=True, indent=2) + "\n")
    files, _links, _strays = mf.walk_overlay(overlay)  # identity file is part of the listing
    entries = {rel: mf.sha256_file(overlay / rel) for rel in files}
    _atomic_write(overlay / mf.OVERLAY_MANIFEST, mf.manifest_text(entries))
    print("wrote %s and %s (%d files listed)" % (mf.OVERLAY_IDENTITY, mf.OVERLAY_MANIFEST, len(entries)))
    return mf.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
