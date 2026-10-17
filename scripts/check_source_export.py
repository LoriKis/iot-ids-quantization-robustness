"""Check that a local source ZIP includes every tracked file and key directory.

Read only. Accept a plain git archive or a ZIP with one enclosing project folder.
Run against the same local revision used to create the archive.
"""
import argparse
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def check(path):
    tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode().split("\0")
    tracked = {p for p in tracked if p}
    with zipfile.ZipFile(path) as archive:
        names = {n.replace("\\", "/") for n in archive.namelist()}
        prefixes = {n[:-len("README.md")] for n in names if n.endswith("README.md")}
        prefix = max(prefixes or {""}, key=lambda p: sum(p + name in names for name in tracked))
        missing = sorted(name for name in tracked if prefix + name not in names)
        for directory in ["configs", "docs", "experiments"]:
            if not any(n.startswith(prefix + directory + "/") for n in names):
                missing.append(directory + "/ (entire directory)")
        bad = archive.testzip()
    if missing or bad:
        raise ValueError(f"Export incomplete/corrupt: missing={missing}; bad_member={bad}")
    return len(tracked)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    args = parser.parse_args()
    try:
        print(f"Verified {check(args.zip)} tracked files, key directories and ZIP CRCs")
    except ValueError as exc:
        parser.exit(1, str(exc) + "\n")
