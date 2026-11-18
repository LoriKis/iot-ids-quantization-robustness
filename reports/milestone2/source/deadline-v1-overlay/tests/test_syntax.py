"""Python 3.9 syntax of every overlay .py, LF-only text files, and `bash -n` on the wrappers when bash exists."""
from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

OVERLAY = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = (".py", ".sh", ".json", ".md", ".txt", ".ini", ".sha256")
SKIP_DIRS = {"__pycache__", ".pytest_cache", "artifacts"}


def overlay_files(suffixes):
    found = []
    for path in sorted(OVERLAY.rglob("*")):
        rel = path.relative_to(OVERLAY)
        if path.is_file() and not (set(rel.parts) & SKIP_DIRS) and path.name.endswith(suffixes):
            found.append(path)
    return found


@pytest.mark.parametrize("path", overlay_files((".py",)), ids=lambda p: p.relative_to(OVERLAY).as_posix())
def test_python39_syntax(path):
    ast.parse(path.read_text(encoding="utf-8"), str(path), feature_version=(3, 9))


def test_text_files_are_lf_only():
    bad = [p.relative_to(OVERLAY).as_posix() for p in overlay_files(TEXT_SUFFIXES) if b"\r" in p.read_bytes()]
    assert not bad, "CR bytes in: %s" % bad


def test_wrappers_exist_and_use_strict_mode():
    names = {"_env.sh", "verify.sh", "timing.sh", "launch.sh", "resume.sh", "report.sh", "status.sh"}
    folder = OVERLAY / "scripts" / "vm_deadline"
    assert names <= {p.name for p in folder.glob("*.sh")}
    assert "set -euo pipefail" in (folder / "_env.sh").read_text(encoding="utf-8")
    for name in names - {"_env.sh"}:
        assert name and "_env.sh" in (folder / name).read_text(encoding="utf-8")


def test_bash_syntax_of_wrappers():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available on this machine: `bash -n` SKIPPED (run it on the VM)")
    for path in sorted((OVERLAY / "scripts" / "vm_deadline").glob("*.sh")):
        done = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True, timeout=60)
        if done.returncode != 0 and "No such file" in done.stderr:
            pytest.skip("bash is present but cannot read Windows paths: SKIPPED")
        assert done.returncode == 0, "%s: %s" % (path.name, done.stderr)
