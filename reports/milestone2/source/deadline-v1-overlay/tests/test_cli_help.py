"""Every CLI answers --help without importing torch; timing and launch/resume refuse missing or zero hours."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from m2_deadline import campaign, timing

OVERLAY = Path(__file__).resolve().parents[1]
CLIS = ("timing", "campaign", "report", "verify_overlay", "make_manifest")

HELP_PROGRAM = (
    "import sys, importlib\n"
    "mod = importlib.import_module('m2_deadline.' + sys.argv[1])\n"
    "try:\n"
    "    mod.main(['--help'])\n"
    "except SystemExit as exc:\n"
    "    code = exc.code\n"
    "else:\n"
    "    code = 'no-exit'\n"
    "bad = [m for m in ('torch', 'research', 'pandas', 'matplotlib', 'psutil') if m in sys.modules]\n"
    "print('CODE=%s BAD=%s' % (code, ','.join(bad)))\n"
)


@pytest.mark.parametrize("cli", CLIS)
def test_help_exits_0_without_heavy_imports(cli):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(OVERLAY)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    done = subprocess.run([sys.executable, "-B", "-c", HELP_PROGRAM, cli], capture_output=True, text=True, env=env,
                          cwd=str(OVERLAY), timeout=120)
    assert done.returncode == 0, done.stderr
    assert "CODE=0 BAD=" in done.stdout
    assert done.stdout.strip().endswith("BAD=")  # nothing heavy was imported
    assert "usage:" in done.stdout


def test_timing_without_or_with_zero_hours_exits_2(tmp_path):
    out = str(tmp_path / "out")
    assert timing.main(["--out", out]) == 2
    assert timing.main(["--hours-remaining", "0", "--out", out]) == 2
    assert timing.main(["--hours-remaining", "13", "--out", out]) == 2
    assert not Path(out).exists()


@pytest.mark.parametrize("command", ["launch", "resume"])
def test_launch_and_resume_without_or_with_zero_hours_exit_2(tmp_path, command):
    out = str(tmp_path / "out")
    assert campaign.main([command, "--out", out]) == 2
    assert campaign.main([command, "--hours-remaining", "0", "--out", out]) == 2
    assert not Path(out).exists()
