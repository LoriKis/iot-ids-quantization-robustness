#!/usr/bin/env python3
"""Static shell checks for scripts/vm/*.sh. Uses `bash -n` when a working bash exists; otherwise only the static checks.

Static checks: shebang, `set -euo pipefail`, no CR bytes, even number of unescaped double quotes per line,
balanced if/fi, case/esac and function braces. Without bash, `bash -n` is reported as pending VM verification.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def static_problems(path: Path) -> list:
    raw = path.read_bytes()
    problems = []
    if b"\r" in raw:
        problems.append("contains CR (CRLF line endings)")
    text = raw.decode("utf-8")
    lines = text.split("\n")
    if not lines[0].startswith("#!/usr/bin/env bash"):
        problems.append("first line is not '#!/usr/bin/env bash'")
    if not any(line.strip() == "set -euo pipefail" for line in lines):
        problems.append("missing 'set -euo pipefail'")
    counts = {"if": 0, "fi": 0, "case": 0, "esac": 0, "open": 0, "close": 0}
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        code = re.sub(r"\\.", "", line)
        if code.count('"') % 2:
            problems.append(f"line {number}: odd number of double quotes")
        for key, pattern in (("if", r"(?:^|;\s*)if\b"), ("fi", r"(?:^|;\s*)fi\b"), ("case", r"^case\b"), ("esac", r"^esac\b"),
                             ("open", r"^[A-Za-z_][A-Za-z0-9_]*\(\)\s*\{\s*$"), ("close", r"^\}\s*$")):
            if re.search(pattern, stripped):
                counts[key] += 1
    for a, b in (("if", "fi"), ("case", "esac"), ("open", "close")):
        if counts[a] != counts[b]:
            problems.append(f"unbalanced {a}/{b}: {counts[a]} vs {counts[b]}")
    return problems


def working_bash():
    bash = shutil.which("bash")
    if not bash:
        return None
    try:
        probe = subprocess.run([bash, "-c", "echo ok"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return bash if probe.returncode == 0 and probe.stdout.strip() == "ok" else None


def main(argv=None) -> int:
    root = Path(argv[0]) if argv else ROOT
    scripts = sorted((root / "scripts" / "vm").glob("*.sh"))
    if not scripts:
        print("FAIL: no scripts/vm/*.sh found")
        return 1
    failed = False
    bash = working_bash()
    for path in scripts:
        problems = static_problems(path)
        rel = path.relative_to(root).as_posix()
        if bash and not problems:
            result = subprocess.run([bash, "-n", rel], capture_output=True, text=True, cwd=root)
            if result.returncode != 0:
                problems.append("bash -n: " + (result.stderr.strip() or "failed"))
        print(f"{'FAIL' if problems else 'ok  '} {rel}")
        for problem in problems:
            print(f"     {problem}")
        failed |= bool(problems)
    print(f"{len(scripts)} script(s) checked; bash -n: " + ("run" if bash else "pending VM verification (no working bash here)"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
