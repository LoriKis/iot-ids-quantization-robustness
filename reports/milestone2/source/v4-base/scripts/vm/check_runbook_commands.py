#!/usr/bin/env python3
"""Check that every command in the fenced code blocks of VM_RUNBOOK.md resolves to a real entry point.

Allowed forms (after stripping `VAR=value` prefixes, `nohup`, trailing `&` and redirections):
  bash scripts/vm/<existing .sh> [args]       python3|.venv/bin/python scripts/vm/<existing .py> [args]
  .venv/bin/python [-B] -m research.<existing module> <subcommand defined in that module> [--flags defined there]
  .venv/bin/python -m pytest ...              sha256sum, tmux, kill, tail, ls, cat, grep, df, free, nproc, sudo apt, cd, export
Flags used with bash scripts must appear in the script text. Static only: nothing is executed.
"""
from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAIN = {"sha256sum", "tmux", "kill", "tail", "ls", "cat", "grep", "df", "free", "nproc", "cd", "export", "uname",
         "lscpu", "python3.11", "head", "wc", "du", "mkdir", "less", "pkill", "pgrep", "top", "htop", "nl"}


def code_lines(text: str):
    in_block = False
    pending = ""
    for number, line in enumerate(text.split("\n"), 1):
        if line.strip().startswith("```"):
            in_block = not in_block
            pending = ""
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        yield number, (pending + stripped).strip()
        pending = ""


def clean(command: str) -> list:
    command = re.sub(r"\s+2>&1", "", command)
    command = re.sub(r"\s+>\s*\S+", "", command)
    command = re.sub(r"\s+&\s*$", "", command)
    tokens = shlex.split(command, comments=True)
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        tokens.pop(0)
    if tokens and tokens[0] == "$":
        tokens.pop(0)
    while tokens and tokens[0] == "nohup":
        tokens.pop(0)
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        tokens.pop(0)
    return tokens


def check(tokens: list, root: Path) -> str | None:
    if not tokens:
        return None
    head = tokens[0]
    if head == "bash" and len(tokens) >= 2 and tokens[1].startswith("scripts/vm/"):
        script = root / tokens[1]
        if not script.is_file():
            return f"missing script {tokens[1]}"
        body = script.read_text(encoding="utf-8")
        for flag in (t for t in tokens[2:] if t.startswith("--")):
            if flag not in body:
                return f"flag {flag} not handled by {tokens[1]}"
        return None
    if head in ("python3", ".venv/bin/python") and len(tokens) >= 2 and tokens[1].startswith("scripts/vm/"):
        return None if (root / tokens[1]).is_file() else f"missing script {tokens[1]}"
    if head == ".venv/bin/python":
        rest = [t for t in tokens[1:] if t not in ("-B",)]
        if rest[:2] == ["-m", "pytest"]:
            return None
        if rest[:1] == ["-m"] and len(rest) >= 2 and rest[1].startswith("research."):
            module = root / (rest[1].replace(".", "/") + ".py")
            if not module.is_file():
                return f"missing module {rest[1]}"
            body = module.read_text(encoding="utf-8")
            if len(rest) >= 3 and not rest[2].startswith("-"):
                if not re.search(r'add_parser\(\s*"%s"' % re.escape(rest[2]), body):
                    return f"subcommand {rest[2]} not defined in {rest[1]}"
            for flag in (t for t in rest[2:] if t.startswith("--")):
                if f'"{flag}"' not in body:
                    return f"flag {flag} not defined in {rest[1]}"
            return None
        if rest[:1] == ["-c"]:
            return None
        return "unrecognized .venv/bin/python invocation"
    if head == "sudo" and tokens[1:2] == ["apt"]:
        return None
    if head == "cd" or head in PLAIN:
        return None
    if head in ("python3", "python3.11") and tokens[1:2] in (["--version"], ["-m"]):
        return None
    return f"command not in the allow-list: {head}"


def main(argv=None) -> int:
    root = Path(argv[0]) if argv else ROOT
    runbook = root / "VM_RUNBOOK.md"
    if not runbook.is_file():
        runbook = root / "docs" / "VM_RUNBOOK.md"
    text = runbook.read_text(encoding="utf-8")
    bad = checked = 0
    for number, line in code_lines(text):
        problem = check(clean(line), root)
        checked += 1
        if problem:
            bad += 1
            print(f"{runbook.name}:{number}: {problem}: {line}")
    print(f"{checked} runbook command line(s) checked, {bad} unresolved")
    return 1 if bad or not checked else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
