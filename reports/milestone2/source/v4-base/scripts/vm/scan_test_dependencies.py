#!/usr/bin/env python3
"""Static scan (stdlib; executes nothing): which paths do the tests, research code and scripts reference that are not
present in this tree, which tests call git/subprocess, and which writes in tests/ lie outside a temporary directory.

    python scripts/vm/scan_test_dependencies.py [--root DIR] [--write-report PATH]

Every finding needs a decision in DECISIONS (edit this file, keep the reason short). Exit 1 if any finding is undecided.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH_RE = re.compile(r"^(?:\./)?(runs|work|artifacts|reports|configs|upstream|docs|notes|licenses|scripts|research|tests|output)/[\w./\-]+$")
WRITE_CALLS = {"write_text", "write_bytes", "to_csv", "savefig", "dump", "mkdir", "copy", "copy2", "copyfile",
               "copytree", "atomic_json", "atomic_torch", "durable_write_bytes", "durable_write_json", "touch"}
TEMP_HINTS = ("tmp_path", "tmp_dir", "tempfile", "tmpdir", "TemporaryDirectory", "tmp")
GIT_NAMES = {"git", "verify_upstream", "check_output", "run", "Popen"}

# (kind, file-or-path) -> decision. kind: missing_path | git | write.
DECISIONS: dict = {
    ("missing_path", "tests/test_m2_guard.py"): "negative-test path that must not exist; no change",
    ("missing_path", "tests/test_m2b_state.py"): "synthetic paths inside tmp_path fixtures; no change",
    ("missing_path", "research/paper_described_report.py"): "Milestone 1 report default under runs/ (evidence excluded from the package); no M2 stage calls it",
    ("missing_path", "research/analyze.py"): "Milestone 1 plotting scratch dir under work/ (excluded); not used by any M2B stage or test",
    ("missing_path", "research/milestone1_report.py"): "Milestone 1 preflight file under work/ (excluded); not used by any M2B stage or test",
    ("missing_path", "research/paper_described_adam.py"): "Milestone 1 run directories under runs/ (excluded evidence); the VM does not rerun Milestone 1",
    ("missing_path", "scripts/check_m2_integrity.py"): "local preservation check naming excluded evidence; not run on the VM and not invoked by the suite",
    ("missing_path", "scripts/check_milestone1_adam_review.py"): "Milestone 1 review script reading runs/ (excluded); not run on the VM",
    ("missing_path", "scripts/check_milestone1_review.py"): "Milestone 1 review script reading runs/ (excluded); not run on the VM",
    ("missing_path", "scripts/complete_milestone1.py"): "Milestone 1 completion script reading runs/ (excluded); not run on the VM",    ("missing_path", "tests/test_m2b_package.py"): "synthetic file names inside a fake repository under tmp_path; no change",
    ("git", "tests/test_m2b_package.py"): "exercises the common.git shim on a synthetic tree; no real git needed",
    ("git", "tests/test_pipeline.py"): "verify_upstream() is served by the common.git shim in the package (tree id verified); no skip needed",
}


def literal_paths(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and PATH_RE.match(node.value):
            yield node.lineno, node.value


def git_calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("verify_upstream", "check_output", "Popen") or (name == "run" and "subprocess" in ast.unparse(func)):
                yield node.lineno, ast.unparse(node)[:90]
            elif name == "git" or name == "_git":
                yield node.lineno, ast.unparse(node)[:90]


def write_calls(tree, source):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        is_open = name == "open" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
            and isinstance(node.args[1].value, str) and any(c in node.args[1].value for c in "wax")
        shutil_copy = name.startswith("copy") and "shutil" not in ast.unparse(func)
        if (name in WRITE_CALLS and not shutil_copy) or is_open:
            yield node.lineno, name, ast.unparse(node)[:100]


def scan(root: Path) -> dict:
    root = Path(root)
    findings = {"missing_path": [], "git": [], "write": []}
    for area in ("tests", "research", "scripts"):
        for path in sorted((root / area).rglob("*.py")):
            if "__pycache__" in path.parts or path.name in ("scan_test_dependencies.py",):
                continue
            rel = path.relative_to(root).as_posix()
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError as exc:
                findings["git"].append((rel, exc.lineno or 0, f"unparsable: {exc.msg}"))
                continue
            for line, value in literal_paths(tree):
                if any(ch in value for ch in "*{}"):
                    continue
                if not (root / value.removeprefix("./")).exists():
                    findings["missing_path"].append((rel, line, value))
            if area == "tests":
                for line, text in git_calls(tree):
                    findings["git"].append((rel, line, text))
                temp_based = any(hint in source for hint in TEMP_HINTS)
                for line, name, text in write_calls(tree, source):
                    if not temp_based:
                        findings["write"].append((rel, line, f"{name}: {text}"))
    return findings


def decision_for(kind: str, rel: str, detail: str):
    for key in ((kind, rel, detail), (kind, rel), (kind, detail)):
        if key in DECISIONS:
            return DECISIONS[key]
    return None


def render(findings: dict) -> tuple:
    lines = ["# Package test dependencies (static scan, Milestone 2B VM package)", "",
             "Generated by `scripts/vm/scan_test_dependencies.py`; nothing was executed. Each row has a decision.", ""]
    undecided = 0
    titles = {"missing_path": "Referenced repository paths that do not exist in this tree",
              "git": "Tests that call git or subprocess (the VM package has no .git; common.git serves a fixed table)",
              "write": "Writes in tests that are not obviously redirected to a temporary directory"}
    for kind in ("missing_path", "git", "write"):
        lines += [f"## {titles[kind]}", ""]
        rows = findings[kind]
        if not rows:
            lines += ["None.", ""]
            continue
        lines += ["| file | line | detail | decision |", "|---|---|---|---|"]
        for rel, line, detail in rows:
            decision = decision_for(kind, rel, detail)
            if decision is None:
                undecided += 1
                decision = "**UNDECIDED**"
            lines.append(f"| `{rel}` | {line} | `{detail.replace('|', '/')}` | {decision} |")
        lines.append("")
    lines += [f"Undecided findings: {undecided}", ""]
    return "\n".join(lines), undecided


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--write-report", default=None)
    args = parser.parse_args(argv)
    findings = scan(Path(args.root))
    text, undecided = render(findings)
    if args.write_report:
        Path(args.write_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_report).write_bytes(text.encode("utf-8"))
    print(text)
    return 1 if undecided else 0


if __name__ == "__main__":
    sys.exit(main())
