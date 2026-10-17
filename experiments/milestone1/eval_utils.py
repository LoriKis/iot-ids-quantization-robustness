"""
eval_utils.py — Shared utilities for Milestone 1 evaluation wrappers.

Provides:
  tee_run(cmd, result_path, cwd)
      Run a subprocess, streaming its combined stdout+stderr to both the
      console and a result file simultaneously (tee pattern).

  resolve_eval_script(upstream_root, readme_path)
      Parse the upstream README.md or notes/pave_reproduction_notes.md to
      locate the primary evaluation entry-point script.

These utilities contain no attack logic and require no adversarial caveat.
All evaluation decisions (attack type, parameters) are made by the upstream
PAVE scripts invoked via tee_run.
"""

from __future__ import annotations

import datetime
import pathlib
import re
import subprocess
import sys
from typing import Sequence


# ---------------------------------------------------------------------------
# tee_run
# ---------------------------------------------------------------------------

def tee_run(
    cmd: Sequence[str],
    result_path: pathlib.Path | str,
    cwd: pathlib.Path | str,
) -> int:
    """
    Execute *cmd* as a subprocess, streaming combined stdout+stderr to both
    the calling process's stdout and *result_path*.

    The result file is created (or overwritten) with a timestamped header and
    footer so that wall-clock runtime is visible in the captured output.

    Args:
        cmd:         Sequence of command-line tokens (passed to subprocess.Popen).
        result_path: Path to the output capture file. Parent directories are
                     created automatically.
        cwd:         Working directory for the subprocess. Should be set to
                     upstream/PAVE_artifact/ so that relative imports resolve.

    Returns:
        The upstream script's exit code (int). Callers should propagate this
        as their own exit code to preserve pipeline failure signals.

    Example::

        rc = tee_run(
            ["python", "code/revision_experiments.py", "--exp", "proj", "--quick"],
            result_path=pathlib.Path("results/milestone1/clean.txt"),
            cwd=pathlib.Path("upstream/PAVE_artifact"),
        )
        sys.exit(rc)
    """
    result_path = pathlib.Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cwd = pathlib.Path(cwd)

    start_ts = datetime.datetime.utcnow().isoformat() + "Z"
    cmd_str = " ".join(str(t) for t in cmd)

    with open(result_path, "w", encoding="utf-8") as fout:
        fout.write(f"# ── Run started:  {start_ts}\n")
        fout.write(f"# ── Command:      {cmd_str}\n")
        fout.write(f"# ── Working dir:  {cwd}\n")
        fout.write("#\n")
        fout.flush()

        proc = subprocess.Popen(
            [str(t) for t in cmd],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # line-buffered
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fout.write(line)
            fout.flush()

        proc.wait()

        end_ts = datetime.datetime.utcnow().isoformat() + "Z"
        fout.write("#\n")
        fout.write(f"# ── Run ended:    {end_ts}\n")
        fout.write(f"# ── Exit code:    {proc.returncode}\n")

    return proc.returncode


# ---------------------------------------------------------------------------
# resolve_eval_script
# ---------------------------------------------------------------------------

def resolve_eval_script(
    upstream_root: pathlib.Path | str,
    notes_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """
    Locate the primary PAVE evaluation script.

    Search order:
      1. Parse ``notes/pave_reproduction_notes.md`` for lines under
         "Evaluation Script (from README)" that contain a ``.py`` path.
      2. Parse ``upstream_root/README.md`` directly for fenced code blocks
         that reference a ``python`` invocation.
      3. Fall back to ``upstream_root/code/revision_experiments.py`` if it
         exists (the known primary entry-point as of the cloned commit).

    Args:
        upstream_root: Path to upstream/PAVE_artifact.
        notes_dir:     Path to notes/ directory.  Defaults to
                       upstream_root/../../notes/.

    Returns:
        Resolved absolute pathlib.Path to the evaluation script.

    Raises:
        FileNotFoundError: if no candidate is found anywhere.
    """
    upstream_root = pathlib.Path(upstream_root).resolve()

    if notes_dir is None:
        notes_dir = upstream_root.parent.parent / "notes"
    else:
        notes_dir = pathlib.Path(notes_dir).resolve()

    # ── Strategy 1: check discovery report ───────────────────────────────────
    report = notes_dir / "pave_reproduction_notes.md"
    if report.exists():
        text = report.read_text(encoding="utf-8", errors="replace")
        # Look for a line like:  python revision_experiments.py ...
        for line in text.splitlines():
            m = re.search(r"python\s+([\w/\\.-]+\.py)", line)
            if m:
                candidate = upstream_root / m.group(1)
                if candidate.exists():
                    return candidate

    # ── Strategy 2: parse upstream README ────────────────────────────────────
    readme = upstream_root / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="replace")
        in_block = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_block = not in_block
                continue
            if in_block and ("python" in stripped or stripped.startswith("py ")):
                m = re.search(r"(?:python|py)\s+([\w/\\.-]+\.py)", stripped)
                if m:
                    candidate = upstream_root / m.group(1)
                    if candidate.exists():
                        return candidate

    # ── Strategy 3: known fallback ───────────────────────────────────────────
    fallback = upstream_root / "code" / "revision_experiments.py"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Cannot locate the PAVE evaluation script inside {upstream_root}. "
        "Run discover_artifacts.py first to generate notes/pave_reproduction_notes.md."
    )
