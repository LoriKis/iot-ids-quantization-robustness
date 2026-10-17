"""
run_clean.py — Milestone 1: Clean (unperturbed) Evaluation Wrapper.

Executes the PAVE Standard Transformer evaluation on the processed
CICIoT2023 test split using the authors' own evaluation script
(code/revision_experiments.py).

No adversarial perturbations are applied. This establishes the clean
accuracy / MCC baseline that adversarial evaluations are measured against.

Seed policy:
    No seed override is injected by this wrapper. The upstream script
    defaults to seeds [1, 7, 13] as defined in its argparse defaults.
    To reproduce exactly, the upstream default is used without modification.

Output:
    results/milestone1/clean.txt  —  full stdout+stderr of the upstream run.

Usage:
    python experiments/milestone1/run_clean.py
    python experiments/milestone1/run_clean.py --dry-run   # print command only
    python experiments/milestone1/run_clean.py --quick     # 1 dataset, 1 seed, 3 epochs
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Add experiments/milestone1 to path so eval_utils imports cleanly
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_utils import tee_run, resolve_eval_script

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_UPSTREAM     = _PROJECT_ROOT / "upstream" / "PAVE_artifact"
_RESULT_PATH  = _PROJECT_ROOT / "results" / "milestone1" / "clean.txt"


def build_cmd(eval_script: pathlib.Path, quick: bool = False) -> list[str]:
    """
    Build the subprocess command for the clean (no-attack) evaluation.

    Maps to the upstream README invocation:
        python revision_experiments.py --exp proj --datasets CICIoT

    The 'proj' experiment reports:
        - MCC clean (Standard Transformer + AGT)
        - MCC unconstrained PGD
        - MCC projected PGD
        - Protocol validity rate of unconstrained adversarial examples

    We request all three datasets and all three seeds to match the full
    paper baseline. Use --quick to run a smoke test (1 dataset, 1 seed).

    NOTE: 'proj' was chosen because it directly outputs clean MCC alongside
    adversarial MCC, giving a self-contained baseline table per run.
    """
    cmd = [
        sys.executable,
        str(eval_script.relative_to(_UPSTREAM)),
        "--exp", "proj",
        "--datasets", "CICIoT", "CICIoMT", "CICIoV",
        "--out", str((_PROJECT_ROOT / "results" / "milestone1").resolve()),
    ]
    if quick:
        cmd.append("--quick")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved command without executing it.",
    )
    ap.add_argument(
        "--quick", action="store_true",
        help="Pass --quick to upstream (1 dataset, 1 seed, 3 epochs — smoke test).",
    )
    args = ap.parse_args()

    # Resolve the upstream evaluation script
    try:
        eval_script = resolve_eval_script(_UPSTREAM, _PROJECT_ROOT / "notes")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    cmd = build_cmd(eval_script, quick=args.quick)

    if args.dry_run:
        print("Resolved command (dry-run):")
        print(" ".join(cmd))
        print(f"Working dir : {_UPSTREAM}")
        print(f"Output file : {_RESULT_PATH}")
        sys.exit(0)

    print(f"Running clean evaluation -> {_RESULT_PATH}")
    print(f"Command: {' '.join(cmd)}")
    print(f"CWD    : {_UPSTREAM}")
    print()

    rc = tee_run(cmd=cmd, result_path=_RESULT_PATH, cwd=_UPSTREAM)
    sys.exit(rc)


if __name__ == "__main__":
    main()
