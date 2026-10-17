"""
run_pgd.py — Milestone 1: PGD Adversarial Evaluation Wrapper.

ADVERSARIAL VALIDITY CAVEAT
============================
Unconstrained feature-space adversarial perturbations (FGSM, PGD) applied to
network traffic features may produce feature vectors that do not correspond to
any physically realizable network packet sequence. Results should be interpreted
in the feature-space threat model only.

The PAVE framework provides both unconstrained (this wrapper) and
constraint-respecting (projected) evaluation modes. The 'proj' experiment
reports projected-PGD MCC alongside unconstrained-PGD MCC and the validity
rate of unconstrained examples, directly quantifying the overstated robustness
gap that arises when protocol validity is not enforced.
============================

Executes the PAVE PGD adversarial evaluation on the processed CICIoT2023
test split using the authors' own evaluation script at the **authors'
default epsilon, alpha, and steps** (ε = 0.10, α = 0.01, steps = 10,
as defined in sota_compare.py: PGD = dict(eps=0.1, alpha=0.01, iters=10)).

No researcher-specified epsilon, alpha, or steps are injected into the
upstream command. The upstream module-level PGD constant governs the
attack entirely.

Seed policy:
    No seed override is injected by this wrapper. The upstream script
    defaults to seeds [1, 7, 13].

Output:
    results/milestone1/pgd.txt  —  full stdout+stderr of the upstream run.

Usage:
    python experiments/milestone1/run_pgd.py
    python experiments/milestone1/run_pgd.py --dry-run
    python experiments/milestone1/run_pgd.py --quick
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_utils import tee_run, resolve_eval_script

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_UPSTREAM     = _PROJECT_ROOT / "upstream" / "PAVE_artifact"
_RESULT_PATH  = _PROJECT_ROOT / "results" / "milestone1" / "pgd.txt"


def build_cmd(eval_script: pathlib.Path, quick: bool = False) -> list[str]:
    """
    Build the subprocess command for the PGD evaluation.

    The 'proj' sub-experiment in revision_experiments.py reports:
        - MCC clean
        - MCC unconstrained PGD  (seq_pgd using PGD dict constant)
        - MCC projected PGD      (constraint-respecting)
        - Protocol validity rate

    All PGD parameters (epsilon, alpha, iters) come from the upstream
    module-level constant PGD = dict(eps=0.1, alpha=0.01, iters=10).
    No parameter literals appear in this wrapper.
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
        help="Pass --quick to upstream (smoke test).",
    )
    args = ap.parse_args()

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
        print()
        print("Note: PGD epsilon/alpha/steps are NOT specified on the command line.")
        print("      The upstream module-level PGD dict governs the attack.")
        sys.exit(0)

    print(f"Running PGD evaluation -> {_RESULT_PATH}")
    print(f"Command: {' '.join(cmd)}")
    print(f"CWD    : {_UPSTREAM}")
    print()

    rc = tee_run(cmd=cmd, result_path=_RESULT_PATH, cwd=_UPSTREAM)
    sys.exit(rc)


if __name__ == "__main__":
    main()
