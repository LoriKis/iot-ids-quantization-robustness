"""
run_fgsm.py — Milestone 1: FGSM Adversarial Evaluation Wrapper.

ADVERSARIAL VALIDITY CAVEAT
============================
Unconstrained feature-space adversarial perturbations (FGSM, PGD) applied to
network traffic features may produce feature vectors that do not correspond to
any physically realizable network packet sequence. Results should be interpreted
in the feature-space threat model only.

The PAVE framework provides both unconstrained (this wrapper) and
constraint-respecting (projected) evaluation modes. The 'proj' experiment
(run via revision_experiments.py) reports both MCC values and the
validity rate of unconstrained adversarial examples, directly quantifying
how much robustness is overstated when protocol validity is ignored.
============================

Executes the PAVE FGSM adversarial evaluation on the processed CICIoT2023
test split using the authors' own evaluation script at the **authors'
default epsilon** (ε = 0.10, as defined in sota_compare.py: EPS = 0.1).

No researcher-specified epsilon is injected into the upstream command.
The upstream script's own module-level constant governs the attack strength.

Seed policy:
    No seed override is injected by this wrapper. The upstream script
    defaults to seeds [1, 7, 13].

Output:
    results/milestone1/fgsm.txt  —  full stdout+stderr of the upstream run.

Usage:
    python experiments/milestone1/run_fgsm.py
    python experiments/milestone1/run_fgsm.py --dry-run
    python experiments/milestone1/run_fgsm.py --quick
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
_RESULT_PATH  = _PROJECT_ROOT / "results" / "milestone1" / "fgsm.txt"


def build_cmd(eval_script: pathlib.Path, quick: bool = False) -> list[str]:
    """
    Build the subprocess command for the FGSM evaluation.

    The 'proj' sub-experiment in revision_experiments.py runs unconstrained
    PGD (seq_pgd at the module-level PGD constant) AND projected PGD. To
    isolate the FGSM signal at the authors' default epsilon we use the
    'tuning' sub-experiment which calls seq_fgsm(m, Xc, yv, EPS) directly
    using the module-level EPS=0.1 constant.

    No epsilon argument is passed on the command line — the upstream constant
    governs the attack magnitude entirely. This satisfies the Milestone 1
    constraint that no researcher-chosen attack parameter may be injected.
    """
    cmd = [
        sys.executable,
        str(eval_script.relative_to(_UPSTREAM)),
        "--exp", "tuning",
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
        print("Note: FGSM epsilon is NOT specified on the command line.")
        print("      The upstream module-level constant EPS=0.1 governs the attack.")
        sys.exit(0)

    print(f"Running FGSM evaluation -> {_RESULT_PATH}")
    print(f"Command: {' '.join(cmd)}")
    print(f"CWD    : {_UPSTREAM}")
    print()

    rc = tee_run(cmd=cmd, result_path=_RESULT_PATH, cwd=_UPSTREAM)
    sys.exit(rc)


if __name__ == "__main__":
    main()
