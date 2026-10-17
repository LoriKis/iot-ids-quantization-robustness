#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Milestone 1 Full Pipeline Orchestration
#
# Runs the complete Milestone 1 pipeline in dependency order:
#   1. Environment bootstrap  (setup_environment.sh)
#   2. Artifact discovery     (discover_artifacts.py)
#   3. Clean evaluation       (run_clean.py)
#   4. FGSM evaluation        (run_fgsm.py)
#   5. PGD evaluation         (run_pgd.py)
#   6. Upstream integrity     (check_upstream_clean.sh)
#
# Usage:
#   bash experiments/milestone1/run_all.sh
#   bash experiments/milestone1/run_all.sh --quick      # smoke test mode
#   bash experiments/milestone1/run_all.sh --skip-setup # skip env bootstrap
#
# Environment:
#   Activate the venv before calling this script, OR let setup_environment.sh
#   create it first (default).  The Python wrappers use the active interpreter.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS="$PROJECT_ROOT/results/milestone1"
QUICK=""
SKIP_SETUP=0

for arg in "$@"; do
  [[ "$arg" == "--quick"      ]] && QUICK="--quick"
  [[ "$arg" == "--skip-setup" ]] && SKIP_SETUP=1
done

echo "================================================================"
echo " PAVE Milestone 1 — Full Pipeline"
echo " Project root : $PROJECT_ROOT"
echo " Quick mode   : ${QUICK:-no}"
echo "================================================================"
echo ""

T0=$(date +%s)

step() {
  local n="$1"; local label="$2"
  echo "──────────────────────────────────────────────────────────────"
  echo " Step $n: $label"
  echo "──────────────────────────────────────────────────────────────"
}

# ── Step 1: Environment bootstrap ───────────────────────────────────────────
if [[ $SKIP_SETUP -eq 0 ]]; then
  step 1 "Environment bootstrap"
  bash "$SCRIPT_DIR/setup_environment.sh"
else
  echo "Step 1: Environment bootstrap — SKIPPED (--skip-setup)"
fi
echo ""

# ── Step 2: Artifact discovery ───────────────────────────────────────────────
step 2 "Artifact discovery"
python "$SCRIPT_DIR/discover_artifacts.py"
echo ""

# ── Step 3: Clean evaluation ─────────────────────────────────────────────────
step 3 "Clean evaluation"
python "$SCRIPT_DIR/run_clean.py" $QUICK
echo ""

# ── Step 4: FGSM evaluation ──────────────────────────────────────────────────
step 4 "FGSM evaluation"
python "$SCRIPT_DIR/run_fgsm.py" $QUICK
echo ""

# ── Step 5: PGD evaluation ───────────────────────────────────────────────────
step 5 "PGD evaluation"
python "$SCRIPT_DIR/run_pgd.py" $QUICK
echo ""

# ── Step 6: Upstream integrity check ─────────────────────────────────────────
step 6 "Upstream integrity check"
bash "$SCRIPT_DIR/check_upstream_clean.sh"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
T1=$(date +%s)
ELAPSED=$(( T1 - T0 ))
echo "================================================================"
echo " Pipeline complete in ${ELAPSED}s."
echo ""
echo " Result files:"
for f in "$RESULTS/clean.txt" "$RESULTS/fgsm.txt" "$RESULTS/pgd.txt"; do
  if [[ -f "$f" ]]; then
    SIZE=$(wc -c < "$f")
    echo "   $(basename $f)   ${SIZE} bytes"
  else
    echo "   $(basename $f)   MISSING"
  fi
done
echo "================================================================"
