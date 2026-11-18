#!/usr/bin/env bash
# Stage 5: seed-1 validation only. Usage: 05_seed1_validation.sh [5a|5b]
#   5a (default): calibration, training of V1 and V3, PTQ, clean evaluation, pair check, then status and estimate.
#   5b: step 0 (E4 grid and budget decision) and seed-1 validation attacks, then status and estimate.
# Never runs the test partition, other seeds or QAT. Pending VM verification.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
PART="${1:-5a}"
case "$PART" in
  5a|5b) ;;
  *) echo "usage: bash scripts/vm/05_seed1_validation.sh [5a|5b]" >&2; exit 2 ;;
esac
log "05$PART"
require_marker 04
require_venv
resolve_workers

if [[ "$PART" == "5a" ]]; then
  rm -f "$RUN_MARKERS/05.ok" "$RUN_MARKERS/05a.ok"
  "$PY" -m research.m2b_campaign run --run-name "$RUN_NAME" --phase validation --seeds 1 \
    --stages calib train ptq clean paircheck --workers "$WORKERS"
  "$PY" -m research.m2b_campaign status --run-name "$RUN_NAME" --phase validation --seed 1
  "$PY" -m research.m2b_campaign estimate --run-name "$RUN_NAME" --workers "$WORKERS"
  echo "5a done. Read the estimate above, then run: bash scripts/vm/05_seed1_validation.sh 5b"
  mark_ok 05a
else
  require_marker 05a
  rm -f "$RUN_MARKERS/05.ok" "$RUN_MARKERS/05b.ok"
  "$PY" -m research.m2b_campaign run --run-name "$RUN_NAME" --phase validation --seeds 1 \
    --stages step0 attack --workers "$WORKERS"
  "$PY" -m research.m2b_campaign status --run-name "$RUN_NAME" --phase validation --seed 1
  "$PY" -m research.m2b_campaign estimate --run-name "$RUN_NAME" --workers "$WORKERS"
  mark_ok 05b
  mark_ok 05
fi