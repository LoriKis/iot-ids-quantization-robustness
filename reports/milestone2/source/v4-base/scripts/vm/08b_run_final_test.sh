#!/usr/bin/env bash
# Stage 8b: final-test evaluation. Needs the freeze SHA-256 printed by stage 8a. Usage: 08b_run_final_test.sh <SHA256>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: bash scripts/vm/08b_run_final_test.sh <64-hex SHA-256 from stage 8a>" >&2
  exit 2
fi
log 08b
clear_marker 08b
require_marker 08a
require_venv
resolve_workers

"$PY" -m research.m2b_campaign run --run-name "$RUN_NAME" --phase test --seeds 1 7 13 29 \
  --freeze-name eval_freeze --confirm-freeze "$1" --workers "$WORKERS"
"$PY" -m research.m2b_campaign status --run-name "$RUN_NAME" --phase test
mark_ok 08b
