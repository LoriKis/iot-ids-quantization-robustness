#!/usr/bin/env bash
# Stage 6: validation for seeds 7, 13 and 29. Completed seed-1 units are verified and reused, never rerun.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
log 06
clear_marker 06
require_marker 05
require_venv
resolve_workers

"$PY" -m research.m2b_campaign run --run-name "$RUN_NAME" --phase validation --seeds 1 7 13 29 \
  --workers "$WORKERS"
"$PY" -m research.m2b_campaign status --run-name "$RUN_NAME" --phase validation
mark_ok 06