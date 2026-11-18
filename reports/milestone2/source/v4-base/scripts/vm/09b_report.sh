#!/usr/bin/env bash
# Stage 9b: tables, robustness curves and the efficiency report. Test report only when stage 8b finished.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
log 09b
clear_marker 09b
require_marker 06
require_venv

"$PY" -m research.m2b_campaign report --run-name "$RUN_NAME" --phase validation
if [[ -f "$RUN_MARKERS/09a.ok" ]]; then
  "$PY" -m research.m2b_campaign report --run-name "$RUN_NAME" --phase efficiency
else
  echo "efficiency report skipped: stage 09a not completed"
fi
if [[ -f "$RUN_MARKERS/08b.ok" ]]; then
  "$PY" -m research.m2b_campaign report --run-name "$RUN_NAME" --phase test --freeze-name eval_freeze
else
  echo "test report skipped: stage 08b not completed"
fi
mark_ok 09b
