#!/usr/bin/env bash
# Stage 8a: freeze the evaluation configuration. Runs no inference. Usage: 08a_freeze_eval.sh [--include-qat]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
EXTRA=()
if [[ "${1:-}" == "--include-qat" ]]; then
  EXTRA=(--include-qat)
elif [[ -n "${1:-}" ]]; then
  echo "usage: bash scripts/vm/08a_freeze_eval.sh [--include-qat]" >&2
  exit 2
fi
log 08a
clear_marker 08a
require_marker 06
if [[ ${#EXTRA[@]} -gt 0 ]]; then require_marker 07; fi
require_venv

"$PY" -m research.m2b_campaign freeze-eval --run-name "$RUN_NAME" --seeds 1 7 13 29 --freeze-name eval_freeze ${EXTRA[@]+"${EXTRA[@]}"}
echo "Freeze SHA-256 (needed by stage 8b):"
cat "$RUN_DIR/freezes/eval_freeze.json.sha256"
echo
mark_ok 08a
