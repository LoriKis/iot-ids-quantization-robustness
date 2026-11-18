#!/usr/bin/env bash
# Stage 9a: serialized size, latency and memory. Run on a quiet machine, alone. Usage: 09a_efficiency.sh [--include-qat]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
EXTRA=()
if [[ "${1:-}" == "--include-qat" ]]; then
  EXTRA=(--include-qat)
elif [[ -n "${1:-}" ]]; then
  echo "usage: bash scripts/vm/09a_efficiency.sh [--include-qat]" >&2
  exit 2
fi
log 09a
clear_marker 09a
require_marker 06
if [[ ${#EXTRA[@]} -gt 0 ]]; then require_marker 08a; fi
require_venv

"$PY" -m research.m2b_campaign efficiency --run-name "$RUN_NAME" --seeds 1 7 13 29 ${EXTRA[@]+"${EXTRA[@]}"}
mark_ok 09a
