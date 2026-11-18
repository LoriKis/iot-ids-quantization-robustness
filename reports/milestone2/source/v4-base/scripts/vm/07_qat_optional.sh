#!/usr/bin/env bash
# Stage 7 (optional): QAT arms and the matched FP32 fine-tuning control, validation phase only.
# Usage: bash scripts/vm/07_qat_optional.sh --i-want-qat
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [[ "${1:-}" != "--i-want-qat" ]]; then
  echo "QAT is opt-in. Re-run with: bash scripts/vm/07_qat_optional.sh --i-want-qat" >&2
  exit 2
fi
log 07
clear_marker 07
require_marker 06
require_venv
resolve_workers

M2_ENABLE_QAT=yes "$PY" -m research.m2b_campaign run --run-name "$RUN_NAME" --phase validation --seeds 1 7 13 29 \
  --stages qat --opt-in qat --workers "$WORKERS"
"$PY" -m research.m2b_campaign status --run-name "$RUN_NAME" --phase validation
mark_ok 07
