#!/usr/bin/env bash
# Stage 3: non-training preflight (imports, dataset, PAVE, split, configs, INT8 backend, disk/RAM, writable paths).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
log 03
clear_marker 03
require_marker 02
require_venv

"$PY" -m research.m2b_campaign preflight --run-name "$RUN_NAME"
echo "Stage 03 passed. Recommended workers are in $RUN_DIR/preflight.json."
mark_ok 03