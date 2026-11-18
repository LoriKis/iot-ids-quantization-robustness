#!/usr/bin/env bash
# Show which units completed. Usage: status.sh [--phase validation|test] [--seed N] [--verify]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_venv
echo "stage markers:"
ls "$GLOBAL_MARKERS" "$RUN_MARKERS" 2>/dev/null || echo "  (none yet)"
"$PY" -m research.m2b_campaign status --run-name "$RUN_NAME" "$@"
