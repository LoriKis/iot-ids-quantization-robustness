#!/usr/bin/env bash
# Timing estimate only. Measures on train and validation data, prints an estimate, writes the budget lock, starts no training.
# Usage: bash scripts/vm_deadline/timing.sh <HOURS_REMAINING> [extra m2_deadline.timing flags]
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
require_hours "${1:-}"
require_venv

H="$1"
shift
run_logged timing-wrapper "$PY" -m m2_deadline.timing --hours-remaining "$H" --v4-root "$V4_ROOT" --out "$OUT" "$@"
