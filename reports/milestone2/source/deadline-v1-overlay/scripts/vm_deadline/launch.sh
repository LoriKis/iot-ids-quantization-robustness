#!/usr/bin/env bash
# Start the locked campaign. Needs timing/budget_lock.json from timing.sh and the hours left right now.
# Usage: nohup bash scripts/vm_deadline/launch.sh <HOURS_REMAINING> > artifacts/m2-deadline-v1/logs/launch.nohup.out 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
require_hours "${1:-}"
require_venv

H="$1"
shift
run_logged launch-wrapper "$PY" -m m2_deadline.campaign launch --hours-remaining "$H" --v4-root "$V4_ROOT" --out "$OUT" "$@"
