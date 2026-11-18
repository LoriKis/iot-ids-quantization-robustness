#!/usr/bin/env bash
# Continue an interrupted run at the first missing or invalid stage. H is the hours left now, not at launch.
# Usage: nohup bash scripts/vm_deadline/resume.sh <HOURS_REMAINING_NOW> > artifacts/m2-deadline-v1/logs/resume.nohup.out 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
require_hours "${1:-}"
require_venv

H="$1"
shift
run_logged resume-wrapper "$PY" -m m2_deadline.campaign resume --hours-remaining "$H" --v4-root "$V4_ROOT" --out "$OUT" "$@"
