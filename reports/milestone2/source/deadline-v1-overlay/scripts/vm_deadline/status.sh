#!/usr/bin/env bash
# Print stage markers, last log line and elapsed times. Read-only.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
require_venv

"$PY" -m m2_deadline.campaign status --v4-root "$V4_ROOT" --out "$OUT" "$@"
