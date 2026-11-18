#!/usr/bin/env bash
# Regenerate results/, report/ and the figures from the result files under OUT. Loads no model.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
require_venv

run_logged report-wrapper "$PY" -m m2_deadline.report --out "$OUT" --v4-root "$V4_ROOT" "$@"
