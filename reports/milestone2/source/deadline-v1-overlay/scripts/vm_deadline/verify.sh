#!/usr/bin/env bash
# Verify the overlay files and V4 with the system python3 (stdlib only; runs before the venv is used).
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

PYTHONPATH="$O" run_logged verify python3 -B -m m2_deadline.verify_overlay --overlay "$O" --v4-root "$V4_ROOT"
