#!/usr/bin/env bash
# Stage 1: verify the transferred folder. Stdlib and coreutils only; no venv, no torch. Pending VM verification.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
log 01
clear_marker 01

echo "== sha256sum -c MANIFEST.sha256"
sha256sum -c --quiet MANIFEST.sha256
echo "== package checks (file set, symlinks, line endings, dataset, PAVE tree)"
python3 scripts/vm/verify_package.py
echo "== bash -n scripts/vm/*.sh"
for f in scripts/vm/*.sh; do bash -n "$f"; done
echo "Stage 01 passed."
mark_ok 01