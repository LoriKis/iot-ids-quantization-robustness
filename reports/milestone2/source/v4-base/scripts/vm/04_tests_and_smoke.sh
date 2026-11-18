#!/usr/bin/env bash
# Stage 4: full regression suite, then the bounded smoke test (train and validation subsets only; no test partition).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
log 04
clear_marker 04
require_marker 03
require_venv

echo "== pytest (all tests; every skip reason is printed)"
"$PY" -B -m pytest -p no:cacheprovider -q -rs tests
echo "== package file set must be unchanged by the tests"
python3 scripts/vm/verify_package.py --manifest-only
echo "== bounded smoke"
"$PY" -m research.m2b_campaign smoke --run-name "$RUN_NAME" --max-minutes 45
echo "Stage 04 passed."
mark_ok 04