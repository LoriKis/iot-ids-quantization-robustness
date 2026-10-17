#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
CONFIG=configs/full.json
PHASE=validation
for arg in "$@"; do
  case "$arg" in
    --quick) CONFIG=configs/smoke.json; PHASE=both ;;
    --skip-setup) ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
if [[ ! -x .venv/bin/python ]]; then bash scripts/setup.sh; fi
exec .venv/bin/python -m research.runner --config "$CONFIG" --run-dir "runs/milestone1-$(date -u +%Y%m%dT%H%M%SZ)" --phase "$PHASE"
