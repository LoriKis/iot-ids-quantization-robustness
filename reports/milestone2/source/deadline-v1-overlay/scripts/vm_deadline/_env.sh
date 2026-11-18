#!/usr/bin/env bash
# Sourced by every wrapper in scripts/vm_deadline (not an entry point). Wrappers write no markers and nothing under V4.
set -euo pipefail

O="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
V4_ROOT="${V4_ROOT:-$HOME/IOT-network-traffic-DL-M2-VM-v4}"
PY="$V4_ROOT/.venv/bin/python"
OUT="$O/artifacts/m2-deadline-v1"

export PYTHONPATH="$V4_ROOT:$O"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="$OUT/.mplconfig"
export V4_ROOT
# The code sets per-stage thread counts itself.
unset OMP_NUM_THREADS MKL_NUM_THREADS

mkdir -p "$OUT/logs" "$MPLCONFIGDIR"
cd "$O"

# require_venv: exit 2 when the V4 environment is missing.
require_venv() {
  if [[ ! -x "$PY" ]]; then
    echo "V4 Python not found at $PY. Set V4_ROOT to the verified V4 folder (default: ~/IOT-network-traffic-DL-M2-VM-v4)." >&2
    exit 2
  fi
}

# require_hours <value>: hours remaining is mandatory for timing, launch and resume.
require_hours() {
  if [[ -z "${1:-}" || "${1:-}" == --* ]]; then
    echo "Usage: bash scripts/vm_deadline/$(basename "$0") <HOURS_REMAINING> (the actual hours left until the deadline, 0 < H <= 12)" >&2
    exit 2
  fi
}

# run_logged NAME CMD...: tee output to $OUT/logs/NAME-<utc>.log and return the command's exit status.
run_logged() {
  local name="$1" log rc=0
  shift
  log="$OUT/logs/${name}-$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "[$name] log: $log"
  "$@" 2>&1 | tee -a "$log" || rc="${PIPESTATUS[0]}"
  return "$rc"
}
