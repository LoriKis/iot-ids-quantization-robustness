#!/usr/bin/env bash
# Sourced by every stage script (not an entry point). Pending VM verification.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT/output/.mplconfig"

RUN_NAME="${RUN_NAME:-m2b-v1}"
if ! [[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]{1,64}$ ]]; then
  echo "RUN_NAME must match ^[A-Za-z0-9._-]{1,64}\$ (got '$RUN_NAME')" >&2
  exit 2
fi
PY=".venv/bin/python"
RUN_DIR="output/$RUN_NAME"
GLOBAL_MARKERS="output/.stage_markers"
RUN_MARKERS="$RUN_DIR/stage_markers"

# Stage 01 and 02 markers are global (they precede any run directory); all others belong to the run.
marker_dir() {
  case "$1" in
    01|02) echo "$GLOBAL_MARKERS" ;;
    *) echo "$RUN_MARKERS" ;;
  esac
}

# log <stage>: mirror stdout and stderr of the rest of the script into output/<run>/logs/<stage>-<utc>.log
log() {
  local stage="$1"
  mkdir -p "$RUN_DIR/logs" "$MPLCONFIGDIR"
  LOG_FILE="$RUN_DIR/logs/${stage}-$(date -u +%Y%m%dT%H%M%SZ).log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "[$stage] log: $LOG_FILE"
}

# clear_marker <stage>: a failed re-run must not leave an old marker behind.
clear_marker() {
  rm -f "$(marker_dir "$1")/$1.ok"
}

# mark_ok <stage>: LAST line of a stage script; only reached when every earlier command exited 0 (set -e).
mark_ok() {
  local dir
  dir="$(marker_dir "$1")"
  mkdir -p "$dir"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) exit=0" > "$dir/$1.ok"
  echo "[$1] marker written: $dir/$1.ok"
}

# require_marker <stage>...: exit 2 naming the first stage that has not finished.
require_marker() {
  local stage
  for stage in "$@"; do
    if [[ ! -f "$(marker_dir "$stage")/$stage.ok" ]]; then
      echo "Stage $stage has not completed (missing $(marker_dir "$stage")/$stage.ok). Run it first." >&2
      exit 2
    fi
  done
}

require_venv() {
  if [[ ! -x "$PY" ]]; then
    echo "$PY not found. Run: bash scripts/vm/02_setup_env.sh" >&2
    exit 2
  fi
}

# Worker count: explicit WORKERS wins; else preflight's recommendation; else 1.
resolve_workers() {
  if [[ -z "${WORKERS:-}" ]]; then
    WORKERS="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["recommended_workers"])' "$RUN_DIR/preflight.json" 2>/dev/null || echo 1)"
  fi
  export WORKERS
  echo "workers: $WORKERS (override with WORKERS=N)"
}
