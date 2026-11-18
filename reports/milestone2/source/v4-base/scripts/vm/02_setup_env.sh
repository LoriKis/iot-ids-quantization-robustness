#!/usr/bin/env bash
# Stage 2: create .venv and install pinned dependencies. Needs internet unless M2_WHEELHOUSE is set. Never trains.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
log 02
clear_marker 02
require_marker 01

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 not found. Install it (for example: sudo apt install python3.11 python3.11-venv)." >&2
  exit 2
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "x86_64 required, found $(uname -m)" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  python3.11 -m venv .venv
fi
if [[ -n "${M2_WHEELHOUSE:-}" ]]; then
  echo "offline install from M2_WHEELHOUSE=$M2_WHEELHOUSE"
  "$PY" -m pip install --disable-pip-version-check --no-index --find-links "$M2_WHEELHOUSE" -r requirements-linux-cpu.txt
else
  "$PY" -m pip install --disable-pip-version-check -r requirements-linux-cpu.txt
fi
"$PY" -m pip check
"$PY" -c "import torch; print('torch', torch.__version__); print('supported_engines', torch.backends.quantized.supported_engines)"
echo "Stage 02 passed."
mark_ok 02