#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if [[ ! -e .venv/bin/python ]]; then "$PYTHON_BIN" -m venv .venv; fi
.venv/bin/python -c "import sys; assert sys.version_info[:2] == (3,11), 'Use Python 3.11'"
.venv/bin/python -m pip install --cache-dir work/pip-cache -r requirements.txt
.venv/bin/python scripts/setup_upstream.py
.venv/bin/python -m pip check
