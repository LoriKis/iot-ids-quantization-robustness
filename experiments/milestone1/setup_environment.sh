#!/usr/bin/env bash
# =============================================================================
# setup_environment.sh — Milestone 1 Environment Bootstrap
#
# Creates an isolated venv (or conda env if upstream environment.yml exists),
# installs all PAVE dependencies, and captures provenance into notes/.
#
# Usage:
#   bash experiments/milestone1/setup_environment.sh [--force]
#
# --force   : delete and recreate the venv if it already exists.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPSTREAM="$PROJECT_ROOT/upstream/PAVE_artifact"
NOTES="$PROJECT_ROOT/notes"
VENV_NAME=".venv_pave_m1"
VENV_PATH="$PROJECT_ROOT/$VENV_NAME"
CONDA_ENV_NAME="pave_m1"
FORCE=0

for arg in "$@"; do
  [[ "$arg" == "--force" ]] && FORCE=1
done

echo "================================================================"
echo " PAVE Milestone 1 — Environment Bootstrap"
echo " Project root : $PROJECT_ROOT"
echo " Upstream     : $UPSTREAM"
echo "================================================================"

# ── Verify upstream clone exists ────────────────────────────────────────────
if [[ ! -d "$UPSTREAM/.git" ]]; then
  echo "ERROR: $UPSTREAM does not appear to be a git repository."
  echo "       Run: git clone https://github.com/robotumbel/PAVE_artifact.git $UPSTREAM"
  exit 1
fi

# ── Detect upstream environment spec ────────────────────────────────────────
USE_CONDA=0
if [[ -f "$UPSTREAM/environment.yml" ]]; then
  echo "Detected: upstream/environment.yml — using conda branch."
  USE_CONDA=1
elif [[ -f "$UPSTREAM/requirements.txt" ]]; then
  echo "Detected: upstream/requirements.txt — using venv branch."
else
  echo "No environment.yml or requirements.txt found."
  echo "Falling back to venv with manually specified packages (from README)."
  echo "WARNING: recorded in $NOTES/milestone1_environment.txt" | tee -a "$NOTES/milestone1_environment.txt"
fi

# ── Create environment ───────────────────────────────────────────────────────
if [[ $USE_CONDA -eq 1 ]]; then
  # ── Conda branch ─────────────────────────────────────────────────────────
  if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    if [[ $FORCE -eq 1 ]]; then
      echo "Removing existing conda env '$CONDA_ENV_NAME'..."
      conda env remove -n "$CONDA_ENV_NAME" -y
    else
      echo "Conda env '$CONDA_ENV_NAME' already exists. Use --force to recreate."
    fi
  fi
  if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "Creating conda env from $UPSTREAM/environment.yml ..."
    conda env create -f "$UPSTREAM/environment.yml" -n "$CONDA_ENV_NAME"
  fi
  PIP="conda run -n $CONDA_ENV_NAME pip"
  PYTHON="conda run -n $CONDA_ENV_NAME python"

else
  # ── venv branch ──────────────────────────────────────────────────────────
  if [[ -d "$VENV_PATH" ]]; then
    if [[ $FORCE -eq 1 ]]; then
      echo "Removing existing venv at $VENV_PATH ..."
      rm -rf "$VENV_PATH"
    else
      echo "venv already exists at $VENV_PATH. Use --force to recreate."
    fi
  fi

  if [[ ! -d "$VENV_PATH" ]]; then
    echo "Creating venv at $VENV_PATH with $(python3.10 --version 2>/dev/null || python3 --version) ..."
    # Prefer python3.10 if available; fall back to python3
    PY_BIN=$(command -v python3.10 2>/dev/null || command -v python3)
    "$PY_BIN" -m venv "$VENV_PATH"
  fi

  PIP="$VENV_PATH/bin/pip"
  PYTHON="$VENV_PATH/bin/python"

  echo "Upgrading pip ..."
  "$PIP" install --upgrade pip --quiet

  # Install from upstream requirements.txt if present, else use README deps
  if [[ -f "$UPSTREAM/requirements.txt" ]]; then
    echo "Installing from upstream/requirements.txt ..."
    "$PIP" install -r "$UPSTREAM/requirements.txt"
  else
    echo "Installing packages listed in upstream README ..."
    "$PIP" install torch numpy pandas scikit-learn scipy matplotlib
  fi
fi

# ── Capture provenance ───────────────────────────────────────────────────────
echo ""
echo "Capturing provenance ..."

COMMIT=$(git -C "$UPSTREAM" rev-parse HEAD)
PY_VERSION=$($PYTHON --version 2>&1)

# Overwrite environment.txt with final values
cat > "$NOTES/milestone1_environment.txt" <<EOF
PAVE Artifact — Environment Provenance
=======================================

Git commit hash (upstream/PAVE_artifact):
$COMMIT

Python interpreter:
$PY_VERSION

Clone URL:
https://github.com/robotumbel/PAVE_artifact.git

Environment type:
$(if [[ $USE_CONDA -eq 1 ]]; then echo "conda ($CONDA_ENV_NAME)"; else echo "venv ($VENV_NAME)"; fi)

Bootstrap completed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "Writing pip freeze to $NOTES/milestone1_packages.txt ..."
$PIP freeze > "$NOTES/milestone1_packages.txt"

echo ""
echo "================================================================"
echo " Bootstrap complete."
echo " Commit  : $COMMIT"
echo " Python  : $PY_VERSION"
echo " Packages: $NOTES/milestone1_packages.txt"
echo "================================================================"
