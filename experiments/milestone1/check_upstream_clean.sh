#!/usr/bin/env bash
# =============================================================================
# check_upstream_clean.sh — Upstream Integrity Check (REQ-NF-01)
#
# Verifies that no file inside upstream/PAVE_artifact/ has been created,
# modified, or deleted by researcher scripts during this milestone.
#
# Exit codes:
#   0   upstream is clean (no changes)
#   1   upstream has uncommitted changes — CONSTRAINT VIOLATION
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPSTREAM="$PROJECT_ROOT/upstream/PAVE_artifact"

echo "================================================================"
echo " Upstream Integrity Check — REQ-NF-01"
echo " Checking: $UPSTREAM"
echo "================================================================"

if [[ ! -d "$UPSTREAM/.git" ]]; then
  echo "ERROR: $UPSTREAM is not a git repository."
  exit 1
fi

STATUS=$(git -C "$UPSTREAM" status --porcelain)

if [[ -n "$STATUS" ]]; then
  echo ""
  echo "CONSTRAINT VIOLATION: upstream/PAVE_artifact has uncommitted changes:"
  echo "$STATUS"
  echo ""
  echo "Milestone 1 requires zero upstream modification (REQ-NF-01)."
  echo "Review the diff with: git -C $UPSTREAM diff"
  exit 1
fi

echo "OK: upstream/PAVE_artifact is clean — no researcher modifications detected."
exit 0
