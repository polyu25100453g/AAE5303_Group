#!/usr/bin/env bash
# AAE5303 Assignment 2 — cleanup to avoid disk bloat and conflicts.
# Run from repo root: ./scripts/cleanup.sh
# In Docker, extracted_data and ground_truth are under /data; on host we only remove local dirs.

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

echo "Cleaning under: $REPO_ROOT"

# Local output and evo results (if created by Docker as root, run: sudo rm -rf output)
if [ -d "${REPO_ROOT}/output" ]; then
  rm -rf "${REPO_ROOT}/output" 2>/dev/null || echo "Hint: permission denied — run: sudo rm -rf ${REPO_ROOT}/output"
fi
echo "Cleaned ${REPO_ROOT}/output (or see hint above)"

# Extracted image dir (if you extracted on host instead of in container)
if [ -d "${REPO_ROOT}/extracted_data" ]; then
  rm -rf "${REPO_ROOT}/extracted_data"
  echo "Removed ${REPO_ROOT}/extracted_data"
fi

# Optional: cleanup inside Docker /data (run this *inside* container if needed)
# rm -rf /data/extracted_data /data/ground_truth.txt
echo "Done. To clean Docker /data, run inside container: rm -rf /data/extracted_data /data/ground_truth.txt"
