#!/usr/bin/env bash
# Run OpenSplat on an existing COLMAP workspace with OOM-safe fallbacks.
# Usage:
#   bash scripts/train_opensplat_only.sh [colmap_dir] [output_dir]
# Defaults:
#   colmap_dir = input/colmap_project
#   output_dir = output/opensplat_fallback_YYYYMMDD_HHMMSS

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPEN_SPLAT_BIN="${ROOT_DIR}/../third_party/OpenSplat/build/opensplat"
COLMAP_WORKSPACE="${1:-${ROOT_DIR}/input/colmap_project}"
ITERATIONS="${ITERATIONS:-1500}"
# Space-separated list; first success wins
OPENSPLAT_FALLBACK="${OPENSPLAT_FALLBACK:-${ITERATIONS} 800 500 300 200 100}"

cd "${ROOT_DIR}"

if [[ ! -x "${OPEN_SPLAT_BIN}" ]]; then
  echo "ERROR: opensplat not found or not executable: ${OPEN_SPLAT_BIN}"
  exit 1
fi
if [[ ! -d "${COLMAP_WORKSPACE}" ]]; then
  echo "ERROR: COLMAP workspace not found: ${COLMAP_WORKSPACE}"
  exit 1
fi

if [[ -n "${2:-}" ]]; then
  RUN_OUT="$2"
else
  RUN_OUT="${ROOT_DIR}/output/opensplat_fallback_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${RUN_OUT}"

echo "COLMAP: ${COLMAP_WORKSPACE}"
echo "Output: ${RUN_OUT}/scene.splat"
echo "Fallback iterations: ${OPENSPLAT_FALLBACK}"

set +e
for n in ${OPENSPLAT_FALLBACK}; do
  echo ""
  echo "=== OpenSplat attempt: -n ${n} ==="
  rm -f "${RUN_OUT}/scene.splat"
  LOG="${RUN_OUT}/opensplat_n${n}.log"
  "${OPEN_SPLAT_BIN}" "${COLMAP_WORKSPACE}" -n "${n}" -o "${RUN_OUT}/scene.splat" 2>&1 | tee "${LOG}"
  ec="${PIPESTATUS[0]}"
  if [[ -f "${RUN_OUT}/scene.splat" ]]; then
    echo ""
    echo "OK: ${RUN_OUT}/scene.splat (iterations=${n}, exit=${ec})"
    exit 0
  fi
  echo "WARN: no scene.splat (exit=${ec}). Trying lower -n..."
done
set -e

echo ""
echo "ERROR: OpenSplat did not produce scene.splat. See logs in ${RUN_OUT}"
exit 1
