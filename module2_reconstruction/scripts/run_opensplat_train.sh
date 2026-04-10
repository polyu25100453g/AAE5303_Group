#!/usr/bin/env bash
set -euo pipefail

# Run inside container at /workspace/module2_reconstruction.
# It auto-detects common OpenSplat entrypoints.

ROOT="/workspace/module2_reconstruction"
INPUT_DIR="${ROOT}/input/images"
OUTPUT_DIR="${ROOT}/output/opensplat_run_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: input image dir not found: ${INPUT_DIR}"
  exit 1
fi

PNG_COUNT=$(ls "${INPUT_DIR}"/*.png 2>/dev/null | wc -l || true)
JPG_COUNT=$(ls "${INPUT_DIR}"/*.jpg 2>/dev/null | wc -l || true)
TOTAL=$((PNG_COUNT + JPG_COUNT))
if [[ "${TOTAL}" -le 0 ]]; then
  echo "ERROR: no images found in ${INPUT_DIR}"
  exit 1
fi

echo "Found ${TOTAL} images."
echo "Output dir: ${OUTPUT_DIR}"

run_and_log() {
  local cmd="$1"
  local log_path="${LOG_DIR}/opensplat_$(date +%Y%m%d_%H%M%S).log"
  echo "Running: ${cmd}"
  bash -lc "${cmd}" 2>&1 | tee "${log_path}"
  echo "Log saved: ${log_path}"
}

# Highest priority: user-provided command template.
# You can pass OPENSPLAT_CMD, and use {input} {output} placeholders.
if [[ -n "${OPENSPLAT_CMD:-}" ]]; then
  CMD="${OPENSPLAT_CMD//\{input\}/${INPUT_DIR}}"
  CMD="${CMD//\{output\}/${OUTPUT_DIR}}"
  run_and_log "${CMD}"
  echo "Done with OPENSPLAT_CMD."
  exit 0
fi

# Auto-detect common entrypoints.
if command -v opensplat >/dev/null 2>&1; then
  run_and_log "opensplat train --data \"${INPUT_DIR}\" --output \"${OUTPUT_DIR}\""
  exit 0
fi

if [[ -f "/workspace/OpenSplat/train.py" ]]; then
  run_and_log "python3 /workspace/OpenSplat/train.py --data \"${INPUT_DIR}\" --output \"${OUTPUT_DIR}\""
  exit 0
fi

if [[ -f "/workspace/OpenSplat/main.py" ]]; then
  run_and_log "python3 /workspace/OpenSplat/main.py --data \"${INPUT_DIR}\" --output \"${OUTPUT_DIR}\""
  exit 0
fi

echo "ERROR: Could not detect OpenSplat entrypoint."
echo "Please rerun with explicit command template, e.g.:"
echo "  OPENSPLAT_CMD='opensplat train --data {input} --output {output}' bash scripts/docker_run_opensplat.sh --auto-train"
exit 2
