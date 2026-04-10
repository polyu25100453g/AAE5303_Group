#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOD3_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# If INPUT_DIR is unset or empty, use the first existing candidate (module2 symlink after prepare_inputs, then module1 paths).
if [ -z "${INPUT_DIR:-}" ]; then
  for cand in \
    "${MOD3_ROOT}/../module2_reconstruction/input/images" \
    "${MOD3_ROOT}/../module1_vo/extracted_data" \
    "${MOD3_ROOT}/../module1_vo/dataset/AMtown02/images" \
    "${MOD3_ROOT}/../module1_vo/dataset/HKisland_GNSS03/images"; do
    if [ -d "$cand" ]; then
      INPUT_DIR="$(cd "$cand" && pwd)"
      break
    fi
  done
fi
if [ -z "${INPUT_DIR:-}" ]; then
  echo "No image directory found. Set INPUT_DIR, or run module1 Docker (saves to module1_vo/extracted_data) and/or:" >&2
  echo "  cd module2_reconstruction && python3 scripts/prepare_inputs.py --images \"<dir>\"" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${MOD3_ROOT}/output}"
MAX_IMAGES="${MAX_IMAGES:-200}"

echo "=== Module 3: Semantic Segmentation ==="
echo "INPUT_DIR=${INPUT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_IMAGES=${MAX_IMAGES}"

python3 "${MOD3_ROOT}/scripts/infer_segmentation.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-images "${MAX_IMAGES}"

echo "Done. See:"
echo "  ${OUTPUT_DIR}/masks"
echo "  ${OUTPUT_DIR}/color_masks"
echo "  ${OUTPUT_DIR}/overlays"
echo "  ${OUTPUT_DIR}/summary.json"
