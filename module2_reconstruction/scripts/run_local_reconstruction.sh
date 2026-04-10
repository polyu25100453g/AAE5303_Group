#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPEN_SPLAT_BIN="${ROOT_DIR}/../third_party/OpenSplat/build/opensplat"
MAX_IMAGES="${MAX_IMAGES:-800}"
ITERATIONS="${ITERATIONS:-2000}"
THREADS="${THREADS:-4}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-1280}"
# Higher defaults to improve sparse model density and downstream splat quality.
MAX_FEATURES="${MAX_FEATURES:-4096}"
SEQUENTIAL_OVERLAP="${SEQUENTIAL_OVERLAP:-40}"
START_INDEX="${START_INDEX:-0}"
STRIDE="${STRIDE:-1}"

cd "${ROOT_DIR}"
. .venv/bin/activate

python3 scripts/prepare_inputs.py
python3 scripts/build_colmap_model.py \
  --max-images "${MAX_IMAGES}" \
  --threads "${THREADS}" \
  --max-image-size "${MAX_IMAGE_SIZE}" \
  --max-features "${MAX_FEATURES}" \
  --sequential-overlap "${SEQUENTIAL_OVERLAP}" \
  --start-index "${START_INDEX}" \
  --stride "${STRIDE}" \
  --camera-config "input/camera/camera_config.yaml"

COLMAP_WORKSPACE="${ROOT_DIR}/input/colmap_project"
RUN_OUT="${ROOT_DIR}/output/local_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_OUT}"

if [[ -x "${OPEN_SPLAT_BIN}" ]]; then
  echo "Found opensplat binary: ${OPEN_SPLAT_BIN}"
  echo "Running OpenSplat on ${COLMAP_WORKSPACE}"
  # OOM (exit 137) fallback: retry with fewer iterations until scene.splat exists.
  OPENSPLAT_FALLBACK="${OPENSPLAT_FALLBACK:-${ITERATIONS} 800 500 300 200 100}"
  set +e
  for n in ${OPENSPLAT_FALLBACK}; do
    echo "=== OpenSplat attempt: -n ${n} ==="
    rm -f "${RUN_OUT}/scene.splat"
    LOG="${RUN_OUT}/opensplat_n${n}.log"
    "${OPEN_SPLAT_BIN}" "${COLMAP_WORKSPACE}" -n "${n}" -o "${RUN_OUT}/scene.splat" 2>&1 | tee "${LOG}"
    ec="${PIPESTATUS[0]}"
    if [[ -f "${RUN_OUT}/scene.splat" ]]; then
      echo "OpenSplat done: ${RUN_OUT}/scene.splat (iterations=${n}, exit=${ec})"
      break
    fi
    echo "WARN: no scene.splat (exit=${ec}). Retrying with lower -n if any left..."
  done
  set -e
  if [[ ! -f "${RUN_OUT}/scene.splat" ]]; then
    echo "ERROR: OpenSplat did not produce scene.splat. COLMAP model: ${COLMAP_WORKSPACE}"
    echo "You can retry training only: bash scripts/train_opensplat_only.sh"
    exit 1
  fi
else
  echo "OpenSplat binary not found at ${OPEN_SPLAT_BIN}"
  echo "COLMAP sparse model is ready at: ${COLMAP_WORKSPACE}"
  echo "Next: build OpenSplat binary, then rerun this script."
fi
