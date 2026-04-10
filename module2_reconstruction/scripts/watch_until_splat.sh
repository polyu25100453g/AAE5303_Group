#!/usr/bin/env bash
# Watch a run directory: if pipeline dies without scene.splat, retry OpenSplat only.
# Usage: RUN_OUT=/path/to/output/local_YYYYMMDD_HHMMSS bash scripts/watch_until_splat.sh
# Optional: COLMAP_WORKSPACE=... (default: input/colmap_project)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_OUT="${RUN_OUT:?Set RUN_OUT to output/local_... directory}"
COLMAP_WORKSPACE="${COLMAP_WORKSPACE:-${ROOT_DIR}/input/colmap_project}"
INTERVAL_SEC="${INTERVAL_SEC:-120}"

cd "${ROOT_DIR}"
. .venv/bin/activate

echo "Watching ${RUN_OUT} for scene.splat (interval ${INTERVAL_SEC}s)"

while true; do
  if [[ -f "${RUN_OUT}/scene.splat" ]]; then
    echo "OK: ${RUN_OUT}/scene.splat exists — exiting watcher."
    exit 0
  fi

  if pgrep -f "opensplat.*${RUN_OUT}" >/dev/null 2>&1 || pgrep -f "run_local_reconstruction" >/dev/null 2>&1; then
    sleep "${INTERVAL_SEC}"
    continue
  fi

  echo "WARN: no scene.splat and no pipeline process — retrying OpenSplat into ${RUN_OUT}"
  mkdir -p "${RUN_OUT}"
  if bash "${ROOT_DIR}/scripts/train_opensplat_only.sh" "${COLMAP_WORKSPACE}" "${RUN_OUT}"; then
    echo "OK: recovered ${RUN_OUT}/scene.splat"
    exit 0
  fi

  echo "Retry failed; sleeping ${INTERVAL_SEC}s before next check..."
  sleep "${INTERVAL_SEC}"
done
