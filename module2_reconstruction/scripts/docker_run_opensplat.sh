#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-liangyu99/opensplat-cpu:latest}"
WORKDIR="/workspace/module2_reconstruction"
CMD=""
AUTO_TRAIN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cmd)
      CMD="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --auto-train)
      AUTO_TRAIN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MODULE2_DIR}/.." && pwd)"

python3 "${SCRIPT_DIR}/prepare_inputs.py"

echo "Using docker image: ${IMAGE}"
echo "Mount repo root: ${REPO_ROOT} -> /workspace"
echo "Tip: you can override image by --image <name> or IMAGE=<name>"

if [[ -n "${CMD}" ]]; then
  docker run --rm -it \
    --name opensplat_module2 \
    -v "${REPO_ROOT}:/workspace:rw" \
    -w "${WORKDIR}" \
    "${IMAGE}" \
    bash -lc "${CMD}"
elif [[ "${AUTO_TRAIN}" -eq 1 ]]; then
  if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Image ${IMAGE} not found locally, trying to pull..."
    if ! docker pull "${IMAGE}"; then
      echo ""
      echo "ERROR: Failed to pull docker image: ${IMAGE}"
      echo "If you see 'unauthorized: email must be verified', fix by:"
      echo "  1) Verify your Docker Hub email"
      echo "  2) docker logout && docker login"
      echo "Or use a local/accessible image:"
      echo "  bash scripts/docker_run_opensplat.sh --auto-train --image <your_image>"
      exit 2
    fi
  fi
  docker run --rm -it \
    --name opensplat_module2 \
    -v "${REPO_ROOT}:/workspace:rw" \
    -w "${WORKDIR}" \
    "${IMAGE}" \
    bash -lc "chmod +x scripts/run_opensplat_train.sh && scripts/run_opensplat_train.sh"
else
  docker run --rm -it \
    --name opensplat_module2 \
    -v "${REPO_ROOT}:/workspace:rw" \
    -w "${WORKDIR}" \
    "${IMAGE}" \
    bash
fi
