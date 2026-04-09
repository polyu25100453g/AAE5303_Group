#!/usr/bin/env bash
# AAE5303 Assignment 2 — WSL/Windows: run ORB-SLAM3 pipeline in Docker (ROS1)
# Usage: from repo root, ./scripts/docker_run_hkisland.sh
# Requires: bag at /mnt/d/HKisland_GNSS03.bag (WSL) or D:\HKisland_GNSS03.bag (Windows path in WSL = /mnt/d/...)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Bag path: WSL uses /mnt/d/; adjust if your bag is elsewhere
BAG_HOST="${BAG_PATH:-/mnt/d/HKisland_GNSS03.bag}"
BAG_IN_CONTAINER="/data/HKisland_GNSS03.bag"

IMAGE="${DOCKER_IMAGE:-liangyu99/orbslam3_ros1:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-orbslam3_hkisland}"

if [ ! -f "$BAG_HOST" ]; then
  echo "Bag not found at: $BAG_HOST"
  echo "Set BAG_PATH if needed, e.g. BAG_PATH=/path/to/HKisland_GNSS03.bag $0"
  exit 1
fi

# Remove existing container if present (idempotent)
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting container: $CONTAINER_NAME"
docker run -it --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --privileged \
  --network=host \
  -e DISPLAY="${DISPLAY:-host.docker.internal:0.0}" \
  -e QT_X11_NO_MITSHM=1 \
  -v "$BAG_HOST:$BAG_IN_CONTAINER:ro" \
  -v "$REPO_ROOT:/workspace:rw" \
  "$IMAGE" \
  bash -c "
    echo 'Container ready. Running pipeline...'
    cd /workspace
    chmod +x scripts/run_pipeline.sh
    ./scripts/run_pipeline.sh '$BAG_IN_CONTAINER' /data/extracted_data /data/ground_truth.txt /workspace/output
  "

echo "Pipeline finished. Check output/ and leaderboard/submission_filled.json"
