#!/usr/bin/env bash
# Use more aggressive ORB config (fallback) then run the same pipeline.
# Call like run_pipeline.sh: ./scripts/run_pipeline_fallback.sh /data/AMtown02.bag /data/extracted_data /data/ground_truth.txt /workspace/output

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Backup current camera config and switch to fallback
if [ -f "${REPO_ROOT}/docs/camera_config.yaml" ]; then
  cp "${REPO_ROOT}/docs/camera_config.yaml" "${REPO_ROOT}/docs/camera_config.yaml.bak"
fi
cp "${REPO_ROOT}/docs/camera_config_mono_fallback.yaml" "${REPO_ROOT}/docs/camera_config.yaml"
echo "Using fallback ORB config (nFeatures=3500, lower FAST thresholds)."

exec "${REPO_ROOT}/scripts/run_pipeline.sh" "$@"
