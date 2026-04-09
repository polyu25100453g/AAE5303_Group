#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for AAE5303 Assignment 2 inside the Docker container.
# Steps:
# 1) Extract images from rosbag to /data/extracted_data
# 2) Extract RTK ground truth to /data/ground_truth.txt
# 3) Generate ORB-SLAM3 camera config DJI_Camera.yaml from docs/camera_config.yaml
# 4) Run ORB-SLAM3 monocular (mono_tum) to produce CameraTrajectory.txt
# 5) Evaluate with evo (ATE, RPE, Completeness) using scripts/evaluate_vo_accuracy.py
# 6) Auto-fill leaderboard/submission_template.json → submission_filled.json
# 7) Generate trajectory plot with evo_traj
# 8) Generate Markdown report at output/report.md

WORKSPACE="/workspace"

BAG_PATH="${1:-/data/AMtown02.bag}"
EXTRACTED_DIR="${2:-/data/extracted_data}"
GT_PATH="${3:-/data/ground_truth.txt}"
OUTPUT_DIR="${4:-${WORKSPACE}/output}"
EVO_WORKDIR="${OUTPUT_DIR}/evo_results"
DATASET_NAME="$(basename "${BAG_PATH}" .bag)"

echo "Using BAG_PATH=${BAG_PATH}"
echo "Using EXTRACTED_DIR=${EXTRACTED_DIR}"
echo "Using GT_PATH=${GT_PATH}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"

mkdir -p "${EXTRACTED_DIR}" "${OUTPUT_DIR}" "${EVO_WORKDIR}"

if [ -f "/opt/ros/noetic/setup.bash" ]; then
  set +u
  # ShellCheck disable=SC1091
  source /opt/ros/noetic/setup.bash
  set -u
fi

echo "================================================================================"
echo "1) Extract images from rosbag to ${EXTRACTED_DIR}"
echo "================================================================================"
python3 "${WORKSPACE}/extract_images_final.py" \
  "${BAG_PATH}" \
  --output "${EXTRACTED_DIR}"

echo "================================================================================"
echo "2) Extract RTK ground truth to ${GT_PATH}"
echo "================================================================================"
python3 "${WORKSPACE}/extract_rtk_groundtruth.py" \
  "${BAG_PATH}" \
  --output "${GT_PATH}"

# Also copy ground truth into OUTPUT_DIR so it is available on the host filesystem
cp "${GT_PATH}" "${OUTPUT_DIR}/ground_truth.txt"

echo "================================================================================"
echo "3) Prepare ORB-SLAM3 camera config DJI_Camera.yaml"
echo "================================================================================"
ORB_ROOT="/root/ORB_SLAM3"
mkdir -p "${ORB_ROOT}/Examples/Monocular"
cp "${WORKSPACE}/docs/camera_config.yaml" \
   "${ORB_ROOT}/Examples/Monocular/DJI_Camera.yaml"

echo "Camera config written to ${ORB_ROOT}/Examples/Monocular/DJI_Camera.yaml"

echo "================================================================================"
echo "4) Ensure headless Pangolin + force CameraTrajectory output"
echo "================================================================================"
export DEBIAN_FRONTEND=noninteractive
if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "Installing Xvfb (required for headless Pangolin on WSL/Docker)..."
  apt-get update -y
  apt-get install -y xvfb
fi

# This Docker image ships a modified mono_tum.cc that only saves KeyFrameTrajectory.
# Patch it to also save full-frame CameraTrajectory (EuRoC/TUM-compatible), then rebuild.
MONO_SRC="${ORB_ROOT}/Examples/Monocular/mono_tum.cc"
if [ -f "${MONO_SRC}" ]; then
  echo "Patching mono_tum.cc to save full-frame CameraTrajectory (EuRoC) plus KeyFrameTrajectory..."
  python3 - <<'PY'
from pathlib import Path
mono_path = Path("/root/ORB_SLAM3/Examples/Monocular/mono_tum.cc")
text = mono_path.read_text(encoding="utf-8", errors="ignore")

target_block = 'SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");\n    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");'
if target_block in text:
    print("mono_tum.cc already patched for SaveTrajectoryEuRoC + KeyFrame; skipping.")
else:
    # First, if an earlier SaveTrajectoryTUM patch exists, remove it.
    if 'SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");' in text:
        text = text.replace(
            'SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");\n    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");',
            'SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");\n    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");'
        )
        print("Replaced SaveTrajectoryTUM with SaveTrajectoryEuRoC in mono_tum.cc.")
    else:
        needle = 'SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");'
        if needle not in text:
            raise SystemExit("Could not find SaveKeyFrameTrajectoryTUM line to patch in mono_tum.cc.")
        text = text.replace(
            needle,
            'SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");\n    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");'
        )
        print("Inserted SaveTrajectoryEuRoC + SaveKeyFrameTrajectoryTUM into mono_tum.cc.")

    mono_path.write_text(text, encoding="utf-8")
    print("mono_tum.cc patched successfully.")
PY
  echo "Rebuilding ORB-SLAM3 (to update mono_tum binary)..."
  cd "${ORB_ROOT}/build"
  make -j"$(nproc)"
else
  echo "ERROR: mono_tum.cc not found at ${MONO_SRC}"
  exit 1
fi

echo "================================================================================"
echo "5) Run ORB-SLAM3 monocular (mono_tum) and save trajectories"
echo "================================================================================"
cd "${ORB_ROOT}"
rm -f "${ORB_ROOT}/CameraTrajectory.txt" "${ORB_ROOT}/KeyFrameTrajectory.txt"

set +e
xvfb-run -a ./Examples/Monocular/mono_tum \
  Vocabulary/ORBvoc.txt \
  Examples/Monocular/DJI_Camera.yaml \
  "${EXTRACTED_DIR}"
ORB_EXIT=$?
set -e

if [ "${ORB_EXIT}" -ne 0 ]; then
  echo "WARNING: ORB-SLAM3 exited with code ${ORB_EXIT}. Will still check trajectory files."
fi

if [ ! -f "${ORB_ROOT}/CameraTrajectory.txt" ]; then
  echo "ERROR: CameraTrajectory.txt not generated. Refusing to evaluate with KeyFrame-only trajectory."
  echo "Check that mono_tum.cc now calls SLAM.SaveTrajectoryEuRoC(\"CameraTrajectory.txt\"); and rebuild succeeded."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mv -f "${ORB_ROOT}/CameraTrajectory.txt" "${OUTPUT_DIR}/CameraTrajectory.txt"
if [ -f "${ORB_ROOT}/KeyFrameTrajectory.txt" ]; then
  mv -f "${ORB_ROOT}/KeyFrameTrajectory.txt" "${OUTPUT_DIR}/KeyFrameTrajectory.txt"
fi
echo "Saved trajectories to ${OUTPUT_DIR}/CameraTrajectory.txt and ${OUTPUT_DIR}/KeyFrameTrajectory.txt"

echo "================================================================================"
echo "6) Convert EuRoC-style trajectory timestamps (ns) to TUM seconds"
echo "================================================================================"
python3 - << 'PY'
from pathlib import Path

# Inside this container pipeline, OUTPUT_DIR is always /workspace/output.
out_dir = Path("/workspace/output")
src = out_dir / "CameraTrajectory.txt"
dst = out_dir / "CameraTrajectory_sec.txt"

if not src.exists():
    raise SystemExit(f"Source trajectory not found: {src}")

with src.open("r", encoding="utf-8", errors="ignore") as f_in, dst.open(
    "w", encoding="utf-8"
) as f_out:
    for line in f_in:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            t_raw = float(parts[0])
        except ValueError:
            continue

        # Heuristic: if timestamp looks like nanoseconds (EuRoC), convert to seconds.
        if t_raw > 1e12:
            t_sec = t_raw / 1e9
        else:
            t_sec = t_raw

        parts[0] = f"{t_sec:.6f}"
        f_out.write(" ".join(parts) + "\n")

print(f"Converted trajectory written to: {dst}")
PY

echo "================================================================================"
echo "7) Evaluate trajectory with evo (ATE/RPE/Completeness)"
echo "================================================================================"
python3 "${WORKSPACE}/scripts/evaluate_vo_accuracy.py" \
  --groundtruth "${OUTPUT_DIR}/ground_truth.txt" \
  --estimated "${OUTPUT_DIR}/CameraTrajectory_sec.txt" \
  --t-max-diff 0.1 \
  --delta-m 10.0 \
  --workdir "${EVO_WORKDIR}" \
  --json-out "${OUTPUT_DIR}/evaluation_report.json"

echo "Evaluation report saved to ${OUTPUT_DIR}/evaluation_report.json"

echo "================================================================================"
echo "7) Auto-fill leaderboard submission JSON"
echo "================================================================================"
python3 - "${WORKSPACE}" << 'EOF'
import json
import os
import sys

workspace = sys.argv[1]
eval_path = os.path.join(workspace, "output", "evaluation_report.json")
template_path = os.path.join(workspace, "leaderboard", "submission_template.json")
out_path = os.path.join(workspace, "leaderboard", "submission_filled.json")

with open(eval_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

with open(template_path, "r", encoding="utf-8") as f:
    submission = json.load(f)

for key in ("ate_rmse_m", "rpe_trans_drift_m_per_m", "rpe_rot_drift_deg_per_100m", "completeness_pct"):
    if key in metrics and "metrics" in submission:
        submission["metrics"][key] = metrics[key]

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2, sort_keys=True)

print(f"Filled leaderboard submission written to: {out_path}")
EOF

echo "================================================================================"
echo "8) Generate trajectory plots (2x2: Before/After Alignment, ATE dist, ATE along traj)"
echo "================================================================================"
MPLBACKEND=Agg python3 "${WORKSPACE}/scripts/generate_report_figures.py" \
  --gt "${OUTPUT_DIR}/ground_truth.txt" \
  --est "${OUTPUT_DIR}/CameraTrajectory_sec.txt" \
  --out "${OUTPUT_DIR}/trajectory.png" \
  --t-max-diff 0.1 \
  --title-suffix "${DATASET_NAME}" \
  --evo-ape-zip "${EVO_WORKDIR}/ate.zip"
echo "Trajectory plot saved to ${OUTPUT_DIR}/trajectory.png"
mkdir -p "${WORKSPACE}/figures"
cp "${OUTPUT_DIR}/trajectory.png" "${WORKSPACE}/figures/trajectory_evaluation.png"
echo "Copied to ${WORKSPACE}/figures/trajectory_evaluation.png (for README)"

echo "================================================================================"
echo "9) Generate Markdown report"
echo "================================================================================"
python3 - "${OUTPUT_DIR}" "${GT_PATH}" "${BAG_PATH}" << 'EOF'
import json
import os
import sys
import textwrap

out_dir, gt_path, bag_path = sys.argv[1], sys.argv[2], sys.argv[3]

report_json = os.path.join(out_dir, "evaluation_report.json")
traj_png = os.path.join(out_dir, "trajectory.png")

with open(report_json, "r", encoding="utf-8") as f:
    m = json.load(f)

md = f"""# ORB-SLAM3 Evaluation Report

- **Dataset**: {os.path.basename(bag_path)}
- **Ground truth**: `{os.path.basename(gt_path)}`
- **Estimated**: `CameraTrajectory.txt`

## Key Metrics (Sim(3) aligned, t_max_diff = 0.1 s, delta = 10 m)

- **ATE RMSE (m)**: {m['ate_rmse_m']:.6f}
- **RPE trans drift (m/m)**: {m['rpe_trans_drift_m_per_m']:.6f}
- **RPE rot drift (deg/100m)**: {m['rpe_rot_drift_deg_per_100m']:.6f}
- **Completeness (%)**: {m['completeness_pct']:.2f} ({m['matched_poses']} / {m['gt_poses']})

## Trajectory

The following figure is saved by `evo_traj`:

![Trajectory](trajectory.png)

## Configuration

- **Camera config**: `docs/camera_config.yaml` → `Examples/Monocular/DJI_Camera.yaml`
- **ORB-SLAM3 mode**: Monocular (`mono_tum`)
- **Vocabulary**: `Vocabulary/ORBvoc.txt`
- **Image folder**: `/data/extracted_data`
"""

with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
    f.write(textwrap.dedent(md))

print(f"Markdown report written to: {os.path.join(out_dir, 'report.md')}")
EOF

echo "================================================================================"
echo "Pipeline completed successfully."
echo "Outputs:"
echo "  - Trajectory: ${OUTPUT_DIR}/CameraTrajectory.txt"
echo "  - Metrics JSON: ${OUTPUT_DIR}/evaluation_report.json"
echo "  - Leaderboard JSON: ${WORKSPACE}/leaderboard/submission_filled.json"
echo "  - Trajectory plot: ${OUTPUT_DIR}/trajectory.png"
echo "  - Markdown report: ${OUTPUT_DIR}/report.md"

