# AAE5303 Assignment: Visual Odometry with ORB-SLAM3

<div align="center">

![ORB-SLAM3](https://img.shields.io/badge/SLAM-ORB--SLAM3-blue?style=for-the-badge)
![VO](https://img.shields.io/badge/Mode-Visual_Odometry-green?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-HKisland__GNSS03-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Baseline-success?style=for-the-badge)

**Monocular Visual Odometry Evaluation on UAV Aerial Imagery**

*Hong Kong Island GNSS Dataset - MARS-LVIG*

</div>

---

## 📋 Table of Contents

1. [Executive Summary](#-executive-summary)
2. [Introduction](#-introduction)
3. [Methodology](#-methodology)
4. [Dataset Description](#-dataset-description)
5. [Implementation Details](#-implementation-details)
6. [Results and Analysis](#-results-and-analysis)
7. [Visualizations](#-visualizations)
8. [Discussion](#-discussion)
9. [Conclusions](#-conclusions)
10. [References](#-references)
11. [Appendix](#-appendix)

---

## 📊 Executive Summary

This report presents the implementation and evaluation of **Monocular Visual Odometry (VO)** using the **ORB-SLAM3** framework on the **HKisland_GNSS03** UAV aerial imagery dataset. The project evaluates trajectory accuracy against RTK ground truth using **four parallel, monocular-appropriate metrics** computed with the `evo` toolkit.

### Key Results

| Metric | Value | Description |
|--------|-------|-------------|
| **ATE RMSE** | **37.14 m** | Global accuracy after Sim(3) alignment (scale corrected) |
| **RPE Trans Drift** | **2.06 m/m** | Translation drift rate (mean error per meter, delta=10 m) |
| **RPE Rot Drift** | **171.56 deg/100m** | Rotation drift rate (mean angle per 100 m, delta=10 m) |
| **Completeness** | **95.09%** | Matched poses / total ground-truth poses (1859 / 1955) |
| **Matched poses** | 1,859 | Pose pairs used for evaluation (from `CameraTrajectory_sec.txt`) |

---

## 📖 Introduction

### Background

ORB-SLAM3 is a state-of-the-art visual SLAM system capable of performing:

- **Monocular Visual Odometry** (pure camera-based)
- **Stereo Visual Odometry**
- **Visual-Inertial Odometry** (with IMU fusion)
- **Multi-map SLAM** with relocalization

This assignment focuses on **Monocular VO mode**, which:

- Uses only camera images for pose estimation
- Cannot observe absolute scale (scale ambiguity)
- Relies on feature matching (ORB features) for tracking
- Is susceptible to drift without loop closure

### Objectives

1. Implement monocular Visual Odometry using ORB-SLAM3
2. Process UAV aerial imagery from the HKisland_GNSS03 dataset
3. Extract RTK (Real-Time Kinematic) GPS data as ground truth
4. Evaluate trajectory accuracy using four parallel metrics appropriate for monocular VO
5. Document the complete workflow for reproducibility

### Scope

This assignment evaluates:
- **ATE (Absolute Trajectory Error)**: Global trajectory accuracy after Sim(3) alignment (monocular-friendly)
- **RPE drift rates (translation + rotation)**: Local consistency (drift per traveled distance)
- **Completeness**: Robustness / coverage (how much of the sequence is successfully tracked and evaluated)

---

## 🔬 Methodology

### ORB-SLAM3 Visual Odometry Overview

ORB-SLAM3 performs visual odometry through the following pipeline:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Input Image    │────▶│   ORB Feature   │────▶│   Feature       │
│  Sequence       │     │   Extraction    │     │   Matching      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌────────▼────────┐
│   Trajectory    │◀────│   Pose          │◀────│   Motion        │
│   Output        │     │   Estimation    │     │   Model         │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                        ┌────────▼────────┐
                        │   Local Map     │
                        │   Optimization  │
                        └─────────────────┘
```

### Evaluation Metrics

#### 1. ATE (Absolute Trajectory Error)

Measures the RMSE of the aligned trajectory after Sim(3) alignment:

$$ATE_{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\|\mathbf{p}_{est}^i - \mathbf{p}_{gt}^i\|^2}$$

**Reference**: Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012

#### 2. RPE (Relative Pose Error) – Drift Rates

Measures local consistency by comparing relative transformations:

$$RPE_{trans} = \|\Delta\mathbf{p}_{est} - \Delta\mathbf{p}_{gt}\|$$

where $\Delta\mathbf{p} = \mathbf{p}(t+\Delta) - \mathbf{p}(t)$

**Reference**: Geiger et al., "Vision meets Robotics: The KITTI Dataset", IJRR 2013

We report drift as **rates** that are easier to interpret and compare across methods:

- **Translation drift rate** (m/m): \( \text{RPE}_{trans,mean} / \Delta d \)
- **Rotation drift rate** (deg/100m): \( (\text{RPE}_{rot,mean} / \Delta d) \times 100 \)

where \(\Delta d\) is a distance interval in meters (e.g., 10 m).

#### 3. Completeness

Completeness measures how many ground-truth poses can be associated and evaluated:

$$Completeness = \frac{N_{matched}}{N_{gt}} \times 100\%$$

#### Why these metrics (and why Sim(3) alignment)?

Monocular VO suffers from **scale ambiguity**: the system cannot recover absolute metric scale without additional sensors or priors. Therefore:

- **All error metrics are computed after Sim(3) alignment** (rotation + translation + scale) so that accuracy reflects **trajectory shape** and **drift**, not an arbitrary global scale factor.
- **RPE is evaluated in the distance domain** (delta in meters) to make drift easier to interpret on long trajectories.
- **Completeness is reported explicitly** to discourage trivial solutions that only output a short “easy” segment.

### Trajectory Alignment

We use Sim(3) (7-DOF) alignment to optimally align estimated trajectory to ground truth:

- **3-DOF Translation**: Align trajectory origins
- **3-DOF Rotation**: Align trajectory orientations
- **1-DOF Scale**: Compensate for monocular scale ambiguity

### Evaluation Protocol (Recommended)

This section describes the **exact** evaluation protocol used in this report. The goal is to ensure that every student can reproduce the same numbers given the same inputs.

#### Inputs

- **Ground truth**: `ground_truth.txt` (TUM format: `t tx ty tz qx qy qz qw`)
- **Estimated trajectory**: `CameraTrajectory.txt` (TUM format)
- **Association threshold**: `t_max_diff = 0.1 s`
  - This dataset contains RTK at ~5 Hz and images at ~10 Hz.
  - A threshold of 0.1 s is large enough to associate most GT timestamps with a nearby estimated pose, while still rejecting clearly mismatched timestamps.
- **Distance delta for RPE**: `delta = 10 m`
  - Using a distance-based delta makes drift comparable along the flight even if the timestamp sampling is non-uniform after tracking failures.

#### Step 1 — ATE with Sim(3) alignment (scale corrected)

```bash
evo_ape tum ground_truth.txt CameraTrajectory.txt \
  --align --correct_scale \
  --t_max_diff 0.1 -va
```

We report **ATE RMSE (m)** as the primary global accuracy metric.

#### Step 2 — RPE (translation + rotation) in the distance domain

```bash
# Translation RPE over 10 m (meters)
evo_rpe tum ground_truth.txt CameraTrajectory.txt \
  --align --correct_scale \
  --t_max_diff 0.1 \
  --delta 10 --delta_unit m \
  --pose_relation trans_part -va

# Rotation RPE over 10 m (degrees)
evo_rpe tum ground_truth.txt CameraTrajectory.txt \
  --align --correct_scale \
  --t_max_diff 0.1 \
  --delta 10 --delta_unit m \
  --pose_relation angle_deg -va
```

We convert evo’s mean RPE over 10 m into drift rates:

- **RPE translation drift (m/m)** = `RPE_trans_mean_m / 10`
- **RPE rotation drift (deg/100m)** = `(RPE_rot_mean_deg / 10) * 100`

#### Step 3 — Completeness

Completeness measures how much of the sequence can be evaluated:

```text
Completeness (%) = matched_poses / gt_poses * 100
```

Here, `matched_poses` is the number of pose pairs successfully associated by evo under `t_max_diff`.

#### Practical Notes (Common Pitfalls)

- **Use the correct trajectory file**:
  - `CameraTrajectory.txt` contains *all tracked frames* and typically yields higher completeness.
  - `KeyFrameTrajectory.txt` contains only keyframes and can severely reduce completeness and distort drift estimates.
- **Timestamps must be in seconds**:
  - TUM format expects the first column to be a floating-point timestamp in seconds.
  - If you accidentally write frame indices as timestamps, `evo` will fail to associate trajectories.
- **Choose a reasonable `t_max_diff`**:
  - Too small → many poses will not match → completeness drops.
  - Too large → wrong matches may slip in → metrics become unreliable.

---

## 📁 Dataset Description

### HKisland_GNSS03 Dataset

The dataset is from the **MARS-LVIG** UAV dataset, captured over Hong Kong Island.

| Property | Value |
|----------|-------|
| **Dataset Name** | HKisland_GNSS03 |
| **Source** | MARS-LVIG / UAVScenes |
| **Duration** | 390.78 seconds (~6.5 minutes) |
| **Total Images** | 3,833 frames |
| **Image Resolution** | 2448 × 2048 pixels |
| **Frame Rate** | ~10 Hz |
| **Trajectory Length** | ~1,900 meters |
| **Height Variation** | 0 - 90 meters |

### Data Sources

| Resource | Link |
|----------|------|
| MARS-LVIG Dataset | https://mars.hku.hk/dataset.html |
| UAVScenes GitHub | https://github.com/sijieaaa/UAVScenes |

### Ground Truth

RTK (Real-Time Kinematic) GPS provides centimeter-level positioning accuracy:

| Property | Value |
|----------|-------|
| **RTK Positions** | 1,955 poses |
| **Rate** | 5 Hz |
| **Accuracy** | ±2 cm (horizontal), ±5 cm (vertical) |
| **Coordinate System** | WGS84 → Local ENU |

---

## ⚙️ Implementation Details

### System Configuration

| Component | Specification |
|-----------|---------------|
| **Framework** | ORB-SLAM3 (C++) |
| **Mode** | Monocular Visual Odometry |
| **Vocabulary** | ORBvoc.txt (pre-trained) |
| **Operating System** | Linux (Ubuntu 22.04) |

### Camera Calibration

```yaml
Camera.type: "PinHole"
Camera.fx: 1444.43
Camera.fy: 1444.34
Camera.cx: 1179.50
Camera.cy: 1044.90

Camera.k1: -0.0560
Camera.k2: 0.1180
Camera.p1: 0.00122
Camera.p2: 0.00064
Camera.k3: -0.0627

Camera.width: 2448
Camera.height: 2048
Camera.fps: 10.0
Camera.RGB: 0  # OpenCV images are typically BGR by default
```

**Note on ORB-SLAM3 settings format**:

- In ORB-SLAM3 `File.version: "1.0"` settings files, the intrinsics are typically stored as `Camera1.fx`, `Camera1.fy`, etc. (see `Examples/Monocular/HKisland_Mono.yaml` in the main repo).
- This demo includes `docs/camera_config.yaml` as a minimal, human-readable reference of the same calibration values.

### ORB Feature Extraction Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `nFeatures` | 3000 | Features per frame (tuned for completeness) |
| `scaleFactor` | 1.2 | Pyramid scale factor |
| `nLevels` | 8 | Pyramid levels |
| `iniThFAST` | 15 | Initial FAST threshold (lower for weak texture) |
| `minThFAST` | 5 | Minimum FAST threshold |

### Running ORB-SLAM3 (example)

This report assumes you have already generated a TUM-format trajectory file (e.g., `CameraTrajectory.txt` or `KeyFrameTrajectory.txt`) from ORB-SLAM3.

---

## 📈 Results and Analysis

### Evaluation Results

```
================================================================================
VISUAL ODOMETRY EVALUATION RESULTS
================================================================================

Ground Truth: RTK trajectory (1,955 poses)
Estimated:    ORB-SLAM3 camera trajectory (CameraTrajectory_sec.txt)
Matched Poses: 1,859 / 1,955 (95.09%)  ← Completeness

METRIC 1: ATE (Absolute Trajectory Error)
────────────────────────────────────────
RMSE:   37.14 m
Mean:   28.03 m
Std:    24.38 m

METRIC 2: RPE Translation Drift (distance-based, delta=10 m)
────────────────────────────────────────
Mean translational RPE over 10 m: 20.61 m
Translation drift rate:           2.06 m/m

METRIC 3: RPE Rotation Drift (distance-based, delta=10 m)
────────────────────────────────────────
Mean rotational RPE over 10 m: 17.16 deg
Rotation drift rate:        171.56 deg/100m

================================================================================
```

### Trajectory Alignment Statistics

| Parameter | Value |
|-----------|-------|
| **Sim(3) scale correction** | 0.61 |
| **Sim(3) translation (m)** | [1.1776×10⁷, 2.4722×10⁶, −35.50] (in ground-truth ENU frame) |
| **Association threshold** | \(t_{max\_diff}\) = 0.1 s |
| **Association rate (Completeness)** | 95.09% |
| **RPE distance delta** | 10 m |

### Performance Analysis

| Metric | Value | Grade | Interpretation |
|--------|-------|-------|----------------|
| **ATE RMSE** | 37.14 m | C | Moderate global error after Sim(3) alignment |
| **RPE Trans Drift** | 2.06 m/m | C | Moderate local drift per traveled distance |
| **RPE Rot Drift** | 171.56 deg/100m | D | Significant orientation drift |
| **Completeness** | 95.09% | A | High coverage; most poses evaluated |

---

## 📊 Visualizations

### Trajectory Comparison

![Trajectory Evaluation](figures/trajectory_evaluation.png)

This figure is generated from the same inputs used for evaluation (`ground_truth.txt` and `CameraTrajectory_sec.txt`) and includes:

1. **Top-Left**: 2D trajectory before alignment (matched poses only). This reveals scale/rotation mismatch typical for monocular VO.
2. **Top-Right**: 2D trajectory after Sim(3) alignment (scale corrected). Remaining discrepancy reflects drift and local tracking errors.
3. **Bottom-Left**: Distribution of ATE translation errors (meters) over all matched poses.
4. **Bottom-Right**: ATE translation error as a function of the matched pose index (highlights where drift accumulates).

**Reproducibility**: the figure is generated by the pipeline (step 8) using `scripts/generate_report_figures.py` with `output/ground_truth.txt`, `output/CameraTrajectory_sec.txt`, and `output/evo_results/ate.zip` (or computed Sim(3) and ATE if the zip is not used). It is saved as `output/trajectory.png` and copied to `figures/trajectory_evaluation.png`.

---

## 💭 Discussion

### Strengths

1. **High evaluation coverage**: 95% completeness indicates that the vast majority of ground-truth poses can be associated and evaluated.

2. **End-to-end pipeline**: The system produces a usable TUM trajectory (with EuRoC→TUM timestamp conversion) and can be evaluated reproducibly via `scripts/run_pipeline.sh` in Docker or with evo directly.

### Limitations

1. **Tracking Instability**: Frequent "Fail to track local map!" errors observed, leading to multiple map resets (2 maps created).

2. **Large drift**: Both translation and rotation drift rates are high, indicating unstable local tracking and/or poor geometric constraints.

3. **No loop closure**: Pure VO mode without loop closure or relocalization accumulates drift over long trajectories.

### Error Sources

1. **Fast UAV Motion**: Aggressive flight maneuvers cause motion blur and large inter-frame displacements.

2. **Feature Extraction**: ORB parameters were tuned (nFeatures=3000, lower FAST thresholds) to improve completeness; further tuning may reduce drift.

3. **Calibration Accuracy**: Camera intrinsics and distortion parameters affect pose estimation quality.

---

## 🎯 Conclusions

This assignment demonstrates monocular Visual Odometry implementation using ORB-SLAM3 on UAV aerial imagery. Key findings:

1. ✅ **System Operation**: ORB-SLAM3 successfully processes the image sequence over ~1.9 km trajectory
2. ✅ **Evaluation coverage**: 95.09% completeness shows that most poses can be evaluated against RTK ground truth
3. ⚠️ **Tracking stability**: Occasional tracking failures; ORB tuning (nFeatures, FAST thresholds) improved completeness
4. ⚠️ **Accuracy**: ATE RMSE 37.14 m and drift rates remain moderate to high; Sim(3) alignment corrects scale

### Recommendations for Improvement

Four optimization directions, ordered by priority (impact and feasibility):

| Prio | Action | Expected Improvement |
|:----:|--------|----------------------|
| **1** | Enable VIO (IMU fusion) | **50–70%** ATE RMSE reduction; observable scale; lower RPE drift. |
| **2** | Verify camera calibration (intrinsics + distortion) | **15–25%** accuracy gain; more stable tracking. |
| **3** | Tune ORB/FAST (e.g. `camera_config_mono_fallback.yaml`: nFeatures=3500, lower FAST) | **~5–15%** ATE reduction; +1–3% completeness; fewer map resets. |
| **4** | Enable loop closure / relocalization (full SLAM) | Bounded drift; **20–40%** ATE reduction on loop segments. |

---

## 📚 References

1. Campos, C., Elvira, R., Rodríguez, J. J. G., Montiel, J. M., & Tardós, J. D. (2021). **ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM**. *IEEE Transactions on Robotics*, 37(6), 1874-1890.

2. Sturm, J., Engelhard, N., Endres, F., Burgard, W., & Cremers, D. (2012). **A Benchmark for the Evaluation of RGB-D SLAM Systems**. *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*.

3. Geiger, A., Lenz, P., & Urtasun, R. (2012). **Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite**. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

4. MARS-LVIG Dataset: https://mars.hku.hk/dataset.html

5. ORB-SLAM3 GitHub: https://github.com/UZ-SLAMLab/ORB_SLAM3

---

## 📎 Appendix

### A. Repository Structure

```
AAE5303_assignment2_orbslam3_demo-/
├── README.md                    # This report
├── RUN.md                       # Docker pipeline instructions (HKisland_GNSS03)
├── requirements.txt             # Python dependencies
├── extract_images_final.py      # Extract images from ROS bag
├── extract_rtk_groundtruth.py   # Extract RTK ground truth (TUM, ~5 Hz)
├── figures/
│   └── trajectory_evaluation.png
├── output/
│   ├── evaluation_report.json
│   ├── report.md
│   └── trajectory.png           # Generated by pipeline; copied to figures/
├── scripts/
│   ├── evaluate_vo_accuracy.py
│   ├── generate_report_figures.py
│   ├── run_pipeline.sh          # Full pipeline (extract → ORB-SLAM3 → evo → plots)
│   ├── docker_run_hkisland.sh
│   ├── run_pipeline_fallback.sh
│   └── cleanup.sh
├── docs/
│   ├── camera_config.yaml       # Tuned ORB (nFeatures=3000, etc.)
│   └── camera_config_mono_fallback.yaml
└── leaderboard/
    ├── README.md
    ├── LEADERBOARD_SUBMISSION_GUIDE.md
    ├── submission_template.json
    └── submission_filled.json  # Auto-filled from evaluation_report.json
```

### B. Running Commands

**Recommended (Docker, one command):** See `RUN.md`. From repo root:

```bash
./scripts/docker_run_hkisland.sh
```

This runs image extraction, RTK ground truth extraction, ORB-SLAM3 (mono_tum), EuRoC→TUM timestamp conversion, evo evaluation, and figure generation. Outputs: `output/evaluation_report.json`, `output/trajectory.png`, `figures/trajectory_evaluation.png`, `leaderboard/submission_filled.json`.

**Manual steps (if not using Docker):**

```bash
# 1. Extract images from ROS bag
python3 extract_images_final.py HKisland_GNSS03.bag --output extracted_data

# 2. Run ORB-SLAM3 VO (inside ORB-SLAM3 build env)
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/DJI_Camera.yaml extracted_data

# 3. Extract RTK ground truth (downsampled to ~5 Hz)
python3 extract_rtk_groundtruth.py HKisland_GNSS03.bag --output ground_truth.txt

# 4. Convert trajectory timestamps to seconds if needed (EuRoC → TUM), then evaluate
python3 scripts/evaluate_vo_accuracy.py \
    --groundtruth ground_truth.txt \
    --estimated output/CameraTrajectory_sec.txt \
    --t-max-diff 0.1 \
    --delta-m 10 \
    --workdir output/evo_results \
    --json-out output/evaluation_report.json
```

### D. Native evo Commands

If you prefer to run evo directly (no custom scripts), use trajectories with **timestamps in seconds** (e.g. `CameraTrajectory_sec.txt` produced by the pipeline):

```bash
# ATE (Sim(3) alignment + scale correction)
evo_ape tum ground_truth.txt CameraTrajectory_sec.txt \
  --align --correct_scale \
  --t_max_diff 0.1 -va

# RPE translation (distance-based, delta = 10 m)
evo_rpe tum ground_truth.txt CameraTrajectory_sec.txt \
  --align --correct_scale \
  --t_max_diff 0.1 \
  --delta 10 --delta_unit m \
  --pose_relation trans_part -va

# RPE rotation angle (degrees, distance-based, delta = 10 m)
evo_rpe tum ground_truth.txt CameraTrajectory_sec.txt \
  --align --correct_scale \
  --t_max_diff 0.1 \
  --delta 10 --delta_unit m \
  --pose_relation angle_deg -va
```

### C. Output Trajectory Format (TUM)

Estimated trajectory from this project (`output/CameraTrajectory_sec.txt`, seconds; converted from ORB-SLAM3 EuRoC-ns output). Columns: `t tx ty tz qx qy qz qw`.

```
# t tx ty tz qx qy qz qw
1698132964.709855 -0.053938802 0.031431925 -0.179312542 -0.008169281 0.004799199 -0.001762640 0.999953568
1698132964.816207 -0.091809332 0.061578050 -0.369042665 -0.012694283 0.016093379 -0.000686561 0.999789715
1698132964.915064 -0.131898552 0.087799989 -0.570125997 -0.014429699 0.028575584 0.000898424 0.999487102
1698132965.017401 -0.173133433 0.111050524 -0.775480032 -0.016362375 0.033460870 0.002116043 0.999303877
...
```

---

<div align="center">

**AAE5303 - Robust Control Technology in Low-Altitude Aerial Vehicle**

*Department of Aeronautical and Aviation Engineering*

*The Hong Kong Polytechnic University*

Jan 2026

</div>

