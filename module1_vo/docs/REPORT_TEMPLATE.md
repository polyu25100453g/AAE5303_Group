# AAE5303 Assignment 2 — 简要报告（调参与 Robustness）

## 配置概览

- **数据集**: HKisland_GNSS03.bag（Island 场景：弱纹理、天空占比大、高空快速运动）
- **模式**: 纯 Monocular（mono_tum），无 IMU
- **相机配置**: `docs/camera_config.yaml` → `Examples/Monocular/DJI_Camera.yaml`

## 为提升 Completeness 所做的调参

| 参数 | 原/默认 | 调整后 | 原因 |
|------|--------|--------|------|
| **ORBextractor.nFeatures** | 1500 | **3000** | 弱纹理与天空区域可用特征少，增加每帧特征数有助于维持跟踪，减少 "Fail to track local map!" 与多地图重置。 |
| **ORBextractor.iniThFAST** | 20 | **15** | 降低初始 FAST 阈值，在低对比度区域（天空、水面）能提取更多角点，提高鲁棒性。 |
| **ORBextractor.minThFAST** | 7 | **5** | 当初始阈值未检测到足够角点时，使用更低的备用阈值，减少丢帧。 |
| **ORBextractor.scaleFactor** | 1.2 | 1.2 | 保持与参考一致。 |
| **ORBextractor.nLevels** | 8 | 8 | 保持与参考一致。 |

## 结果摘要（跑完后填写）

- **ATE RMSE (m)**: （见 `output/evaluation_report.json` → `ate_rmse_m`）
- **RPE trans drift (m/m)**: （见 `rpe_trans_drift_m_per_m`）
- **RPE rot drift (deg/100m)**: （见 `rpe_rot_drift_deg_per_100m`）
- **Completeness (%)**: （见 `completeness_pct`，目标：尽量提高）

## 复现

1. 清理：`./scripts/cleanup.sh`
2. 运行：`./scripts/docker_run_hkisland.sh`（或按 `RUN.md` 中 Docker 命令执行）
3. 提交：使用 `leaderboard/submission_filled.json`（由 pipeline 自动填充）

## 备选

若 completeness 仍不理想，可改用 `docs/camera_config_mono_fallback.yaml`（nFeatures=3500，更低 FAST），再重跑 pipeline；或执行 `./scripts/run_pipeline_fallback.sh`（需在容器内、同样参数）。
