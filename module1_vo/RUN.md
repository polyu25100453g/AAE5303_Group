# AAE5303 Assignment 2 — 运行说明（AMtown02）

数据集：`AMtown02.bag`  
路径（WSL）：`/mnt/d/AMtown02.bag`（Windows: `D:\AMtown02.bag`）  
Docker 镜像：`liangyu99/orbslam3_ros1:latest`（ROS1）

目标：**最大化 completeness**，减少 "Fail to track local map!" 与多地图重置。

---

## 1. 清理旧文件（防止空间爆炸和冲突）

在**宿主机**仓库根目录执行：

```bash
cd /home/njz/AAE5303_assignment2_orbslam3_demo-
chmod +x scripts/cleanup.sh
./scripts/cleanup.sh
```

或手动：

```bash
rm -rf output extracted_data
```

若在容器内曾把数据解压到 `/data`，进入容器后可选清理：

```bash
rm -rf /data/extracted_data /data/ground_truth.txt
```

---

## 2. 相机与 ORB 配置（已优化）

- 主配置：`docs/camera_config.yaml`  
  - **ORB**: `nFeatures: 3000`, `iniThFAST: 15`, `minThFAST: 5`, `scaleFactor: 1.2`, `nLevels: 8`  
  - 针对弱纹理、天空占比大、快速运动，提高特征数与降低 FAST 阈值以提升跟踪完整率。
- 若 completeness 仍低，可改用更激进配置：  
  `docs/camera_config_mono_fallback.yaml`（3500 特征，更低 FAST），见下文「备选方案」。

`run_pipeline.sh` 会把 `docs/camera_config.yaml` 拷贝到容器内 `ORB_SLAM3/Examples/Monocular/DJI_Camera.yaml` 使用。

---

## 3. WSL/Windows 一键 Docker 运行

在**仓库根目录**执行（确保 bag 在 `/mnt/d/AMtown02.bag`）：

```bash
chmod +x scripts/docker_run_amtown02.sh
./scripts/docker_run_amtown02.sh
```

或**手动一条命令**（在 WSL 下，仓库根目录 `$(pwd)` 即 repo 路径）：

```bash
docker run -it --rm \
  --name orbslam3_amtown02 \
  --gpus all \
  --privileged \
  --network=host \
  -e DISPLAY=host.docker.internal:0.0 \
  -e QT_X11_NO_MITSHM=1 \
  -v /mnt/d/AMtown02.bag:/data/AMtown02.bag:ro \
  -v "$(pwd)":/workspace:rw \
  liangyu99/orbslam3_ros1:latest \
  bash -c "cd /workspace && chmod +x scripts/run_pipeline.sh && ./scripts/run_pipeline.sh /data/AMtown02.bag /data/extracted_data /data/ground_truth.txt /workspace/output"
```

若 bag 不在 `/mnt/d/`，可先设置变量再运行脚本：

```bash
export BAG_PATH=/你的/bag/路径/AMtown02.bag
./scripts/docker_run_amtown02.sh
```

---

## 4. 容器内分步执行（可选）

若希望进入容器后自己执行每一步：

```bash
docker run -it --rm \
  --name orbslam3_amtown02 \
  --gpus all --privileged --network=host \
  -e DISPLAY=host.docker.internal:0.0 \
  -v /mnt/d/AMtown02.bag:/data/AMtown02.bag:ro \
  -v "$(pwd)":/workspace:rw \
  liangyu99/orbslam3_ros1:latest \
  bash
```

进入容器后：

```bash
cd /workspace
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh /data/AMtown02.bag /data/extracted_data /data/ground_truth.txt /workspace/output
```

ORB-SLAM3 的 `mono_tum` **不支持**命令行覆盖 `nFeatures` 等参数，所有 ORB 参数均来自 YAML。因此调整 `docs/camera_config.yaml` 后直接重跑上述 pipeline 即可。

---

## 5. 备选方案（completeness 仍低时）

- **纯 Monocular**：当前 pipeline 已是纯单目（`mono_tum`），无 IMU。无需改模式。
- **更激进 ORB**：  
  - 将 `docs/camera_config_mono_fallback.yaml` 覆盖到 `docs/camera_config.yaml`，再执行一次 `run_pipeline.sh`；  
  - 或在**容器内**直接执行：`./scripts/run_pipeline_fallback.sh /data/AMtown02.bag /data/extracted_data /data/ground_truth.txt /workspace/output`（脚本会临时改用 fallback 配置并调用原 pipeline）。

---

## 6. 跑完后的检查与提交

1. **查看输出**  
   - `output/CameraTrajectory.txt`  
   - `output/evaluation_report.json`  
   - `output/trajectory.png`  
   - `output/report.md`

2. **提交用 JSON**  
   - 脚本会自动生成 `leaderboard/submission_filled.json`（从 `evaluation_report.json` 填充）。  
   - 如需手动，可参考 `leaderboard/submission_template.json`，将 `ate_rmse_m`, `rpe_trans_drift_m_per_m`, `rpe_rot_drift_deg_per_100m`, `completeness_pct` 从 `output/evaluation_report.json` 填入。

3. **简要报告**  
   - 可使用 `output/REPORT_TEMPLATE.md` 作为模板，说明为提升 robustness 所做的调参（见下一节）。

---

## 7. 调参说明模板（Robustness / 完整性）

在报告或 `output/REPORT_TEMPLATE.md` 中可写：

- **ORB nFeatures 提高到 2500–3000（如 3000）**：弱纹理与天空区域可用特征少，增加每帧特征数有助于维持跟踪，减少 "Fail to track local map!"。
- **iniThFAST / minThFAST 降低（如 15 / 5）**：低对比度区域（天空、水面、远山）更容易提取到 FAST 角点，提高跟踪鲁棒性。
- **保持 scaleFactor=1.2, nLevels=8**：与官方/参考配置一致，保证尺度金字塔稳定。
- **优先使用纯 Monocular**：本 pipeline 已为单目；若曾用 inertial 易丢跟踪，可确认未启用 IMU（当前配置已为单目）。

按上述步骤可复现、合规，且优先完整性（completeness）。
