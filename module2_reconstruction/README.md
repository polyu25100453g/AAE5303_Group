# AAE5303 Group Project - Module 2 (3D Scene Reconstruction)

本模块用于承接 `module1_vo` 的输出，完成 3D scene reconstruction（OpenSplat）。

## 目标

- 复用 `module1_vo` 的图像与位姿结果
- 通过 OpenSplat 进行 3D Gaussian Splatting 重建
- 产出可展示结果（模型文件、渲染图/视频、指标截图）

## 与 module1_vo 的衔接

默认输入来源：

- 图像：`../module1_vo/extracted_data`
- 位姿（可选）：`../module1_vo/output/CameraTrajectory_sec.txt`
- 相机配置（可选）：`../module1_vo/docs/camera_config.yaml`

若你的组员在其他目录产出了图像/轨迹，可通过脚本参数覆盖。

## 目录结构

```text
module2_reconstruction/
├── README.md
├── RUN.md
├── input/
│   ├── images
│   ├── trajectories
│   └── camera
├── output/
├── logs/
└── scripts/
    ├── prepare_inputs.py
    └── docker_run_opensplat.sh
```

## 快速开始

1) 准备输入（从 module1_vo 拷贝/链接）：

```bash
cd /home/user/slam_ws/AAE5303_Group/module2_reconstruction
python3 scripts/prepare_inputs.py
```

2) 启动 OpenSplat 自动训练（课程镜像）：

```bash
bash scripts/docker_run_opensplat.sh --auto-train
```

3) 结果会写到：

- `/workspace/module2_reconstruction/output`

## 无 Docker 本地方案（我已帮你配好）

当 Docker daemon 不可用时，直接本地跑：

```bash
cd /home/user/slam_ws/AAE5303_Group/module2_reconstruction
source .venv/bin/activate
bash scripts/run_local_reconstruction.sh
```

说明：

- 该脚本会先用 `pycolmap` 从 `input/images` 生成 COLMAP 稀疏模型
- 如果检测到 `../third_party/OpenSplat/build/opensplat`，会继续训练并输出 `.splat`
- 如果未检测到 opensplat 二进制，也会保留可用的 COLMAP 工程用于后续训练

## 可选：手动指定训练命令

如果容器里的默认入口不匹配，可以显式传命令（支持 `{input}` 和 `{output}` 占位符）：

```bash
OPENSPLAT_CMD='opensplat train --data {input} --output {output}' \
bash scripts/docker_run_opensplat.sh --auto-train
```

## 提示

- 你的角色是 3D 重建，这个模块不重新跑 SLAM，只消费 SLAM 结果。
- `input/manifest.json` 会记录当前使用的是哪些输入，方便答辩时复现。
