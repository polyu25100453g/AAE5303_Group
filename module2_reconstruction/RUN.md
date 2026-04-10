# Module 2 Run Guide (OpenSplat)

## 环境说明

- 推荐镜像：`liangyu99/opensplat-cpu:latest`
- 宿主目录：`/home/user/slam_ws/AAE5303_Group/module2_reconstruction`

## 一键流程

```bash
cd /home/user/slam_ws/AAE5303_Group/module2_reconstruction
python3 scripts/prepare_inputs.py
bash scripts/docker_run_opensplat.sh --auto-train
```

默认会直接执行 `scripts/run_opensplat_train.sh`（自动探测 OpenSplat 入口）。

## 容器内建议

进入容器后先检查输入：

```bash
ls /workspace/module2_reconstruction/input
ls /workspace/module2_reconstruction/input/images | wc -l
```

然后执行你们组使用的 OpenSplat 训练/渲染命令，输出到：

```text
/workspace/module2_reconstruction/output
```

## 非交互模式

如果你已经有固定命令，可以直接传入：

```bash
bash scripts/docker_run_opensplat.sh --cmd "echo your_opensplat_command_here"
```

或者用自动训练入口并覆盖命令模板：

```bash
OPENSPLAT_CMD='opensplat train --data {input} --output {output}' \
bash scripts/docker_run_opensplat.sh --auto-train
```

## 常见问题

- 如果 `input/images` 为空，说明 `module1_vo` 的图像路径不一致；重新运行：
  - `python3 scripts/prepare_inputs.py --images /your/real/images/path`
- 如果找不到轨迹文件，不影响启动（轨迹是可选输入），但建议让组员提供 `CameraTrajectory_sec.txt` 便于后续对齐与展示。
