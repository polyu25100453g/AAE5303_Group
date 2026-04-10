# Module 3 Run Guide

## 1) Setup

**一键创建环境**（克隆仓库后 `.venv` 不在 git 里，必须先做）：

```bash
cd ~/AAE5303_Group/module3_segmentation
bash scripts/bootstrap_venv.sh
```

或手动：

```bash
cd ~/AAE5303_Group/module3_segmentation
python3 -m venv .venv
source .venv/bin/activate   # 每次新开终端都要执行，否则会 ModuleNotFoundError: numpy
pip install -r requirements.txt
```

之后所有 `python3 scripts/...` 都在**已 activate 的同一终端**里运行。

**常见错误**：不要把参数粘在一起，例如错误写法 `--tta-ms-checkpoint`；应写成 **`--tta-ms`** 和 **`--checkpoint checkpoints/xxx.pt`** 两个参数。

## 2) Default run (use module1_vo images)

自动优先使用 `../module1_vo/extracted_data/rgb`（抽帧后的相机图）。可选环境变量：

- `CHECKPOINT=/path/to/uavscenes_deeplab.pt` — 使用微调权重（26 类）
- `TTA=1` — 水平翻转 TTA
- `TTA_MS=1` — 多尺度 TTA（更慢，略提分）
- `MAX_IMAGES=500` — 限制张数

```bash
bash scripts/run_module3.sh
```

## 3) Custom input/output

```bash
python3 scripts/infer_segmentation.py \
  --input-dir ../module1_vo/extracted_data \
  --output-dir ./output \
  --max-images 200
```

## 4) Check results

```bash
ls output/masks | head
ls output/overlays | head
cat output/summary.json
```

## 5) Leaderboard metrics (mIoU / Dice / fwIoU)

These numbers **require ground-truth label masks** (PNG, one byte per pixel = class id, same naming as `output/masks/*.png`). Put them in e.g. `gt_masks/` then:

```bash
python3 scripts/evaluate_segmentation.py \
  --pred-dir ./output/masks \
  --gt-dir /path/to/gt_masks \
  --num-classes 21 \
  --as-percent \
  --json-out ./output/segmentation_metrics.json
```

Copy the three values into `leaderboard/submission_segmentation_template.json` under `metrics`, set `group_name` and `project_private_repo_url`, then paste into the site’s **unet** JSON field.

---

## 6) UAVScenes：为什么 COCO 预训练直接算 mIoU 会错？怎么做才对？

**原因**：`deeplabv3_resnet50` 默认是 **21 类 COCO**；UAVScenes 语义标签是 **26 类（id 0–25）**，类别语义与编号都不一致，不能直接拿 GT 和预测算 IoU。

**推荐流程**：

1. **把官方 RGB 彩色标注转为 id 图**（与相机图同名、单通道 0–25，未知像素为 255）：

```bash
python3 scripts/convert_rgb_gt_to_id.py \
  --input-dir /path/to/UAVScenes/.../semantic_rgb \
  --output-dir ./gt_masks_id
```

2. **（建议）先检查图像与 mask 是否对齐**：

```bash
python3 scripts/check_uavscenes_pairs.py \
  --images-dir /path/to/camera_images \
  --masks-id-dir ./gt_masks_id \
  --list-missing 10
```

3. **微调 DeepLab 头为 26 类**（需成对数据：RGB 图目录 + 上一步的 id mask，文件名对齐）：

```bash
python3 scripts/train_deeplab_uavscenes.py \
  --images-dir /path/to/camera_images \
  --masks-id-dir ./gt_masks_id \
  --out checkpoints/uavscenes_deeplab.pt \
  --epochs 60 \
  --batch-size 4 \
  --lr 1e-4
```

训练侧默认：**类别均衡 CE**、**轻量 Focal**、**骨干/头部分组 LR**（`--backbone-lr-ratio`，默认 0.1）、**随机缩放+裁剪**、**ColorJitter**、**翻转**、**warmup + cosine**、**梯度裁剪**、**AMP**、**早停**。可选 **`--dice-weight 0.2~0.4`**（Dice 辅助损失）、**`--resume checkpoints/last.pt`** 断点续训；每个 epoch 会写 **`checkpoints/last.pt`**。评估可加 **`--per-class`** 导出每类 IoU；推理可加 **`--fp16`**（CUDA）。

4. **用微调权重推理**（`--tta` / `--tta-ms` 可叠加）：

```bash
python3 scripts/infer_segmentation.py \
  --input-dir /path/to/camera_images \
  --output-dir ./output_uavscenes_finetuned \
  --checkpoint checkpoints/uavscenes_deeplab.pt \
  --max-images 0 \
  --tta --tta-ms
```

（`--max-images 0`：处理目录内全部图像。）

5. **评估时类别数用 26**：

```bash
python3 scripts/evaluate_segmentation.py \
  --pred-dir ./output_uavscenes_finetuned/masks \
  --gt-dir ./gt_masks_id \
  --num-classes 26 \
  --ignore-label 255 \
  --as-percent \
  --json-out ./output/segmentation_metrics.json
```

课程若要求 **PyTorch-UNet**，可在同一套 **id 标注** 上换用 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) 训练，`--num-classes` 仍设为 **26**；评估脚本不变。
