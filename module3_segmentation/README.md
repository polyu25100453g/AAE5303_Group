# Module 3 - Semantic Segmentation

This module performs semantic segmentation on images from `module1_vo` and
exports:

- label masks (class id per pixel)
- colorized masks
- overlay visualizations (input + segmentation)
- a simple run summary JSON

## Goal

- Reuse outputs from previous modules (especially images from `module1_vo`)
- Produce reproducible semantic segmentation outputs
- Prepare data for optional fusion with Module 2/3D reconstruction

## Inputs

Default input image directory:

- `../module1_vo/extracted_data`

Optional overrides are supported by command line arguments.

## Outputs

Default output root:

- `module3_segmentation/output`

Subfolders:

- `output/masks` (`*.png`, class id map)
- `output/color_masks` (`*.png`, palette visualization)
- `output/overlays` (`*.png`, blended with input image)
- `output/summary.json`

## Quick Start

From repository root:

```bash
cd module3_segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_module3.sh
```

## Notes

- **COCO 预训练仅作演示**：与 UAVScenes **26 类** GT 直接算 mIoU 没有意义，见 `RUN.md` 第 6 节：先 `convert_rgb_gt_to_id.py`，再 `train_deeplab_uavscenes.py` 微调，推理时加 `--checkpoint`。
- 课程推荐的 [PyTorch-UNet](https://github.com/milesial/Pytorch-UNet) 可在同一套 **id 标注** 上替换训练部分；评估仍用 `evaluate_segmentation.py --num-classes 26`。
