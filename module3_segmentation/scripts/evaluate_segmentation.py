#!/usr/bin/env python3
"""
Compute segmentation metrics vs ground-truth label masks (PNG, class id per pixel).

Outputs mIoU, dice_score (macro mean Dice), fwIoU — values can be scaled to 0–100
to match common leaderboard JSON (see --as-percent).

Requires: prediction masks (e.g. output/masks/*.png) and a parallel GT directory
with the same filenames; pixel values 0 .. num_classes-1 (same encoding as preds).
Use --ignore-label (e.g. 255) for void/ignore in GT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_label_array(path: Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im.convert("L"), dtype=np.uint8)
    return arr


def collect_mask_paths(pred_dir: Path) -> list[Path]:
    return sorted(p for p in pred_dir.glob("*.png") if p.is_file())


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_label: int) -> np.ndarray:
    pred = pred.reshape(-1).astype(np.int64)
    gt = gt.reshape(-1).astype(np.int64)
    mask = gt != ignore_label
    pred = pred[mask]
    gt = gt[mask]
    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[valid]
    gt = gt[valid]
    if pred.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    k = gt * num_classes + pred
    bc = np.bincount(k, minlength=num_classes * num_classes)
    return bc.reshape(num_classes, num_classes).astype(np.int64)


def metrics_from_cm(cm: np.ndarray) -> tuple[float, float, float]:
    """Return (miou, macro_dice, fwiou) in 0..1 range."""
    num_classes = cm.shape[0]
    ious: list[float] = []
    dices: list[float] = []
    gt_counts = cm.sum(axis=0).astype(np.float64)
    total_gt = gt_counts.sum()
    fwiou_num = 0.0
    fwiou_den = 0.0

    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        union = tp + fp + fn
        if union <= 0:
            ious.append(float("nan"))
            dices.append(float("nan"))
            continue
        iou = tp / union
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-12)
        ious.append(iou)
        dices.append(dice)
        if total_gt > 0:
            w = gt_counts[c] / total_gt
            if w > 0:
                fwiou_num += w * iou
                fwiou_den += w

    miou = float(np.nanmean(np.array(ious)))
    dice_score = float(np.nanmean(np.array(dices)))
    # Frequency-weighted IoU: weight each class IoU by its GT pixel frequency.
    fwiou = float(fwiou_num / (fwiou_den + 1e-12)) if fwiou_den > 0 else float("nan")
    return miou, dice_score, fwiou


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate segmentation masks vs GT.")
    p.add_argument("--pred-dir", type=Path, required=True, help="Predicted label PNGs (e.g. output/masks).")
    p.add_argument("--gt-dir", type=Path, required=True, help="Ground-truth label PNGs, same filenames as pred.")
    p.add_argument(
        "--num-classes",
        type=int,
        default=21,
        help="Class count: 21 for COCO DeepLab; 26 for UAVScenes (after GT id conversion).",
    )
    p.add_argument("--ignore-label", type=int, default=255, help="Ignore this label in GT (void).")
    p.add_argument(
        "--as-percent",
        action="store_true",
        help="Scale metrics to 0–100 (many leaderboards use 72.73 instead of 0.7273).",
    )
    p.add_argument("--json-out", type=Path, default=None, help="Write metrics JSON to this path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.pred_dir.is_dir():
        raise FileNotFoundError(f"pred-dir not found: {args.pred_dir}")
    if not args.gt_dir.is_dir():
        raise FileNotFoundError(f"gt-dir not found: {args.gt_dir}")

    nearest = getattr(Image, "Resampling", Image).NEAREST
    cm = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    used = 0
    for pred_path in collect_mask_paths(args.pred_dir):
        gt_path = args.gt_dir / pred_path.name
        if not gt_path.is_file():
            continue
        pred = load_label_array(pred_path)
        gt = load_label_array(gt_path)
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred, mode="L").resize((gt.shape[1], gt.shape[0]), nearest),
                dtype=np.uint8,
            )
        cm += confusion_matrix(pred, gt, args.num_classes, args.ignore_label)
        used += 1

    if used == 0:
        raise RuntimeError(
            "No matching pred/GT pairs found (same *.png names in both dirs). "
            "Check --gt-dir and filenames."
        )

    miou, dice, fwiou = metrics_from_cm(cm)
    scale = 100.0 if args.as_percent else 1.0
    out = {
        "miou": round(miou * scale, 4),
        "dice_score": round(dice * scale, 4),
        "fwiou": round(fwiou * scale, 4),
        "num_eval_images": used,
        "num_classes": args.num_classes,
        "ignore_label": args.ignore_label,
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
