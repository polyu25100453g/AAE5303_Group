#!/usr/bin/env python3
"""
Fine-tune DeepLabV3-ResNet50 for UAVScenes (26 classes, ignore 255).

Prepare:
  --images-dir   RGB frames (same basenames as masks)
  --masks-id-dir PNG masks, pixel = class id 0..25 (use convert_rgb_gt_to_id.py on RGB GT)

Example:
  python3 scripts/train_deeplab_uavscenes.py \\
    --images-dir /path/to/cam \\
    --masks-id-dir /path/to/gt_id \\
    --out checkpoints/best.pt \\
    --epochs 30
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from torchvision.transforms import functional as TF

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from uavscenes_cmap import IGNORE_LABEL, NUM_UAVSCENES_CLASSES


def replace_deeplab_head(model: nn.Module, num_classes: int) -> None:
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    in_aux = model.aux_classifier[-1].in_channels
    model.aux_classifier[-1] = nn.Conv2d(in_aux, num_classes, kernel_size=1)


def collect_stems(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    img_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    pairs: list[tuple[Path, Path]] = []
    for p in sorted(images_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in img_ext:
            continue
        rel = p.relative_to(images_dir)
        m = masks_dir / rel.with_suffix(".png")
        if not m.is_file():
            m = masks_dir / f"{p.stem}.png"
        if m.is_file():
            pairs.append((p, m))
    return pairs


class SegPairDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        size: tuple[int, int],
        weights: DeepLabV3_ResNet50_Weights,
        augment_hflip: bool,
    ) -> None:
        self.pairs = pairs
        self.h, self.w = size
        self.mean = torch.tensor(weights.meta["mean"]).view(3, 1, 1)
        self.std = torch.tensor(weights.meta["std"]).view(3, 1, 1)
        self.augment_hflip = augment_hflip

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ip, mp = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")
        img = TF.resize(img, (self.h, self.w), antialias=True)
        mask = TF.resize(mask, (self.h, self.w), interpolation=Image.NEAREST)
        if self.augment_hflip and random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        t = TF.to_tensor(img)
        t = (t - self.mean) / self.std
        m = torch.from_numpy(np.array(mask, dtype=np.int64))
        return t, m


def confusion_accum(pred: torch.Tensor, gt: torch.Tensor, num_classes: int, ignore: int) -> np.ndarray:
    pred = pred.detach().cpu().numpy().reshape(-1)
    gt = gt.detach().cpu().numpy().reshape(-1)
    keep = (gt != ignore) & (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[keep].astype(np.int64)
    gt = gt[keep].astype(np.int64)
    if pred.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    k = gt * num_classes + pred
    bc = np.bincount(k, minlength=num_classes * num_classes)
    return bc.reshape(num_classes, num_classes).astype(np.int64)


def miou_from_cm(cm: np.ndarray) -> float:
    ious = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        u = tp + fp + fn
        if u > 0:
            ious.append(tp / u)
    return float(np.mean(ious)) if ious else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--masks-id-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("checkpoints/uavscenes_deeplab.pt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=520, help="Train resolution (square).")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    wts = DeepLabV3_ResNet50_Weights.DEFAULT
    pairs = collect_stems(args.images_dir, args.masks_id_dir)
    if len(pairs) < 4:
        raise RuntimeError(
            f"Need at least 4 image/mask pairs; found {len(pairs)}. "
            "Check --images-dir / --masks-id-dir and matching filenames."
        )
    random.seed(42)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"train={len(train_pairs)} val={len(val_pairs)}")

    train_ds = SegPairDataset(train_pairs, (args.size, args.size), wts, augment_hflip=True)
    val_ds = SegPairDataset(val_pairs, (args.size, args.size), wts, augment_hflip=False)
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=min(4, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = deeplabv3_resnet50(weights=wts)
    replace_deeplab_head(model, NUM_UAVSCENES_CLASSES)
    model = model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_tr = 0.0
        n = 0
        for x, y in train_ld:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = ce(out["out"], y) + 0.4 * ce(out["aux"], y)
            loss.backward()
            opt.step()
            loss_tr += float(loss.item())
            n += 1
        loss_tr /= max(n, 1)

        model.eval()
        cm = np.zeros((NUM_UAVSCENES_CLASSES, NUM_UAVSCENES_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x)["out"].argmax(dim=1)
                for b in range(pred.size(0)):
                    cm += confusion_accum(pred[b], y[b], NUM_UAVSCENES_CLASSES, IGNORE_LABEL)
        v_miou = miou_from_cm(cm)
        print(f"epoch {epoch:03d}  train_loss={loss_tr:.4f}  val_mIoU={v_miou:.4f}")

        if v_miou > best_miou:
            best_miou = v_miou
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_classes": NUM_UAVSCENES_CLASSES,
                    "ignore_label": IGNORE_LABEL,
                    "epoch": epoch,
                    "val_miou": v_miou,
                },
                args.out,
            )
            print(f"  saved {args.out} (best val_mIoU={best_miou:.4f})")

    print(f"Done. Best val mIoU={best_miou:.4f} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
