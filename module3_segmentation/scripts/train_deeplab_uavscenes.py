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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from torchvision.transforms import ColorJitter
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


def histogram_class_pixels(
    pairs: list[tuple[Path, Path]],
    num_classes: int,
    ignore: int,
    sample_masks: int,
) -> np.ndarray:
    rng = random.Random(42)
    use = pairs if len(pairs) <= sample_masks else rng.sample(pairs, sample_masks)
    h = np.zeros(num_classes, dtype=np.float64)
    for _, mp in use:
        m = np.asarray(Image.open(mp).convert("L"), dtype=np.int64).ravel()
        m = m[(m != ignore) & (m >= 0) & (m < num_classes)]
        if m.size:
            h += np.bincount(m, minlength=num_classes)
    return h


def class_balanced_weights(counts: np.ndarray, beta: float = 0.99) -> torch.Tensor:
    """Effective number of samples weighting (Class-Balanced Loss style)."""
    counts = np.maximum(counts, 1.0)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / (eff + 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


class SegPairDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        size: tuple[int, int],
        weights: DeepLabV3_ResNet50_Weights,
        train: bool,
    ) -> None:
        self.pairs = pairs
        self.h, self.w = size
        self.mean = torch.tensor(weights.meta["mean"]).view(3, 1, 1)
        self.std = torch.tensor(weights.meta["std"]).view(3, 1, 1)
        self.train = train
        self.color_jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ip, mp = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")
        if self.train:
            img = self.color_jitter(img)
        img = TF.resize(img, (self.h, self.w), antialias=True)
        mask = TF.resize(mask, (self.h, self.w), interpolation=Image.NEAREST)
        if self.train and random.random() < 0.5:
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
    p.add_argument("--aux-weight", type=float, default=0.4)
    p.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-balanced CE (default: on, helps rare UAVScenes classes).",
    )
    p.add_argument(
        "--weight-sample-masks",
        type=int,
        default=384,
        help="How many train masks to scan for class frequency (speed vs accuracy).",
    )
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
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

    train_ds = SegPairDataset(train_pairs, (args.size, args.size), wts, train=True)
    val_ds = SegPairDataset(val_pairs, (args.size, args.size), wts, train=False)
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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    if args.no_class_weights:
        ce_w: torch.Tensor | None = None
    else:
        hist = histogram_class_pixels(
            train_pairs,
            NUM_UAVSCENES_CLASSES,
            IGNORE_LABEL,
            min(args.weight_sample_masks, len(train_pairs)),
        )
        ce_w = class_balanced_weights(hist).to(args.device)
        print("class pixel histogram (sampled):", hist.astype(int).tolist())

    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, weight=ce_w)
    use_amp = args.device == "cuda" and not args.no_amp
    scaler = GradScaler(enabled=use_amp)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_tr = 0.0
        n = 0
        for x, y in train_ld:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                out = model(x)
                loss = ce(out["out"], y) + args.aux_weight * ce(out["aux"], y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_tr += float(loss.item())
            n += 1
        loss_tr /= max(n, 1)
        sched.step()

        model.eval()
        cm = np.zeros((NUM_UAVSCENES_CLASSES, NUM_UAVSCENES_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x)["out"].argmax(dim=1)
                for b in range(pred.size(0)):
                    cm += confusion_accum(pred[b], y[b], NUM_UAVSCENES_CLASSES, IGNORE_LABEL)
        v_miou = miou_from_cm(cm)
        lr_now = float(opt.param_groups[0]["lr"])
        print(f"epoch {epoch:03d}  lr={lr_now:.2e}  train_loss={loss_tr:.4f}  val_mIoU={v_miou:.4f}")

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
