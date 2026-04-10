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
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# torchvision 版本差异：部分环境下 Weights.meta 无 mean/std，与 ImageNet 预训练一致即可。
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def mean_std_tensors(wts: DeepLabV3_ResNet50_Weights) -> tuple[torch.Tensor, torch.Tensor]:
    meta = getattr(wts, "meta", None) or {}
    mean = meta.get("mean", _IMAGENET_MEAN)
    std = meta.get("std", _IMAGENET_STD)
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return m, s


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


def soft_dice_loss_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore: int) -> torch.Tensor:
    """1 - mean class Dice on valid pixels (helps boundaries / small objects)."""
    valid = target != ignore
    if int(valid.sum().item()) == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    prob = F.softmax(logits, dim=1)
    t = target.clone()
    t[~valid] = 0
    oh = F.one_hot(t.clamp(min=0, max=num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
    vf = valid.unsqueeze(1).float()
    oh = oh * vf
    pr = prob * vf
    inter = (pr * oh).sum(dim=(0, 2, 3))
    union = pr.pow(2).sum(dim=(0, 2, 3)) + oh.pow(2).sum(dim=(0, 2, 3))
    dice = (2.0 * inter + 1e-5) / (union + 1e-5)
    return 1.0 - dice.mean()


def ce_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    ignore: int,
    focal_gamma: float,
) -> torch.Tensor:
    ce_n = F.cross_entropy(logits, target, weight=weight, ignore_index=ignore, reduction="none")
    if focal_gamma <= 0:
        mask = target != ignore
        return ce_n[mask].mean()
    mask = target != ignore
    pt = torch.exp(-torch.clamp(ce_n, max=50.0))
    focal = (1.0 - pt).pow(2.0) * ce_n
    return ce_n[mask].mean() + focal_gamma * focal[mask].mean()


def lr_at_epoch(epoch: int, base_lr: float, warmup: int, total_epochs: int, eta_min_ratio: float) -> float:
    eta_min = base_lr * eta_min_ratio
    if epoch <= warmup:
        return base_lr * float(epoch) / float(max(warmup, 1))
    t = epoch - warmup - 1
    T = max(total_epochs - warmup, 1)
    return eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * float(t) / float(max(T - 1, 1))))


class SegPairDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        size: tuple[int, int],
        weights: DeepLabV3_ResNet50_Weights,
        train: bool,
        scale_aug: bool,
        scale_min: float,
        scale_max: float,
    ) -> None:
        self.pairs = pairs
        self.h, self.w = size
        self.mean, self.std = mean_std_tensors(weights)
        self.train = train
        self.scale_aug = scale_aug and train
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.color_jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04)

    def __len__(self) -> int:
        return len(self.pairs)

    def _random_scale_crop(self, img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        w0, h0 = img.size
        ch, cw = self.h, self.w
        base = max(ch, cw)
        lo = max(int(base * self.scale_min), base)
        hi = max(int(base * self.scale_max), lo + 8)
        target_min = random.randint(lo, hi)
        scale = target_min / float(min(h0, w0))
        nw = max(int(round(w0 * scale)), cw)
        nh = max(int(round(h0 * scale)), ch)
        img = img.resize((nw, nh), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
        mask = mask.resize((nw, nh), Image.NEAREST)
        top = random.randint(0, max(0, nh - ch))
        left = random.randint(0, max(0, nw - cw))
        img = img.crop((left, top, left + cw, top + ch))
        mask = mask.crop((left, top, left + cw, top + ch))
        return img, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ip, mp = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")
        if self.train:
            img = self.color_jitter(img)
        if self.scale_aug:
            img, mask = self._random_scale_crop(img, mask)
        else:
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
    p.add_argument(
        "--no-scale-aug",
        action="store_true",
        help="Disable random scale+crop (only fixed resize like val).",
    )
    p.add_argument("--scale-min", type=float, default=0.65, help="Min scale vs crop size for train aug.")
    p.add_argument("--scale-max", type=float, default=1.35, help="Max scale vs crop size for train aug.")
    p.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup before cosine decay.")
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 = disable grad norm clip.")
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=0.25,
        help="Extra focal term weight (0 = CE only). Helps hard pixels.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=18,
        help="Early stop if val mIoU does not improve for this many epochs (0 = off).",
    )
    p.add_argument(
        "--backbone-lr-ratio",
        type=float,
        default=0.1,
        help="LR multiplier for ResNet backbone vs segmentation head (fine-tune stability).",
    )
    p.add_argument(
        "--dice-weight",
        type=float,
        default=0.0,
        help="If >0, add soft Dice loss on main branch (e.g. 0.2–0.5).",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint (state_dict); continues until --epochs total.",
    )
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

    use_scale = not args.no_scale_aug
    train_ds = SegPairDataset(
        train_pairs,
        (args.size, args.size),
        wts,
        train=True,
        scale_aug=use_scale,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
    )
    val_ds = SegPairDataset(
        val_pairs,
        (args.size, args.size),
        wts,
        train=False,
        scale_aug=False,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
    )
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

    start_epoch = 0
    resume_best = -1.0
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"--resume not found: {args.resume}")
        try:
            rck = torch.load(args.resume, map_location=args.device, weights_only=False)
        except TypeError:
            rck = torch.load(args.resume, map_location=args.device)
        model = deeplabv3_resnet50(weights=None)
        replace_deeplab_head(model, NUM_UAVSCENES_CLASSES)
        model.load_state_dict(rck["state_dict"], strict=True)
        start_epoch = int(rck.get("epoch", 0))
        resume_best = float(rck.get("val_miou", -1.0))
        print(f"resume: epoch={start_epoch} val_mIoU={resume_best:.4f} from {args.resume}")
    else:
        model = deeplabv3_resnet50(weights=wts)
        replace_deeplab_head(model, NUM_UAVSCENES_CLASSES)
    model = model.to(args.device)
    head_params = list(model.classifier.parameters()) + list(model.aux_classifier.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr * args.backbone_lr_ratio},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

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

    use_amp = args.device == "cuda" and not args.no_amp
    scaler = GradScaler(enabled=use_amp)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    last_path = args.out.parent / "last.pt"
    best_miou = max(resume_best, -1.0)
    stagnant = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        lr_head = lr_at_epoch(epoch, args.lr, args.warmup_epochs, args.epochs, 0.01)
        lr_back = lr_head * args.backbone_lr_ratio
        opt.param_groups[0]["lr"] = lr_back
        opt.param_groups[1]["lr"] = lr_head
        lr_now = lr_head

        model.train()
        loss_tr = 0.0
        n = 0
        for x, y in train_ld:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                out = model(x)
                loss_main = ce_focal_loss(
                    out["out"],
                    y,
                    ce_w,
                    IGNORE_LABEL,
                    args.focal_gamma,
                )
                loss_aux = ce_focal_loss(
                    out["aux"],
                    y,
                    ce_w,
                    IGNORE_LABEL,
                    args.focal_gamma,
                )
                loss = loss_main + args.aux_weight * loss_aux
                if args.dice_weight > 0:
                    loss = loss + args.dice_weight * soft_dice_loss_logits(
                        out["out"], y, NUM_UAVSCENES_CLASSES, IGNORE_LABEL
                    )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(opt)
            scaler.update()
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
        print(
            f"epoch {epoch:03d}  lr_head={lr_now:.2e}  lr_back={lr_back:.2e}  "
            f"train_loss={loss_tr:.4f}  val_mIoU={v_miou:.4f}"
        )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "num_classes": NUM_UAVSCENES_CLASSES,
                "ignore_label": IGNORE_LABEL,
                "epoch": epoch,
                "val_miou": v_miou,
            },
            last_path,
        )

        if v_miou > best_miou + 1e-5:
            best_miou = v_miou
            stagnant = 0
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
        else:
            stagnant += 1
            if args.patience > 0 and stagnant >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs")
                break

    print(f"Done. Best val mIoU={best_miou:.4f} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
