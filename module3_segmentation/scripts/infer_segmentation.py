#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image
from torch.cuda.amp import autocast
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from uavscenes_cmap import NUM_UAVSCENES_CLASSES, palette_uavscenes


def collect_images(input_dir: Path) -> list[Path]:
    # module1_vo extract_images_final.py writes TUM-style layout: <output>/rgb/<timestamp>.png
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    imgs = [
        p
        for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in exts
    ]
    return imgs


def voc_palette() -> np.ndarray:
    # Standard-ish 21-class palette for VOC-style labels.
    palette = np.array(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ],
        dtype=np.uint8,
    )
    return palette


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    idx = np.clip(mask, 0, palette.shape[0] - 1)
    return palette[idx]


def replace_deeplab_head(model: nn.Module, num_classes: int) -> None:
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    in_aux = model.aux_classifier[-1].in_channels
    model.aux_classifier[-1] = nn.Conv2d(in_aux, num_classes, kernel_size=1)


def palette_for_num_classes(num_classes: int) -> np.ndarray:
    if num_classes == NUM_UAVSCENES_CLASSES:
        return palette_uavscenes()
    if num_classes <= 21:
        return voc_palette()
    rng = np.random.default_rng(0)
    base = voc_palette()
    extra = rng.integers(0, 255, size=(num_classes - base.shape[0], 3), dtype=np.uint8)
    return np.vstack([base, extra])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 3 baseline semantic segmentation.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Fine-tuned weights from train_deeplab_uavscenes.py (26-class UAVScenes).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override class count; if --checkpoint is set, defaults to value stored in checkpoint.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Test-time augmentation: average logits with horizontal flip (+~0.5–2 mIoU typical).",
    )
    parser.add_argument(
        "--tta-ms",
        action="store_true",
        help="Multi-scale TTA: average logits at scales 0.875 / 1.0 / 1.125 (original resolution). Slower.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N images (0 = silent).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="CUDA autocast float16 forward (faster, tiny metric change).",
    )
    return parser.parse_args()


def _logits_to_full_res(
    model: nn.Module,
    pil_img: Image.Image,
    tfm,
    device: torch.device,
    scales: list[float],
    with_hflip: bool,
    use_amp: bool,
) -> torch.Tensor:
    w, h = pil_img.size
    acc = None
    for s in scales:
        w2 = max(32, int(round(w * s)))
        h2 = max(32, int(round(h * s)))
        im = pil_img.resize((w2, h2), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)

        def one_forward(pil_i: Image.Image) -> torch.Tensor:
            x = tfm(pil_i).unsqueeze(0).to(device)
            with autocast(enabled=use_amp):
                lo = model(x)["out"]
            return Fn.interpolate(lo, size=(h, w), mode="bilinear", align_corners=False)

        lo = one_forward(im)
        acc = lo if acc is None else acc + lo
        if with_hflip:
            imf = im.transpose(Image.FLIP_LEFT_RIGHT)
            lf = one_forward(imf)
            lf = torch.flip(lf, dims=[3])
            acc = acc + lf
    denom = float(len(scales)) * (2.0 if with_hflip else 1.0)
    return acc / denom


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    images = collect_images(args.input_dir)
    if not images:
        raise RuntimeError(f"No images found in {args.input_dir}")
    if args.max_images > 0:
        images = images[: args.max_images]

    masks_dir = args.output_dir / "masks"
    color_dir = args.output_dir / "color_masks"
    overlay_dir = args.output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    ckpt: dict | None = None
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        try:
            ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location=args.device)
    num_classes = args.num_classes
    if ckpt is not None and isinstance(ckpt, dict) and "num_classes" in ckpt:
        if num_classes is None:
            num_classes = int(ckpt["num_classes"])
    if num_classes is None:
        num_classes = 21

    model = deeplabv3_resnet50(weights=weights)
    if num_classes != 21:
        replace_deeplab_head(model, num_classes)
    if ckpt is not None and isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(args.device).eval()
    tfm = weights.transforms()
    palette = palette_for_num_classes(num_classes)

    processed = 0
    nearest = getattr(Image, "Resampling", Image).NEAREST
    ms_scales = [0.875, 1.0, 1.125]
    use_amp = bool(args.fp16 and args.device == "cuda")

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        with torch.no_grad():
            if args.tta_ms:
                out = _logits_to_full_res(
                    model, img, tfm, args.device, ms_scales, with_hflip=args.tta, use_amp=use_amp
                )
                pred_full = out[0].argmax(0).cpu().numpy().astype(np.uint8)
            else:
                x = tfm(img).unsqueeze(0).to(args.device)
                with autocast(enabled=use_amp):
                    if args.tta:
                        lo = model(x)["out"]
                        xf = torch.flip(x, dims=[3])
                        lf = torch.flip(model(xf)["out"], dims=[3])
                        out = (lo + lf) * 0.5
                    else:
                        out = model(x)["out"]
                pred = out[0].argmax(0).cpu().numpy().astype(np.uint8)
                pred_full = np.array(
                    Image.fromarray(pred, mode="L").resize((w, h), nearest),
                    dtype=np.uint8,
                )
        color = colorize_mask(pred_full, palette)
        base = np.array(img, dtype=np.uint8)
        overlay = (0.6 * base + 0.4 * color).astype(np.uint8)

        stem = img_path.stem
        Image.fromarray(pred_full, mode="L").save(masks_dir / f"{stem}.png")
        Image.fromarray(color, mode="RGB").save(color_dir / f"{stem}.png")
        Image.fromarray(overlay, mode="RGB").save(overlay_dir / f"{stem}.png")
        processed += 1
        if args.progress_every > 0 and processed % args.progress_every == 0:
            print(f"infer: {processed}/{len(images)}", flush=True)

    summary = {
        "model": "torchvision.deeplabv3_resnet50",
        "weights": "DeepLabV3_ResNet50_Weights.DEFAULT",
        "num_classes": num_classes,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "device": args.device,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "num_images_processed": processed,
        "tta": bool(args.tta),
        "tta_ms": bool(args.tta_ms),
        "fp16_autocast": use_amp,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
