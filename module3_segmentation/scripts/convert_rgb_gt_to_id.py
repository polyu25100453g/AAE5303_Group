#!/usr/bin/env python3
"""Convert UAVScenes RGB label images to single-channel class-id PNGs (0–25, unknown→255)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from uavscenes_cmap import CMAP, IGNORE_LABEL, NUM_UAVSCENES_CLASSES


def collect_images(d: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in exts and p.is_file())


def rgb_to_id(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected RGB image HxWx3")
    out = np.full(arr.shape[:2], IGNORE_LABEL, dtype=np.uint8)
    for cid, info in CMAP.items():
        rgb = np.array(info["RGB"], dtype=np.uint8)
        m = np.all(arr == rgb, axis=-1)
        out[m] = cid
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True, help="RGB label images (recursive).")
    p.add_argument("--output-dir", type=Path, required=True, help="Write id PNGs, same relative paths.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = collect_images(args.input_dir)
    if not paths:
        raise RuntimeError(f"No images under {args.input_dir}")

    for src in paths:
        rel = src.relative_to(args.input_dir)
        dst = args.output_dir / rel.with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        rgb = np.array(Image.open(src).convert("RGB"), dtype=np.uint8)
        ids = rgb_to_id(rgb)
        unknown = float(np.sum(ids == IGNORE_LABEL)) / ids.size * 100.0
        if unknown > 50:
            print(f"warn: {src}: {unknown:.1f}% pixels unmatched (check color format / cmap)")
        Image.fromarray(ids, mode="L").save(dst)
    print(f"Converted {len(paths)} masks → {args.output_dir} (classes 0–{NUM_UAVSCENES_CLASSES - 1}, else {IGNORE_LABEL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
