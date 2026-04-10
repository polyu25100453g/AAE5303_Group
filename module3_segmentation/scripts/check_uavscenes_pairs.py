#!/usr/bin/env python3
"""Report how many images match id masks (same relative path or same stem)."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--masks-id-dir", type=Path, required=True)
    p.add_argument("--list-missing", type=int, default=0, help="Print up to N image paths without masks.")
    return p.parse_args()


def count_images(d: Path) -> int:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def main() -> int:
    args = parse_args()
    if not args.images_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.images_dir}")
    if not args.masks_id_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.masks_id_dir}")

    n_img = count_images(args.images_dir)
    pairs = collect_stems(args.images_dir, args.masks_id_dir)
    print(f"images (recursive): {n_img}")
    print(f"matched pairs:        {len(pairs)}")
    print(f"unmatched images:     {n_img - len(pairs)}")
    if args.list_missing > 0 and n_img > len(pairs):
        paired = {p[0] for p in pairs}
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        n = 0
        for p in sorted(args.images_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            if p not in paired:
                print(f"  missing mask: {p}")
                n += 1
                if n >= args.list_missing:
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
