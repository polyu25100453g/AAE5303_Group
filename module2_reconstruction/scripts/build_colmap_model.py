#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from pathlib import Path

import pycolmap


def _parse_camera_config(path: Path) -> dict:
    vals = {}
    if not path.exists():
        return vals
    patt = re.compile(r"^(Camera(?:1)?\.(fx|fy|cx|cy|k1|k2|p1|p2|k3))\s*:\s*([-+eE0-9\.]+)")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = patt.match(line.strip())
        if not m:
            continue
        key = m.group(2)
        vals[key] = float(m.group(3))
    return vals


def make_selected_images(src_dir: Path, dst_dir: Path, max_images: int, start_index: int, stride: int) -> int:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")))
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {src_dir}")

    start = max(0, min(start_index, len(all_imgs) - 1))
    stride = max(1, stride)
    sliced = all_imgs[start::stride]
    if max_images > 0 and len(sliced) > max_images:
        selected = sliced[:max_images]
    else:
        selected = sliced

    for p in selected:
        (dst_dir / p.name).symlink_to(p.resolve())
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build COLMAP sparse model with pycolmap.")
    parser.add_argument("--images", default="input/images", help="Input images directory")
    parser.add_argument("--workspace", default="input/colmap_project", help="COLMAP workspace directory")
    parser.add_argument("--max-images", type=int, default=800, help="Max images to use (0 means all)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for feature/matching/mapping")
    parser.add_argument("--max-image-size", type=int, default=1280, help="Resize max image side for SIFT extraction")
    parser.add_argument("--max-features", type=int, default=2048, help="Max SIFT features per image")
    parser.add_argument("--sequential-overlap", type=int, default=20, help="Frame overlap for sequential matching")
    parser.add_argument("--start-index", type=int, default=0, help="Start image index within sorted list")
    parser.add_argument("--stride", type=int, default=1, help="Sampling stride in sorted image list")
    parser.add_argument("--camera-config", default="input/camera/camera_config.yaml", help="Camera config yaml")
    args = parser.parse_args()

    root = Path.cwd()
    images_src = (root / args.images).resolve()
    workspace = (root / args.workspace).resolve()
    images_sel = workspace / "images"
    db_path = workspace / "database.db"
    sparse_root = workspace / "sparse"

    workspace.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    if sparse_root.exists():
        shutil.rmtree(sparse_root)
    sparse_root.mkdir(parents=True, exist_ok=True)

    selected_count = make_selected_images(images_src, images_sel, args.max_images, args.start_index, args.stride)
    print(f"Selected images: {selected_count}")

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.num_threads = int(args.threads)
    extraction_options.max_image_size = int(args.max_image_size)
    extraction_options.sift.max_num_features = int(args.max_features)

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.num_threads = int(args.threads)
    matching_options.max_num_matches = 8192

    pipeline_options = pycolmap.IncrementalPipelineOptions()
    pipeline_options.num_threads = int(args.threads)
    pipeline_options.mapper.num_threads = int(args.threads)
    pipeline_options.min_model_size = 3

    reader_options = pycolmap.ImageReaderOptions()
    cam_vals = _parse_camera_config((root / args.camera_config).resolve())
    if all(k in cam_vals for k in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")):
        reader_options.camera_model = "OPENCV"
        reader_options.camera_params = ",".join(
            [
                f"{cam_vals['fx']}",
                f"{cam_vals['fy']}",
                f"{cam_vals['cx']}",
                f"{cam_vals['cy']}",
                f"{cam_vals['k1']}",
                f"{cam_vals['k2']}",
                f"{cam_vals['p1']}",
                f"{cam_vals['p2']}",
            ]
        )

    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(images_sel),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
    )
    pairing_options = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = int(args.sequential_overlap)
    pycolmap.match_sequential(
        database_path=str(db_path),
        matching_options=matching_options,
        pairing_options=pairing_options,
    )
    maps = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(images_sel),
        output_path=str(sparse_root),
        options=pipeline_options,
    )

    map_count = len(maps)
    if map_count <= 0:
        raise RuntimeError("No sparse model reconstructed. Try fewer/max-images or better image quality.")

    best_idx = max(maps.keys(), key=lambda k: maps[k].num_reg_images())
    best = maps[best_idx]
    summary = {
        "workspace": str(workspace),
        "database": str(db_path),
        "images_selected": selected_count,
        "sparse_model_count": map_count,
        "threads": int(args.threads),
        "max_image_size": int(args.max_image_size),
        "max_features": int(args.max_features),
        "sequential_overlap": int(args.sequential_overlap),
        "start_index": int(args.start_index),
        "stride": int(args.stride),
        "camera_model": reader_options.camera_model,
        "best_model_id": int(best_idx),
        "best_model_registered_images": int(best.num_reg_images()),
        "best_model_points3D": int(best.num_points3D()),
    }
    (workspace / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
