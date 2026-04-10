#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if src.resolve() == dst.resolve():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare module2 reconstruction inputs from module1 outputs.")
    parser.add_argument(
        "--module1-root",
        default="../module1_vo",
        help="Path to module1_vo root directory (default: ../module1_vo)",
    )
    parser.add_argument(
        "--images",
        default="",
        help="Override image directory path. If empty, auto-detect from module1.",
    )
    parser.add_argument(
        "--trajectory",
        default="",
        help="Override trajectory file path. If empty, use module1 output if available.",
    )
    parser.add_argument(
        "--camera-config",
        default="",
        help="Override camera config path. If empty, use module1 docs/camera_config.yaml if available.",
    )
    args = parser.parse_args()

    module2_root = Path(__file__).resolve().parents[1]
    module1_root = (module2_root / args.module1_root).resolve()

    input_root = module2_root / "input"
    images_link = input_root / "images"
    traj_dir = input_root / "trajectories"
    cam_dir = input_root / "camera"
    ensure_clean_dir(input_root)
    ensure_clean_dir(traj_dir)
    ensure_clean_dir(cam_dir)
    ensure_clean_dir(module2_root / "output")
    ensure_clean_dir(module2_root / "logs")

    candidate_images = []
    if args.images:
        candidate_images.append(Path(args.images).resolve())
    workspace_root = module2_root.parents[1]
    candidate_images.extend(
        [
            (module1_root / "extracted_data").resolve(),
            (module1_root / "dataset/AMtown02/images").resolve(),
            (module1_root / "dataset/HKisland_GNSS03/images").resolve(),
            (workspace_root / "AAE5303_assignment2_orbslam3_demo-/dataset/AMtown02/images").resolve(),
            (workspace_root / "AAE5303_assignment2_orbslam3_demo-/dataset/HKisland_GNSS03/images").resolve(),
        ]
    )
    image_src = next((p for p in candidate_images if p.exists() and p.is_dir()), None)
    if image_src is None:
        raise FileNotFoundError("Cannot find image directory. Use --images to set it explicitly.")

    if images_link.exists() or images_link.is_symlink():
        images_link.unlink()
    images_link.symlink_to(image_src, target_is_directory=True)

    traj_candidates = []
    if args.trajectory:
        traj_candidates.append(Path(args.trajectory).resolve())
    traj_candidates.extend(
        [
            (module2_root / "input/trajectories/CameraTrajectory_sec.txt").resolve(),
            (module1_root / "output/CameraTrajectory_sec.txt").resolve(),
            (module1_root / "output/CameraTrajectory.txt").resolve(),
            (workspace_root / "AAE5303_assignment2_orbslam3_demo-/output/CameraTrajectory_sec.txt").resolve(),
        ]
    )
    traj_src = next((p for p in traj_candidates if p.exists() and p.is_file()), None)
    traj_ok = False
    if traj_src:
        traj_ok = copy_if_exists(traj_src, traj_dir / traj_src.name)

    cam_candidates = []
    if args.camera_config:
        cam_candidates.append(Path(args.camera_config).resolve())
    cam_candidates.extend(
        [
            (module1_root / "docs/camera_config.yaml").resolve(),
            (module1_root / "docs/camera_config_mono_fallback.yaml").resolve(),
        ]
    )
    cam_src = next((p for p in cam_candidates if p.exists() and p.is_file()), None)
    cam_ok = False
    if cam_src:
        cam_ok = copy_if_exists(cam_src, cam_dir / cam_src.name)

    rgb_candidates = [
        (module1_root / "dataset/AMtown02/rgb.txt").resolve(),
        (workspace_root / "AAE5303_assignment2_orbslam3_demo-/dataset/AMtown02/rgb.txt").resolve(),
    ]
    rgb_src = next((p for p in rgb_candidates if p.exists() and p.is_file()), None)
    rgb_ok = False
    if rgb_src:
        rgb_ok = copy_if_exists(rgb_src, input_root / "rgb.txt")

    image_count = len(list(image_src.glob("*.png"))) + len(list(image_src.glob("*.jpg")))
    manifest = {
        "module1_root": str(module1_root),
        "images_dir": str(image_src),
        "images_count": image_count,
        "trajectory_found": traj_ok,
        "trajectory_file": str(traj_src) if traj_src else "",
        "camera_config_found": cam_ok,
        "camera_config_file": str(cam_src) if cam_src else "",
        "rgb_found": rgb_ok,
        "rgb_file": str(rgb_src) if rgb_src else "",
    }
    (input_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Prepared module2 inputs.")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
