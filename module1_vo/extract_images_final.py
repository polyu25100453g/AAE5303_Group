#!/usr/bin/env python3
"""
Extract monocular images from HKisland_GNSS03.bag for ORB-SLAM3.

This script reads the ROS bag and saves images from the left camera
topic `/left_camera/image/compressed` into a TUM-style dataset folder
that can be consumed by ORB-SLAM3's `mono_tum` example.

Output structure (root = --output):

    output/
      rgb/
        <timestamp>.png
        ...
      rgb.txt        # "timestamp relative_path" (TUM format)

Usage (as in the assignment README):

    python3 extract_images_final.py HKisland_GNSS03.bag --output extracted_data
"""

import argparse
import os
from typing import Tuple

import cv2
import rospy  # type: ignore
import rosbag  # type: ignore
from cv_bridge import CvBridge  # type: ignore


IMAGE_TOPIC = "/left_camera/image/compressed"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_images(bag_path: str, output_dir: str) -> None:
    print(f"[extract_images_final] Bag: {bag_path}")
    print(f"[extract_images_final] Output dir: {output_dir}")

    rgb_dir = os.path.join(output_dir, "rgb")
    ensure_dir(rgb_dir)

    rgb_txt_path = os.path.join(output_dir, "rgb.txt")

    bridge = CvBridge()
    count = 0

    with rosbag.Bag(bag_path, "r") as bag, open(rgb_txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write("# timestamp filename\n")

        for topic, msg, t in bag.read_messages(topics=[IMAGE_TOPIC]):
            # Use rosbag time `t` to keep timestamps consistent across topics.
            # (Some messages carry header stamps in a different clock/time base.)
            stamp = t.to_sec()
            ts_str = f"{stamp:.6f}"
            filename = f"{ts_str}.png"
            rel_path = os.path.join("rgb", filename)
            out_path = os.path.join(rgb_dir, filename)

            # Decode compressed image to BGR
            cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite(out_path, cv_img)

            f_txt.write(f"{ts_str} {rel_path}\n")
            count += 1

            if count % 100 == 0:
                print(f"[extract_images_final] Saved {count} images...")

    print(f"[extract_images_final] Done. Total images: {count}")
    print(f"[extract_images_final] rgb.txt written to: {rgb_txt_path}")


def parse_args() -> Tuple[str, str]:
    parser = argparse.ArgumentParser(description="Extract images from HKisland_GNSS03.bag for ORB-SLAM3.")
    parser.add_argument("bag", help="Path to HKisland_GNSS03.bag")
    parser.add_argument("--output", required=True, help="Output directory (e.g., /data/extracted_data)")
    args = parser.parse_args()
    return args.bag, args.output


def main() -> None:
    bag_path, output_dir = parse_args()
    extract_images(bag_path, output_dir)


if __name__ == "__main__":
    main()

