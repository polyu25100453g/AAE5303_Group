#!/usr/bin/env python3
"""
Extract RTK-style ground truth trajectory from HKisland_GNSS03.bag.

This script builds a TUM-format trajectory using DJI local position and attitude:

  - Position:  /dji_osdk_ros/local_position  (geometry_msgs/PointStamped)
  - Attitude:  /dji_osdk_ros/attitude        (geometry_msgs/QuaternionStamped)

For each local_position sample, we pair it with the latest available attitude
and write a TUM-format line:

    t tx ty tz qx qy qz qw

This is suitable as ground truth input for evo and ORB-SLAM3 evaluation.

Usage:

    python3 extract_rtk_groundtruth.py HKisland_GNSS03.bag --output ground_truth.txt
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import rosbag  # type: ignore


POS_TOPIC = "/dji_osdk_ros/local_position"
ATT_TOPIC = "/dji_osdk_ros/attitude"


@dataclass
class Attitude:
    qx: float
    qy: float
    qz: float
    qw: float


def extract_groundtruth(bag_path: str, output_path: str) -> None:
    print(f"[extract_rtk_groundtruth] Bag: {bag_path}")
    print(f"[extract_rtk_groundtruth] Output: {output_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    latest_att: Optional[Attitude] = None
    count = 0
    pos_index = 0  # raw local_position samples counter (datastream is ~50 Hz)

    with rosbag.Bag(bag_path, "r") as bag, open(output_path, "w", encoding="utf-8") as f:
        f.write("# ground truth from HKisland_GNSS03 (local_position + attitude)\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")

        for topic, msg, t in bag.read_messages(topics=[POS_TOPIC, ATT_TOPIC]):
            if topic == ATT_TOPIC:
                # geometry_msgs/QuaternionStamped
                q = msg.quaternion
                latest_att = Attitude(qx=float(q.x), qy=float(q.y), qz=float(q.z), qw=float(q.w))
            elif topic == POS_TOPIC:
                # The local_position topic is high-rate (~50 Hz) while the evaluation
                # protocol (and original assignment description) assumes ~5 Hz RTK poses.
                # To avoid artificially inflating the denominator in the completeness
                # metric, we downsample the raw stream by a factor of 10 and only keep
                # every 10th position sample.
                pos_index += 1
                if pos_index % 10 != 0:
                    continue

                if latest_att is None:
                    continue

                # Use rosbag time `t` to keep timestamps consistent with extracted images.
                stamp = t.to_sec()
                x = float(msg.point.x)
                y = float(msg.point.y)
                z = float(msg.point.z)

                f.write(
                    f"{stamp:.6f} {x:.6f} {y:.6f} {z:.6f} "
                    f"{latest_att.qx:.8f} {latest_att.qy:.8f} "
                    f"{latest_att.qz:.8f} {latest_att.qw:.8f}\n"
                )
                count += 1

                if count % 200 == 0:
                    print(f"[extract_rtk_groundtruth] Written {count} poses...")

    print(f"[extract_rtk_groundtruth] Done. Total poses: {count}")
    print(f"[extract_rtk_groundtruth] Ground truth saved to: {output_path}")


def parse_args() -> Tuple[str, str]:
    parser = argparse.ArgumentParser(description="Extract RTK-like ground truth from HKisland_GNSS03.bag.")
    parser.add_argument("bag", help="Path to HKisland_GNSS03.bag")
    parser.add_argument("--output", required=True, help="Output TUM trajectory file (e.g., ground_truth.txt)")
    args = parser.parse_args()
    return args.bag, args.output


def main() -> None:
    bag_path, output_path = parse_args()
    extract_groundtruth(bag_path, output_path)


if __name__ == "__main__":
    main()

