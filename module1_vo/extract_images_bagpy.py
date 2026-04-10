import os
import argparse
from tqdm import tqdm
import cv2
import bagpy
from bagpy import bagreader
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Extract images from ROS bag using bagpy")
    parser.add_argument("bag_file", help="Path to ROS bag file")
    parser.add_argument("--output", default="extracted_images", help="Output directory")
    parser.add_argument("--topic", default=None, help="Image topic (if known)")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(f"Reading bag: {args.bag_file}")
    b = bagreader(args.bag_file)

    # 列出所有话题，帮助我们找到图像话题
    print("\nAvailable topics in the bag:")
    for topic in b.topic_table['Topics']:
        print("  ->", topic)

    # 如果你知道图像话题，可以在这里指定，例如：
    # image_topic = "/camera/image_raw"   # ←←← 需要修改成正确的
    # 否则我们先打印话题，你告诉我正确的 topic 再跑

    # 临时：先只打印话题，不提取（避免跑太久）
    print("\nPlease tell me which topic contains the RGB images (look for topics with 'image' or 'camera' or 'rgb').")
    print("Common examples: /camera/rgb/image_raw, /left/image_raw, /cam0/image_raw, /image")

if __name__ == "__main__":
    main()
