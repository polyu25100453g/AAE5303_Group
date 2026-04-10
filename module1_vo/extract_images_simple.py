import os
import argparse
import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    parser = argparse.ArgumentParser(description="Extract images from ROS bag")
    parser.add_argument("bag_file", help="Path to the ROS bag file")
    parser.add_argument("--output", default="extracted_images", help="Output directory")
    parser.add_argument("--topic", default="/camera/image_raw", help="Image topic name")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    bridge = CvBridge()
    count = 0

    print(f"Opening bag: {args.bag_file}")
    bag = rosbag.Bag(args.bag_file, 'r')

    for topic, msg, t in bag.read_messages(topics=[args.topic]):
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            filename = os.path.join(args.output, f"frame_{count:06d}.jpg")
            cv2.imwrite(filename, cv_img)
            count += 1
            if count % 500 == 0:
                print(f"Extracted {count} images...")
        except Exception as e:
            print(f"Error processing frame {count}: {e}")

    bag.close()
    print(f"Done! Extracted {count} images to {args.output}")

if __name__ == "__main__":
    main()
