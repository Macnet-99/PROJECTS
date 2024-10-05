
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, default="/content/image.jpg", help="path to input image")
    ap.add_argument("-y", "--yolo", required=False, default="/content/", help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
    return vars(ap.parse_args())

if __name__ == "__main__":
    args = parse_args()
