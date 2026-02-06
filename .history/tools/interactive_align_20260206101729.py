import cv2
import os
import sys
import numpy as np

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "demo.mp4")
REF_PATH = os.path.join(BASE_DIR, "config", "reference.jpg")

def main():
    # 1. Load Video Frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, video_frame = cap.read()
    cap.release()
    if not ret:
        print("❌ Error reading video.")
        return

    # 2. Load Reference
    if not os.path.exists(REF_PATH):
        print("❌ Error reading reference.jpg")
        return
    ref_img = cv2.imread(REF_PATH)

    # Resize ref to match video dimensions initially
    h, w = video_frame.shape[:2]
    ref_img = cv2.resize(ref_img, (w, h))

    # Initial Offsets
    off_x, off_y = 0, 0
    step = 1

    print("controls:")
    print("  [W/A/S/D] - Move Image")
    print("  [SHIFT]   - Move Faster")
    print("  [ENTER]   - Print Offsets and Quit")
    print("  [Q]       - Quit without saving")

    while True:
        # Create a blank canvas matching the video
        canvas = np.zeros_like(video_frame)
        
        # Calculate placing coordinates
        h_ref, w_ref = ref_img.shape[:2]
        
        # Destination on canvas (clipping logic)
        dst_y1 = max(0, off_y)
        dst_y2 = min(h, off_y + h_ref)
        dst_x1 = max(0, off_x)
        dst_x2 = min(w, off_x