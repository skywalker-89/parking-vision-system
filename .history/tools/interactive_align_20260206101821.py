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
        dst_x2 = min(w, off_x + w_ref)

        # Source from reference (clipping logic)
        src_y1 = max(0, -off_y)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        src_x1 = max(0, -off_x)
        src_x2 = src_x1 + (dst_x2 - dst_x1)

        # Place reference onto canvas
        if dst_y2 > dst_y1 and dst_x2 > dst_x1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = ref_img[src_y1:src_y2, src_x1:src_x2]

        # Blend
        blended = cv2.addWeighted(video_frame, 0.6, canvas, 0.4, 0)

        # UI
        cv2.putText(
            blended,
            f"Offset X: {off_x} | Offset Y: {off_y}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Alignment Tool (WASD to move, ENTER to finish)", blended)

        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == 13:  # Enter
            print("\n" + "=" * 40)
            print(f"✅ FINAL OFFSETS FOUND:")
            print(f"OFFSET_X = {off_x}")
            print(f"OFFSET_Y = {off_y}")
            print("=" * 40 + "\n")
            print("👉 COPY THESE NUMBERS into 'spot_manager.py' now.")
            break

        # Movement Logic
        move_step = 10 if (cv2.waitKey(1) & 0xFF) else 1  # Check shift roughly

        if key == ord("w"):
            off_y -= step
        if key == ord("s"):
            off_y += step
        if key == ord("a"):
            off_x -= step
        if key == ord("d"):
            off_x += step

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
