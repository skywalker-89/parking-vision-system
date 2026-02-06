import cv2
import os
import sys

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "demo.mp4")
REF_PATH = os.path.join(BASE_DIR, "config", "reference.jpg")


def main():
    print(f"🎥 Video: {VIDEO_PATH}")
    print(f"🖼️ Reference: {REF_PATH}")

    # 1. Load Video Frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, video_frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Error: Could not read demo.mp4")
        return

    # 2. Load Reference Image
    if not os.path.exists(REF_PATH):
        print(f"❌ Error: Could not find {REF_PATH}")
        return

    ref_img = cv2.imread(REF_PATH)

    # 3. Resize Reference to match Video (Exactly what the Spot Manager does)
    h, w, _ = video_frame.shape
    ref_resized = cv2.resize(ref_img, (w, h))

    # 4. Blend them 50/50
    # This creates a "Ghost" image to see differences
    blended = cv2.addWeighted(video_frame, 0.6, ref_resized, 0.4, 0)

    # 5. Show Result
    print("-" * 30)
    print(f"🎥 Video Resolution:     {w}x{h}")
    print(f"🖼️ Reference Resolution: {ref_img.shape[1]}x{ref_img.shape[0]}")

    if w != ref_img.shape[1] or h != ref_img.shape[0]:
        print("⚠️ WARNING: RESOLUTIONS DO NOT MATCH!")
        print("   The code is stretching your image to fit the video.")
        print("   This 'Squashing' is likely moving the spots.")
    print("-" * 30)

    cv2.imshow("Alignment Check (Press Q to quit)", blended)
    print("👀 LOOK AT THE WINDOW!")
    print(
        "   - If you see 'Double Vision' (Ghosting) -> The camera moved or Aspect Ratio is wrong."
    )
    print("   - If the lines match perfectly -> The code should be working.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
