import cv2
import os
import sys

# Path Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

VIDEO_PATH = os.path.join("videos", "demo.mp4")
REF_PATH = os.path.join("config", "reference.jpg")


def main():
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

    # 3. Resize Reference to match Video (Exactly what the AI does)
    h, w, _ = video_frame.shape
    ref_resized = cv2.resize(ref_img, (w, h))

    # 4. Blend them 50/50
    blended = cv2.addWeighted(video_frame, 0.5, ref_resized, 0.5, 0)

    # 5. Show Result
    print(f"🎥 Video Resolution: {w}x{h}")
    print(f"🖼️ Reference Resolution: {ref_img.shape[1]}x{ref_img.shape[0]}")

    cv2.imshow("Alignment Check (Press Q to quit)", blended)
    print("👀 LOOK AT THE WINDOW!")
    print(
        "   - If the image looks blurry/ghosted -> Your Reference DOES NOT match the Video."
    )
    print("   - If it looks crisp/perfect -> Code bug.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
