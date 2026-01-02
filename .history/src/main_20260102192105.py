import cv2
import os
import sys

# --- PATH FIX: Allow python to find 'modules' folder ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# -------------------------------------------------------

# Import the new module you just made
from modules.detector import detector

# CONFIGURATION
VIDEO_PATH = os.path.join("videos", "demo.mp4")
WINDOW_NAME = "Parking Vision System (Phase 2)"


def main():
    print(f"Loading video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    print("✅ System Loaded. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # -------------------------------------------------
        # 1. DETECTION (The new part!)
        # -------------------------------------------------
        # This calls the function you wrote in detector.py
        cars = detector.detect(frame)

        # -------------------------------------------------
        # 2. VISUALIZATION (Temporary Debug View)
        # -------------------------------------------------
        # We draw the boxes here just to prove it works.
        # Later, we will move this to visualization.py
        for car in cars:
            x1, y1, x2, y2 = car["bbox"]
            conf = car["conf"]

            # Draw rectangle (Green)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw Label
            label = f"Car: {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # -------------------------------------------------
        # 3. DISPLAY
        # -------------------------------------------------
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
