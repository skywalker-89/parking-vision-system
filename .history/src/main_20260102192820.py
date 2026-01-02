import cv2
import os
import sys

# Path Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.detector import detector

VIDEO_PATH = os.path.join("videos", "demo.mp4")
WINDOW_NAME = "Parking Vision System (Phase 3)"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    # -------------------------------------------------
    # 1. INIT HISTORY DICTIONARY
    # Key = Track ID (e.g., 42)
    # Value = How many frames we've seen it
    # -------------------------------------------------
    track_history = {}

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 2. RUN TRACKING
        cars = detector.detect(frame)

        # 3. UPDATE HISTORY
        for car in cars:
            track_id = car["id"]

            # If we've never seen this car ID before, add it
            if track_id not in track_history:
                track_history[track_id] = {"frames": 0}

            # Increment the counter
            track_history[track_id]["frames"] += 1

            # Add the frame count to the car object so we can draw it
            car["frames_seen"] = track_history[track_id]["frames"]

        # 4. VISUALIZATION (Updated for IDs)
        for car in cars:
            x1, y1, x2, y2 = car["bbox"]
            track_id = car["id"]
            frames_seen = car["frames_seen"]

            # Color logic:
            # If seen < 30 frames: RED (Moving/Unstable)
            # If seen > 30 frames: GREEN (Stable/Parked?)
            color = (0, 0, 255)  # Red
            if frames_seen > 30:
                color = (0, 255, 0)  # Green

            # Draw Box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw ID + Counter
            label = f"ID:{track_id} | {frames_seen}"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
