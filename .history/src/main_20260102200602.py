import cv2
import os
import sys
import yaml
import numpy as np
from shapely.geometry import Point, Polygon

# Path Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.detector import detector

VIDEO_PATH = os.path.join("videos", "demo.mp4")
CONFIG_PATH = os.path.join("config", "config.yaml")
WINDOW_NAME = "Parking Vision System (Phase 4)"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_center(bbox):
    """Returns the center (x, y) of the bottom edge of the box"""
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) // 2)
    cy = int(y2)  # We use the bottom center (feet/wheels), not the middle
    return (cx, cy)


def main():
    # 1. LOAD CONFIG
    config = load_config()
    corridor_coords = config["zones"]["corridor"]

    # Create Shapely Polygon (for logic)
    corridor_poly = Polygon(corridor_coords)

    # Create NumPy Array (for drawing)
    corridor_array = np.array(corridor_coords, np.int32).reshape((-1, 1, 2))

    cap = cv2.VideoCapture(VIDEO_PATH)
    track_history = {}

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 2. DETECT & TRACK
        cars = detector.detect(frame)

        # 3. DRAW THE CORRIDOR (Blue line)
        cv2.polylines(
            frame, [corridor_array], isClosed=True, color=(255, 0, 0), thickness=2
        )

        for car in cars:
            track_id = car["id"]

            # Update History
            if track_id not in track_history:
                track_history[track_id] = {"frames": 0}
            track_history[track_id]["frames"] += 1
            frames_seen = track_history[track_id]["frames"]

            # -------------------------------------------------
            # 4. SPATIAL CHECK (Point in Polygon)
            # -------------------------------------------------
            cx, cy = get_center(car["bbox"])
            point = Point(cx, cy)

            # CHECK: Is this car inside our corridor?
            is_inside = corridor_poly.contains(point)

            # VISUALIZATION LOGIC
            # If inside: GREEN text
            # If outside: RED text
            status_color = (0, 0, 255)  # Red (Outside)
            status_text = "OUT"

            if is_inside:
                status_color = (0, 255, 0)  # Green (Inside)
                status_text = "IN"

            # Draw Box
            x1, y1, x2, y2 = car["bbox"]
            cv2.rectangle(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), status_color, 2
            )

            # Draw Center Point
            cv2.circle(frame, (cx, cy), 5, status_color, -1)

            # Draw Label
            label = f"ID:{track_id} | {status_text}"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
