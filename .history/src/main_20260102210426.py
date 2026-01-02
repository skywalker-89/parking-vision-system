import cv2
import os
import sys
import yaml
import numpy as np
from shapely.geometry import Point, Polygon

# Path Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.detector import detector

# CONFIGURATION
# ---------------------------------------------------------
VIDEO_PATH = os.path.join("videos", "demo.mp4")
CONFIG_PATH = os.path.join("config", "config.yaml")
WINDOW_NAME = "Parking Vision System (Phase 5)"
# ---------------------------------------------------------


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_bottom_center(bbox):
    """
    Calculates the bottom-center point of a bounding box.
    Crucial for perspective: We care where the TIRES are, not the roof.
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) // 2)
    cy = int(y2)  # The very bottom edge
    return (cx, cy)


def main():
    # 1. SETUP
    config = load_config()
    corridor_coords = config["zones"]["corridor"]
    corridor_poly = Polygon(corridor_coords)
    corridor_array = np.array(corridor_coords, np.int32).reshape((-1, 1, 2))

    cap = cv2.VideoCapture(VIDEO_PATH)

    # Track History: Stores metadata for every car ID
    track_history = {}

    print("✅ Phase 5 Logic Loaded: Counting Parked Cars...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 2. DETECT & TRACK
        cars = detector.detect(frame)

        # 3. DRAW ZONES
        # Draw the parking corridor in Blue
        cv2.polylines(
            frame, [corridor_array], isClosed=True, color=(255, 100, 0), thickness=2
        )

        # Count only valid parked cars for display
        parked_count = 0

        for car in cars:
            track_id = car["id"]

            # --- UPDATE HISTORY ---
            if track_id not in track_history:
                track_history[track_id] = {"frames_seen": 0, "parked": False}

            track_history[track_id]["frames_seen"] += 1
            frames_seen = track_history[track_id]["frames_seen"]

            # --- PHASE 5 LOGIC: PARKED STATUS ---
            # 1. Calculate Bottom-Center
            cx, cy = get_bottom_center(car["bbox"])
            point = Point(cx, cy)

            # 2. Check Geometry (Inside Zone?)
            is_inside = corridor_poly.contains(point)

            # 3. Apply The "Parked" Definition
            # Rule: Must be inside zone AND seen for > 30 frames (approx 1 sec)
            if is_inside and frames_seen > 30:
                track_history[track_id]["parked"] = True
            else:
                track_history[track_id]["parked"] = False

            # --- VISUALIZATION ---
            is_parked = track_history[track_id]["parked"]

            if is_parked:
                # PARKED STATUS (Green)
                color = (0, 255, 0)
                label = f"ID:{track_id} | PARKED"
                parked_count += 1
            else:
                # DRIVING STATUS (Red/Orange)
                color = (0, 0, 255)
                label = f"ID:{track_id} | DRIVING"

            # Draw Box
            x1, y1, x2, y2 = car["bbox"]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw Center Point (The "Anchor")
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # Yellow Dot

            # Draw Text
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # 4. DASHBOARD (Top Left)
        # Show total count of currently parked cars
        cv2.rectangle(frame, (0, 0), (250, 50), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"PARKED CARS: {parked_count}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
