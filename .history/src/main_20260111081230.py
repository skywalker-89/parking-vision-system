import cv2
import os
import sys
import yaml
from shapely.geometry import Point, Polygon

# Path Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.detector import detector
from src.visualization import Visualizer  # NEW IMPORT

# CONFIGURATION
VIDEO_PATH = os.path.join("videos", "demo.mp4")
CONFIG_PATH = os.path.join("config", "config.yaml")
WINDOW_NAME = "Parking Vision System (Phase 6)"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) // 2)
    cy = int(y2)
    return (cx, cy)


def main():
    # 1. SETUP
    config = load_config()
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Initialize Visualizer
    viz = Visualizer(config)

    # Setup Logic
    corridor_coords = config["zones"]["corridor"]
    corridor_poly = Polygon(corridor_coords)
    track_history = {}

    print("✅ System Ready. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # -------------------------------------------------
        # 2. LOGIC PIPELINE (No Drawing Here!)
        # -------------------------------------------------
        cars = detector.detect(frame)
        parked_count = 0

        for car in cars:
            track_id = car["id"]

            # Update history
            if track_id not in track_history:
                track_history[track_id] = {
                    "frames_seen": 0,
                    "parked": False,
                    "stationary_frames": 0,
                    "last_center": (cx, cy),
                }
            track_history[track_id]["frames_seen"] += 1

            # 🛑 NEW LOGIC: THE MOVEMENT CHECK

            prev_cx, prev_cy = track_history[track_id]["last_center"]

            # Calculate distance moved (Euclidean Distance)
            # Math: sqrt( (x2-x1)^2 + (y2-y1)^2 )
            distance_moved = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5

            # Define "Still": If moved less than 3 pixels, it's basically stopped.
            if distance_moved < 3:
                track_history[track_id]["stationary_frames"] += 1
            else:
                # If it moved, RESET the counter! This is the key fix.
                track_history[track_id]["stationary_frames"] = 0

            # Update "last_center" for the next frame's comparison
            track_history[track_id]["last_center"] = (cx, cy)

            # Check Geometry
            cx, cy = get_bottom_center(car["bbox"])
            car["center_point"] = (cx, cy)  # Save for visualizer

            is_inside = corridor_poly.contains(Point(cx, cy))
            frames_seen = track_history[track_id]["frames_seen"]

            # DECIDE STATUS
            if is_inside and frames_seen > 30:
                track_history[track_id]["parked"] = True
            else:
                track_history[track_id]["parked"] = False

            # Attach status to car object for the visualizer
            car["is_parked"] = track_history[track_id]["parked"]

            if car["is_parked"]:
                parked_count += 1

        # -------------------------------------------------
        # 3. VISUALIZATION PIPELINE (The Artist)
        # -------------------------------------------------
        # A. Draw the Zone
        frame = viz.draw_zone(frame, corridor_coords)

        # B. Draw the Cars
        frame = viz.draw_cars(frame, cars)

        # C. Draw the Dashboard
        frame = viz.draw_dashboard(frame, parked_count)

        # -------------------------------------------------
        # 4. DISPLAY
        # -------------------------------------------------
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
