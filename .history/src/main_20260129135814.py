import cv2
import os
import sys
import yaml
import numpy as np

# Path Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.detector import detector
from modules.parking_logic.spot_manager import SpotManager  # NEW IMPORT
from src.visualization import Visualizer

# CONFIGURATION
VIDEO_PATH = os.path.join("videos", "demo.mp4")
CONFIG_PATH = os.path.join("config", "config.yaml")
WINDOW_NAME = "Parking Vision System (Two-Model Architecture)"


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
    viz = Visualizer(config)

    # NEW: Initialize Spot Manager
    # This will try to load 'modules/parking_logic/spots.pt'
    # If missing, it generates a demo grid.
    spot_manager = SpotManager()

    # 2. INITIAL SCAN
    # We read one frame to detect the static parking spots
    success, first_frame = cap.read()
    if success:
        spot_manager.detect_spots_initial(first_frame)
    else:
        print("❌ Error: Could not read video for initial scan.")
        return

    track_history = {}
    print("✅ System Ready. Using Spot Detection + Car Tracking.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # -------------------------------------------------
        # 3. DETECTION & TRACKING
        # -------------------------------------------------
        cars = detector.detect(frame)

        # -------------------------------------------------
        # 4. CAR STABILITY LOGIC (Phase 5 Logic)
        # We still need this to know if a car is 'Parked' or 'Driving'
        # before we assign it to a spot.
        # -------------------------------------------------
        for car in cars:
            track_id = car["id"]
            cx, cy = get_bottom_center(car["bbox"])
            car["center_point"] = (cx, cy)

            if track_id not in track_history:
                track_history[track_id] = {
                    "stationary_counter": 0,
                    "last_position": (cx, cy),
                    "parked": False,
                }

            # Movement Check
            prev_cx, prev_cy = track_history[track_id]["last_position"]
            distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
            track_history[track_id]["last_position"] = (cx, cy)

            if distance < 5:
                track_history[track_id]["stationary_counter"] += 1
                if track_history[track_id]["stationary_counter"] > 100:
                    track_history[track_id]["stationary_counter"] = 100
            else:
                track_history[track_id]["stationary_counter"] = max(
                    0, track_history[track_id]["stationary_counter"] - 2
                )

            # Determine Parked Status (Score > 45)
            # Note: We removed the 'corridor' check because the Spot Manager handles location now.
            if track_history[track_id]["stationary_counter"] > 45:
                track_history[track_id]["parked"] = True
            elif track_history[track_id]["stationary_counter"] < 20:
                track_history[track_id]["parked"] = False

            car["is_parked"] = track_history[track_id]["parked"]

        # -------------------------------------------------
        # 5. SPOT OCCUPANCY LOGIC
        # -------------------------------------------------
        # This matches the stable cars to the detected spots
        spot_statuses = spot_manager.check_occupancy(cars)

        # Calculate Counts
        total_spots = len(spot_statuses)
        occupied_spots = sum(1 for s in spot_statuses if s["occupied"])

        # -------------------------------------------------
        # 6. VISUALIZATION
        # -------------------------------------------------
        frame = viz.draw_spots(frame, spot_statuses)  # Draw the grid
        frame = viz.draw_cars(frame, cars)  # Draw the cars
        frame = viz.draw_dashboard(frame, total_spots, occupied_spots)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
