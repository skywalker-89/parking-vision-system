import cv2
import os
import sys
import yaml
import numpy as np
from shapely.geometry import Point, Polygon

# Path Fix: Allows Python to find the 'modules' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.detector import detector
from src.visualization import Visualizer

# CONFIGURATION
# ---------------------------------------------------------
VIDEO_PATH = os.path.join("videos", "demo.mp4")
CONFIG_PATH = os.path.join("config", "config.yaml")
WINDOW_NAME = "Parking Vision System (Stabilized)"
# ---------------------------------------------------------


def load_config():
    """Loads the YAML configuration file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_bottom_center(bbox):
    """
    Calculates the bottom-center point of a bounding box.
    This represents the location of the car's tires/ground contact.
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) // 2)
    cy = int(y2)
    return (cx, cy)


def main():
    # 1. SETUP & INITIALIZATION
    config = load_config()
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Initialize the Visualizer (handles all drawing)
    viz = Visualizer(config)

    # Setup Geometric Logic
    corridor_coords = config["zones"]["corridor"]
    corridor_poly = Polygon(corridor_coords)

    # HISTORY TRACKER
    # Stores state for every car ID (position, stationary time, etc.)
    track_history = {}

    print("✅ System Ready. Logic Tuned: High stability for parked cars.")

    while True:
        success, frame = cap.read()
        if not success:
            print("End of video or read error.")
            break

        # -------------------------------------------------
        # 2. DETECTION & LOGIC PIPELINE
        # -------------------------------------------------
        cars = detector.detect(frame)
        parked_count = 0

        for car in cars:
            track_id = car["id"]

            # A. Get Current Position
            cx, cy = get_bottom_center(car["bbox"])
            car["center_point"] = (cx, cy)  # Save for visualizer

            # B. Initialize History if this is a new car
            if track_id not in track_history:
                track_history[track_id] = {
                    "stationary_counter": 0,  # Score: 0 = Moving, 100 = Solidly Parked
                    "last_position": (cx, cy),  # Where was it in the previous frame?
                    "parked": False,  # Current status
                }

            # C. CALCULATE MOVEMENT (Stabilized)
            # Retrieve the position from the previous frame
            prev_cx, prev_cy = track_history[track_id]["last_position"]

            # Calculate Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2)
            distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5

            # Update history with current position for the next loop
            track_history[track_id]["last_position"] = (cx, cy)

            # D. CHECK STATUS
            # Rule 1: Movement Check with Higher Tolerance
            # We increased tolerance to 5 pixels (handles "shaky" boxes better)
            if distance < 5:
                track_history[track_id]["stationary_counter"] += 1
                # Cap the counter at 100 so it doesn't take forever to turn red when leaving
                if track_history[track_id]["stationary_counter"] > 100:
                    track_history[track_id]["stationary_counter"] = 100
            else:
                # IT MOVED! Decay the counter slowly.
                # Only subtract 2. This means a car needs to move for ~1 second straight
                # to lose its "Parked" status. Glitches won't kill the status.
                track_history[track_id]["stationary_counter"] = max(
                    0, track_history[track_id]["stationary_counter"] - 2
                )

            # Rule 2: Is it inside the parking zone?
            is_inside = corridor_poly.contains(Point(cx, cy))

            # Rule 3: Has it been inside AND stationary for enough time?
            stationary_score = track_history[track_id]["stationary_counter"]

            # HYSTERESIS LOGIC:
            # - To BECOME parked, you need a score > 45 (approx 1.5s still)
            # - To STAY parked, you just need a score > 20
            # This prevents flickering when the score hovers around 45.
            if is_inside:
                if stationary_score > 45:
                    track_history[track_id]["parked"] = True
                elif stationary_score < 20:
                    track_history[track_id]["parked"] = False
                # If score is between 20 and 45, keep the PREVIOUS status.
            else:
                # If it leaves the zone, it's definitely not parked.
                track_history[track_id]["parked"] = False

            # Attach final status to the car object for the visualizer to read
            car["is_parked"] = track_history[track_id]["parked"]

            if car["is_parked"]:
                parked_count += 1

        # -------------------------------------------------
        # 3. VISUALIZATION PIPELINE
        # -------------------------------------------------
        # Draw the blue parking zone
        frame = viz.draw_zone(frame, corridor_coords)

        # Draw the cars (Red = Driving, Green = Parked)
        frame = viz.draw_cars(frame, cars)

        # Draw the top dashboard with counts
        frame = viz.draw_dashboard(frame, parked_count)

        # -------------------------------------------------
        # 4. DISPLAY
        # -------------------------------------------------
        cv2.imshow(WINDOW_NAME, frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
