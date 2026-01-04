import cv2
import os
import sys
import yaml

# Path Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.detector import detector
from modules.logic import ZoneManager
from src.visualization import Visualizer

# CONFIGURATION
VIDEO_PATH = os.path.join("videos", "demo.mp4")
CONFIG_PATH = os.path.join("config", "config.yaml")
WINDOW_NAME = "Parking Vision System - Multi-Zone"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    # 1. SETUP
    config = load_config()
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Initialize Visualizer
    viz = Visualizer(config)

    # Initialize Zone Manager (replaces old single-zone logic)
    zone_manager = ZoneManager(config)

    print("System Ready. Press 'q' to quit.")
    print(f"Monitoring {len(zone_manager.get_all_zones())} parking zones\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # -------------------------------------------------
        # 2. DETECTION & LOGIC PIPELINE
        # -------------------------------------------------
        cars = detector.detect(frame)

        # Add center point to each car for visualization
        from modules.logic import get_bottom_center
        for car in cars:
            car["center_point"] = get_bottom_center(car["bbox"])

        # Update all zones with detected vehicles
        zone_manager.update(cars)

        # Get occupancy summary
        occupancy = zone_manager.get_occupancy_summary()
        
        # DEBUG: Print status every 30 frames
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_num % 30 == 0 and len(cars) > 0:
            print(f"\n--- Frame {frame_num} ---")
            for zone_id, info in occupancy.items():
                status_symbol = "OCCUPIED" if info['status'] == "OCCUPIED" else "VACANT  "
                vehicle_list = info['vehicle_ids'] if info['vehicle_ids'] else "none"
                print(f"{zone_id}: {status_symbol} | Vehicles: {vehicle_list}")
            print(f"Total: {zone_manager.get_total_occupied()}/{len(zone_manager.get_all_zones())} occupied")

        # -------------------------------------------------
        # 3. VISUALIZATION PIPELINE
        # -------------------------------------------------
        # Draw all zones
        frame = viz.draw_zones(frame, zone_manager.get_all_zones())

        # Draw the cars
        frame = viz.draw_cars(frame, cars)

        # Draw dashboard with multi-zone info
        total_occupied = zone_manager.get_total_occupied()
        total_zones = len(zone_manager.get_all_zones())
        frame = viz.draw_dashboard(frame, total_occupied, total_zones)

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
