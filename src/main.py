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

    # PERFORMANCE METRICS
    import time
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"info: Total Frames in Video: {total_frames}")
    
    fps_start_time = time.time()
    frame_count = 0

    print("✅ System Ready. Using Static Spot Detection (Crop-based).")

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # Calculate FPS every 10 frames
        if frame_count % 10 == 0:
            elapsed = time.time() - fps_start_time
            current_fps = frame_count / elapsed
            print(f"\r🚀 Speed: {current_fps:.2f} FPS (Frame {frame_count}/{total_frames})", end="")


        # -------------------------------------------------
        # 3. DETECTION & STATIC CHECK
        # -------------------------------------------------
        # Step 1: Detect all cars in the frame (Global Context = Better Accuracy)
        cars = detector.detect(frame)

        # Step 2: Check which static spots contain these cars
        spot_statuses = spot_manager.update_occupancy(cars)

        # Calculate Counts
        parking_spaces = [s for s in spot_statuses if not s.get("is_blocked", False)]
        occupied_spots = sum(1 for s in parking_spaces if s["occupied"])
        total_spots = len(parking_spaces)

        # -------------------------------------------------
        # 4. VISUALIZATION
        # -------------------------------------------------
        frame = viz.draw_spots(frame, spot_statuses)  # Draw the grid (red/green based on occupancy)
        frame = viz.draw_dashboard(frame, total_spots, occupied_spots)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
