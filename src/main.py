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
VIDEO_PATH = os.path.join("videos", "4min_demo.mp4")
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # fallback
    
    print(f"info: Total Frames in Video: {total_frames}, FPS: {fps}")
    
    fps_start_time = time.time()
    frame_count = 0

    print("✅ System Ready. Using Static Spot Detection (Crop-based).")
    
    # SIMULATION STATE
    last_update_day = -1 # track if we already updated for "today"

    while True:
        success, frame = cap.read()
        if not success:
            # Loop video for continuous simulation
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        frame_count += 1
        
        # Calculate Simulation Time
        # Video is 4 mins total.
        # 0-1 min: 2 AM
        # 1-2 min: 9:30 AM
        # 2-3 min: 16:00 PM
        # 3-4 min: 18:00 PM
        
        current_time_sec = frame_count / fps
        cycle_time = current_time_sec % 240 # 240 seconds = 4 minutes loop
        
        sim_time_str = ""
        is_2_am = False
        
        if 0 <= cycle_time < 60:
            sim_time_str = "02:00 AM (Night)"
            is_2_am = True
        elif 60 <= cycle_time < 120:
            sim_time_str = "09:30 AM (Morning)"
        elif 120 <= cycle_time < 180:
            sim_time_str = "16:00 PM (Afternoon)"
        else:
            sim_time_str = "18:00 PM (Evening)"
            
        
        # Calculate FPS every 10 frames
        if frame_count % 10 == 0:
            elapsed = time.time() - fps_start_time
            current_fps = frame_count / elapsed 
            # We don't print here to avoid spamming, will put on frame

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
        # 4. LOGIC: 2 AM UPDATE
        # -------------------------------------------------
        # Condition: It is 2 AM segment AND occupied spots <= 2
        # We use 'int(current_time_sec / 240)' to track "days" loops
        current_day = int(current_time_sec / 240)
        
        if is_2_am and occupied_spots <= 2:
            if current_day > last_update_day:
                print(f"🌙 2 AM DETECTED & LOW TRAFFIC ({occupied_spots} cars). Updating Spots...")
                if spot_manager.detect_spots_from_frame(frame):
                    print("✅ Spots updated successfully.")
                    last_update_day = current_day
                    # Re-calculate occupancy regarding new spots
                    spot_statuses = spot_manager.update_occupancy(cars)
        

        # -------------------------------------------------
        # 5. VISUALIZATION
        # -------------------------------------------------
        frame = viz.draw_spots(frame, spot_statuses)  # Draw the grid (red/green based on occupancy)
        frame = viz.draw_dashboard(frame, total_spots, occupied_spots)
        
        # Overlay Simulation Time
        cv2.putText(frame, f"SIM TIME: {sim_time_str}", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
