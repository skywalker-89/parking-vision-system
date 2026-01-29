import cv2
import os
import numpy as np
from ultralytics import YOLO


class SpotManager:
    def __init__(self, model_path="modules/parking_logic/spots.pt"):
        self.model_path = model_path
        self.spots = []  # Stores [x1, y1, x2, y2] for every spot
        self.model = None

        # Try to load the model, otherwise use Demo Mode
        if os.path.exists(model_path):
            print(f"🅿️ Loading Spot Detection Model: {model_path}")
            self.model = YOLO(model_path)
            self.demo_mode = False
        else:
            print(f"⚠️ Spot Model not found at {model_path}")
            print("   -> Switching to DEMO MODE (Generating fake spots)")
            self.demo_mode = True

    def detect_spots_initial(self, frame):
        """
        Runs ONCE at startup to find where the parking lines are.
        """
        if self.demo_mode:
            self.spots = self._generate_demo_grid(frame)
        else:
            # Run inference using Person 2's model
            # We look for class '0' (assuming 'spot' is the only class)
            results = self.model(frame, conf=0.2)
            self.spots = []
            for r in results:
                for box in r.boxes:
                    # Store as integer coordinates
                    self.spots.append(box.xyxy[0].cpu().numpy().astype(int))

            print(f"✅ Detected {len(self.spots)} parking spots via AI.")

        return self.spots

    def check_occupancy(self, cars):
        """
        Matches Cars to Spots.
        Returns a list of spot dictionaries:
        [{'bbox': [x1,y1,x2,y2], 'occupied': True, 'car_id': 5}, ...]
        """
        spot_statuses = []

        for spot in self.spots:
            sx1, sy1, sx2, sy2 = spot

            # Define the spot area
            spot_poly = self._get_poly(spot)

            is_occupied = False
            occupying_car_id = None

            for car in cars:
                # We check if the Car's Bottom-Center is inside the Spot
                # (This is more accurate than box-overlap for CCTV angles)
                cx, cy = car["center_point"]

                # Check 1: Is the car center inside the spot?
                if self._is_point_in_rect((cx, cy), spot):
                    # Check 2: Is the car actually PARKED? (Using your Phase 6 logic)
                    # If we don't check this, a driving car will turn the spot Red.
                    if car.get("is_parked", False):
                        is_occupied = True
                        occupying_car_id = car["id"]
                        break  # Found a car, stop checking this spot

            spot_statuses.append(
                {"bbox": spot, "occupied": is_occupied, "car_id": occupying_car_id}
            )

        return spot_statuses

    def _is_point_in_rect(self, point, rect):
        """Helper to check if (x,y) is inside [x1,y1,x2,y2]"""
        px, py = point
        rx1, ry1, rx2, ry2 = rect
        return rx1 < px < rx2 and ry1 < py < ry2

    def _get_poly(self, bbox):
        return [
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3]),
        ]

    def _generate_demo_grid(self, frame):
        """Creates a fake grid of spots for testing without the model."""
        h, w, _ = frame.shape
        spots = []
        # Generate 10 spots in a row
        start_x, start_y = 200, 400
        spot_w, spot_h = 100, 150

        for i in range(8):
            x1 = start_x + (i * (spot_w + 10))
            y1 = start_y
            x2 = x1 + spot_w
            y2 = y1 + spot_h
            spots.append([x1, y1, x2, y2])
        return spots
