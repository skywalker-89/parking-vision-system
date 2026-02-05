import cv2
import os
import numpy as np
from ultralytics import YOLO


class SpotManager:
    def __init__(self, model_filename="spots.pt"):
        # 1. Setup Path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, model_filename)

        self.spots = []
        self.model = None

        # 2. Load Model
        if os.path.exists(self.model_path):
            print(f"🅿️ Loading Spot Model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.demo_mode = False
        else:
            print(f"⚠️ Spot Model not found at {self.model_path}")
            print("   -> Switching to DEMO MODE (Generating fake grid)")
            self.demo_mode = True

    def detect_spots_initial(self, frame):
        """
        Runs inference ONCE on the first frame to find all parking spots.
        This defines the grid for the rest of the video.
        """
        if self.demo_mode:
            return self._generate_demo_grid(frame)

        print("🔍 Scanning frame for parking spots (AI)...")
        # Run inference on the first frame
        # conf=0.2 is usually good for static lines/spots detection
        results = self.model(frame, conf=0.2, verbose=True)

        self.spots = []
        for r in results:
            for box in r.boxes:
                # Convert bbox to integer coordinates [x1, y1, x2, y2]
                coords = box.xyxy[0].cpu().numpy().astype(int)
                self.spots.append(coords)

        print(f"✅ AI Detected {len(self.spots)} Parking Spots.")
        return self.spots

    def check_occupancy(self, cars):
        """
        Matches 'Parked' Cars to 'Spots'.
        Returns a list of spots with their status (Occupied/Free).
        """
        spot_statuses = []

        for spot in self.spots:
            sx1, sy1, sx2, sy2 = spot
            spot_area = (sx2 - sx1) * (sy2 - sy1)

            is_occupied = False
            occupying_car_id = None

            for car in cars:
                # 1. Get Geometry
                # Car Bbox
                cx1, cy1, cx2, cy2 = map(int, car["bbox"])
                # Car Point (Bottom Center)
                px, py = car["center_point"]

                # 2. CALCULATE OVERLAP (Robust Intersection)
                # Find intersection box
                ix1 = max(sx1, cx1)
                iy1 = max(sy1, cy1)
                ix2 = min(sx2, cx2)
                iy2 = min(sy2, cy2)

                overlap_score = 0
                if ix2 > ix1 and iy2 > iy1:
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    # Percentage of the SPOT covered by the car
                    overlap_score = intersection_area / spot_area

                # 3. Check "Point Inside" (Perspective)
                point_inside = sx1 < px < sx2 and sy1 < py < sy2

                # 4. COMBINED CHECK
                # Match if (Point is inside) OR (Overlap is significant > 15%)
                if point_inside or overlap_score > 0.15:

                    # 5. Check Status (Must be "Parked")
                    if car.get("is_parked", False):
                        is_occupied = True
                        occupying_car_id = car["id"]
                        break  # Spot is taken, stop checking other cars for this spot

            spot_statuses.append(
                {"bbox": spot, "occupied": is_occupied, "car_id": occupying_car_id}
            )

        return spot_statuses

    def _generate_demo_grid(self, frame):
        """
        Fallback function: Creates a fake row of spots if the model is missing.
        Useful for testing code logic without the .pt file.
        """
        spots = []
        # Create a simple row of 5 spots in the middle of the screen
        start_x, start_y = 200, 400
        spot_w, spot_h = 100, 150

        for i in range(5):
            x1 = start_x + (i * (spot_w + 10))
            x2 = x1 + spot_w
            y2 = start_y + spot_h
            spots.append([x1, start_y, x2, y2])

        return spots
