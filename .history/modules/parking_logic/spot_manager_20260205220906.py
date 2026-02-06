import cv2
import os
import numpy as np
from ultralytics import YOLO


class SpotManager:
    def __init__(self, model_filename="spots.pt"):
        # 1. Setup Path
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
            print("   -> Switching to DEMO MODE")
            self.demo_mode = True

    def detect_spots_initial(self, frame):
        """
        Runs inference ONCE. Includes NMS cleanup.
        """
        if self.demo_mode:
            return self._generate_demo_grid(frame)

        print("🔍 Scanning frame for parking spots (AI)...")
        results = self.model(frame, conf=0.15, verbose=True)

        raw_spots = []
        confidences = []
        detected_classes = set()

        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                raw_spots.append(coords)
                confidences.append(conf)
                detected_classes.add(cls_id)

        # --- DEBUGGING BLOCK ---
        print(f"🧐 DEBUG: The Spot Model detected these Class IDs: {detected_classes}")
        if 0 in detected_classes and len(detected_classes) == 1:
            print(
                "⚠️ WARNING: If Class 0 is 'Car', you are using the Car Model (best.pt) as the Spot Model."
            )
            print("👉 You need the 'slotdet' model from your friend.")
        # -----------------------

        # NMS to remove duplicate spots
        keep_indices = self._nms_xyxy(raw_spots, confidences, iou_th=0.4)
        self.spots = [raw_spots[i] for i in keep_indices]

        # Sort spots
        self.spots = sorted(self.spots, key=lambda b: (b[1], b[0]))

        print(f"✅ AI Detected {len(self.spots)} Parking Spots (Cleaned).")
        return self.spots

    def check_occupancy(self, cars):
        """
        Matches Cars to Spots using Weighted Scoring (IoU + Center Point).
        Prevents overlaps and missed detections.
        """
        # 1. Initialize all spots as Empty
        spot_statuses = []
        for i, spot in enumerate(self.spots, start=1):
            spot_statuses.append(
                {
                    "id": i,
                    "bbox": spot,
                    "occupied": False,
                    "car_id": None,
                    "match_score": 0.0,  # Track the quality of the match
                }
            )

        # 2. Iterate over PARKED cars only
        for car in cars:
            # Only consider cars that have been stable/parked
            if not car.get("is_parked", False):
                continue

            car_box = car["bbox"]
            cx, cy = car["center_point"]

            # Find which spot this car fits BEST
            best_spot_index = -1
            best_score = 0.0

            for idx, spot_status in enumerate(spot_statuses):
                spot_box = spot_status["bbox"]
                sx1, sy1, sx2, sy2 = spot_box

                # --- SCORING LOGIC ---
                # 1. Base Score: Intersection over Union (IoU)
                score = self._iou_xyxy(car_box, spot_box)

                # 2. Boost: Is the Center Point inside the spot?
                # This fixes the "False Free Space" issue where overlap is small but car is clearly there.
                center_inside = sx1 < cx < sx2 and sy1 < cy < sy2
                if center_inside:
                    score += 0.5  # Massive boost for spatial alignment

                # Keep track of the best fitting spot for this specific car
                if score > best_score:
                    best_score = score
                    best_spot_index = idx

            # 3. Assign the car to the winning spot
            # Threshold: Must have at least some overlap (0.15) OR be center-aligned (score > 0.5)
            if best_spot_index != -1 and best_score > 0.15:

                # COLLISION CHECK: Is this spot already taken by a "better" match?
                current_spot_score = spot_statuses[best_spot_index]["match_score"]

                if best_score > current_spot_score:
                    # Capture the spot
                    spot_statuses[best_spot_index]["occupied"] = True
                    spot_statuses[best_spot_index]["car_id"] = car["id"]
                    spot_statuses[best_spot_index]["match_score"] = best_score

        # Cleanup internal score before returning
        for s in spot_statuses:
            del s["match_score"]

        return spot_statuses

    # --- HELPERS ---

    def _iou_xyxy(self, a, b):
        """Calculates Intersection over Union between two boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - inter + 1e-9
        return inter / union

    def _nms_xyxy(self, boxes, scores, iou_th=0.5):
        """Non-Maximum Suppression to remove overlapping spot detections."""
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes, dtype=float)
        scores = np.array(scores, dtype=float)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            rest = order[1:]

            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)

            order = rest[iou <= iou_th]
        return keep

    def _generate_demo_grid(self, frame):
        spots = []
        start_x, start_y = 200, 400
        spot_w, spot_h = 100, 150
        for i in range(5):
            x1 = start_x + (i * (spot_w + 10))
            spots.append([x1, start_y, x1 + spot_w, start_y + spot_h])
        return spots
