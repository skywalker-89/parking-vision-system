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
            self.demo_mode = True

    def detect_spots_initial(self, frame):
        """
        Runs inference ONCE. Applies manual offsets to fix alignment issues.
        """
        if self.demo_mode:
            return self._generate_demo_grid(frame)

        # =========================================================
        # 🔧 ALIGNMENT CONFIGURATION (Paste numbers here!)
        # =========================================================
        OFFSET_X = 0  # <--- REPLACE WITH YOUR NUMBER FROM TOOL
        OFFSET_Y = 0  # <--- REPLACE WITH YOUR NUMBER FROM TOOL
        # =========================================================

        # 1. Find Reference Image
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        reference_path = os.path.join(project_root, "config", "reference.jpg")

        detection_frame = frame

        if os.path.exists(reference_path):
            print(f"🖼️ Using Reference Image: {reference_path}")
            ref_img = cv2.imread(reference_path)
            if ref_img is not None:
                if ref_img.shape[:2] != frame.shape[:2]:
                    detection_frame = cv2.resize(
                        ref_img, (frame.shape[1], frame.shape[0])
                    )
                else:
                    detection_frame = ref_img
        else:
            print("⚠️ No reference.jpg found. Using video frame.")

        # 2. Run AI Detection
        print("🔍 Scanning frame for spots...")
        results = self.model(detection_frame, conf=0.15, verbose=True)

        raw_spots = []
        confidences = []

        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])

                # -----------------------------------------------
                # 3. APPLY SHIFT (The Fix)
                # We add the offset to the AI's detection coordinates
                # -----------------------------------------------
                x1, y1, x2, y2 = coords
                shifted_coords = [
                    x1 + OFFSET_X,
                    y1 + OFFSET_Y,
                    x2 + OFFSET_X,
                    y2 + OFFSET_Y,
                ]

                raw_spots.append(shifted_coords)
                confidences.append(conf)

        # 4. NMS Cleanup
        keep_indices = self._nms_xyxy(raw_spots, confidences, iou_th=0.4)
        self.spots = [raw_spots[i] for i in keep_indices]

        # 5. Sort spots
        self.spots = sorted(self.spots, key=lambda b: (b[1], b[0]))

        print(
            f"✅ Loaded {len(self.spots)} Spots (with Offset X:{OFFSET_X}, Y:{OFFSET_Y})"
        )
        return self.spots

    def check_occupancy(self, cars):
        """Matches Cars to Spots using Weighted Scoring."""
        spot_statuses = []
        for i, spot in enumerate(self.spots, start=1):
            spot_statuses.append(
                {
                    "id": i,
                    "bbox": spot,
                    "occupied": False,
                    "car_id": None,
                    "match_score": 0.0,
                }
            )

        for car in cars:
            if not car.get("is_parked", False):
                continue

            car_box = car["bbox"]
            cx, cy = car["center_point"]

            best_idx = -1
            best_score = 0.0

            for idx, status in enumerate(spot_statuses):
                spot = status["bbox"]
                sx1, sy1, sx2, sy2 = spot

                score = self._iou_xyxy(car_box, spot)
                if sx1 < cx < sx2 and sy1 < cy < sy2:
                    score += 0.5

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx != -1 and best_score > 0.15:
                if best_score > spot_statuses[best_idx]["match_score"]:
                    spot_statuses[best_idx]["occupied"] = True
                    spot_statuses[best_idx]["car_id"] = car["id"]
                    spot_statuses[best_idx]["match_score"] = best_score

        for s in spot_statuses:
            del s["match_score"]
        return spot_statuses

    def _iou_xyxy(self, a, b):
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
        return inter / (area_a + area_b - inter + 1e-9)

    def _nms_xyxy(self, boxes, scores, iou_th=0.5):
        if not boxes:
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
        return []
