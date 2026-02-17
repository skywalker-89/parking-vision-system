import cv2
import os
import numpy as np
import json
from ultralytics import YOLO


from .perspective import PerspectiveManager

class SpotManager:
    def __init__(self, model_filename="spots.pt"):
        # 1. Setup Paths
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(os.path.dirname(self.current_dir))

        # Search for model
        self.model_path = self._find_file(
            model_filename,
            [self.current_dir, os.path.join(self.current_dir, ".."), self.project_root],
        )

        # We will save the coordinates here so we don't need to detect every time
        self.json_path = os.path.join(self.project_root, "config", "spots_data.json")

        self.spots = []
        self.model = None

        # Initialize Perspective Transform
        self.perspective = PerspectiveManager()

        if self.model_path:
            print(f"✅ Spot Model found: {self.model_path}")
            self.model = YOLO(self.model_path)
        else:
            print(f"⚠️ Spot Model '{model_filename}' NOT found.")

    def _find_file(self, filename, search_paths):
        for path in search_paths:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                return full_path
        return None

    def detect_spots_from_frame(self, frame):
        """
        Force re-detection of spots from the provided frame.
        Saves to JSON and updates self.spots.
        """
        print("🔄 Re-detecting spots from current video frame...")
        if self.model:
            detected_spots = self._run_yolo_detection(frame)
            if len(detected_spots) > 0:
                self._save_spots_to_json(detected_spots, frame.shape)
                self.spots = detected_spots  # Already in frame coordinates
                return True
            else:
                print("❌ re-detection failed: No spots found.")
        return False

    def detect_spots_initial(self, video_frame):
        """
        The Master Logic:
        1. Check if 'spots_data.json' exists. If YES -> Load spots, scale to video, DONE.
        2. If NO JSON -> Look for 'reference.jpg'.
        3. If Reference found -> Detect spots, SAVE to JSON, scale to video, DONE.
        4. If No Reference -> Fallback to Manual Grid.
        """
        
        # -------------------------------------------------------------
        # STEP 1: Try Loading from JSON (The "Saved" Mode)
        # -------------------------------------------------------------
        if os.path.exists(self.json_path):
            print(f"📂 Found saved spots file: {self.json_path}")
            try:
                with open(self.json_path, "r") as f:
                    data = json.load(f)

                ref_w = data.get("width", 1920)
                ref_h = data.get("height", 1080)
                saved_spots = data.get("spots", [])

                if len(saved_spots) > 0:
                    print(f"✅ Loaded {len(saved_spots)} spots from JSON.")
                    # Scale spots if the video resolution is different from the saved image
                    self.spots = self._scale_spots(
                        saved_spots, (ref_w, ref_h), video_frame.shape
                    )
                    return self.spots
                else:
                    print("⚠️ JSON file was empty. Re-detecting...")
            except Exception as e:
                print(f"❌ Error reading JSON: {e}")

        # -------------------------------------------------------------
        # STEP 2: Try Detecting from Reference Image (Create the Data)
        # -------------------------------------------------------------
        # We look for reference_debug.jpg first as per your upload
        ref_image_names = [
            "reference_debug.jpg",
            "reference.jpg",
            "config/reference.jpg",
        ]
        ref_path = None

        # Search for reference image
        for name in ref_image_names:
            p = os.path.join(self.project_root, name)
            if not os.path.exists(p):
                p = os.path.join(self.project_root, "config", name)

            if os.path.exists(p):
                ref_path = p
                break

        if ref_path and self.model:
            print(f"🖼️ Detecting spots on Reference Image: {ref_path}")
            ref_img = cv2.imread(ref_path)

            if ref_img is not None:
                # RUN DETECTION ON REFERENCE (Not the video frame!)
                detected_spots = self._run_yolo_detection(ref_img)

                if len(detected_spots) > 0:
                    # SAVE TO JSON IMMEDIATELY
                    self._save_spots_to_json(detected_spots, ref_img.shape)

                    # Scale to video frame and set
                    self.spots = self._scale_spots(
                        detected_spots, ref_img.shape, video_frame.shape
                    )
                    return self.spots
                else:
                    print(
                        "❌ Model ran on reference image but found 0 spots. Check model/image."
                    )
            
            # If reference image detection failed, fall through to fallback
        
        # -------------------------------------------------------------
        # STEP 3: Fallback (Manual Grid)
        # -------------------------------------------------------------
        print(
            "⚠️ No JSON, No Reference Image (or 0 detections). Using MANUAL DEMO GRID."
        )
        self.spots = self._generate_demo_grid(video_frame)
        return self.spots

    def _run_yolo_detection(self, image):
        """Runs the YOLO model on a specific image."""
        print("🔍 Running YOLO on reference image...")
        results = self.model(
            image, conf=0.12, verbose=False
        )  # Low confidence to catch everything
        raw_spots = []
        confidences = []

        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0])
                raw_spots.append(coords)
                confidences.append(conf)

        # Apply Non-Maximum Suppression (cleanup overlaps)
        keep_indices = self._nms_xyxy(raw_spots, confidences, iou_th=0.4)
        final_spots = [raw_spots[i] for i in keep_indices]

        # Sort top-to-bottom, left-to-right
        final_spots = sorted(final_spots, key=lambda b: (b[1], b[0]))
        return final_spots

    def _save_spots_to_json(self, spots, shape):
        h, w = shape[:2]
        data = {"width": w, "height": h, "spots": spots}
        try:
            # Ensure config dir exists
            config_dir = os.path.dirname(self.json_path)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)

            with open(self.json_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"💾 SAVED {len(spots)} spots to {self.json_path}")
            print("👉 You can manually edit this file to fix spot positions!")
        except Exception as e:
            print(f"❌ Failed to save JSON: {e}")

    def _scale_spots(self, spots, src_shape, dest_shape):
        """Scales coordinates if video resolution != reference resolution."""
        src_w, src_h = (
            src_shape[:2] if len(src_shape) == 2 else (src_shape[1], src_shape[0])
        )
        dest_h, dest_w = dest_shape[:2]

        if src_w == dest_w and src_h == dest_h:
            return spots

        print(f"📐 Scaling spots from {src_w}x{src_h} to {dest_w}x{dest_h}")
        scale_x = dest_w / src_w
        scale_y = dest_h / src_h

        scaled_spots = []
        for x1, y1, x2, y2 in spots:
            scaled_spots.append(
                [
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y),
                ]
            )
        return scaled_spots

    def update_occupancy(self, cars):
        """
        Updates occupancy using a Global Best-Match strategy (Hungarian-style greedy).
        1. Calculate distances for ALL (car, spot) pairs.
        2. Sort by distance (closest first).
        3. Assign matches if distance < Threshold.
        """
        # 1. Initialize all spots as empty
        spot_statuses = [
            {"id": i, "bbox": spot, "occupied": False, "car_id": None}
            for i, spot in enumerate(self.spots, start=1)
        ]

        # 2. Collect all candidate matches
        # Format: (distance, car_index, spot_index)
        candidates = []
        
        # Pre-calculate BEV positions to avoid re-calc in loops
        cars_bev = [self.perspective.get_car_footprint(c["bbox"]) for c in cars]
        spots_bev = [self.perspective.get_spot_center_bev(s["bbox"]) for s in spot_statuses]

        for c_idx, (cx, cy) in enumerate(cars_bev):
            for s_idx, (sx, sy) in enumerate(spots_bev):
                # Calculate Euclidean Distance
                dist = ((cx - sx)**2 + (cy - sy)**2)**0.5
                
                # Loose threshold to filter out obvious non-matches
                if dist < 70: 
                    candidates.append((dist, c_idx, s_idx))

        # 3. Sort by distance (Closest matches first)
        candidates.sort(key=lambda x: x[0])

        # 4. Assign matches
        occupied_cars = set()
        occupied_spots = set()

        for dist, c_idx, s_idx in candidates:
            # If car or spot is already taken, skip
            if c_idx in occupied_cars or s_idx in occupied_spots:
                continue

            # Assign
            spot_statuses[s_idx]["occupied"] = True
            spot_statuses[s_idx]["car_id"] = cars[c_idx]["id"]
            
            occupied_cars.add(c_idx)
            occupied_spots.add(s_idx)
        
        return spot_statuses

    def _iou_xyxy(self, a, b):
        # Basic IoU implementation
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
        # Basic NMS
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
        # Fallback grid
        spots = []
        h, w = frame.shape[:2]

        # Tuned for standard parking demo videos
        start_x_left = int(w * 0.28)
        width_left = int(w * 0.13)
        start_x_right = int(w * 0.44)
        width_right = int(w * 0.13)
        start_y = int(h * 0.15)
        spot_height = int(h * 0.055)
        gap = 4

        for i in range(12):
            y1 = start_y + (i * (spot_height + gap))
            y2 = y1 + spot_height
            spots.append([start_x_left, y1, start_x_left + width_left, y2])
            spots.append([start_x_right, y1, start_x_right + width_right, y2])

        print(f"⚠️ Generated {len(spots)} DEMO spots (Reference failed).")
        return spots