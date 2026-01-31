import cv2
import os
import numpy as np
from ultralytics import YOLO


class SpotManager:
    def __init__(self, model_path="modules/parking_logic/spots.pt"):
        self.model_path = model_path
        self.spots = []  # Stores [x1, y1, x2, y2] for every spot
        self.spot_masks = []  # Stores polygon points for accurate occupancy checking
        self.spot_classes = []  # Stores class ID: 0=blocked_space, 1=parking_space
        self.model = None

        # load model
        if os.path.exists(model_path):
            print(f"Loading Spot Detection Model: {model_path}")
            self.model = YOLO(model_path)
            self.demo_mode = False
        else:
            print(f"⚠️ Spot Model not found at {model_path}")
            print("   -> Switching to DEMO MODE (Generating fake spots)")
            self.demo_mode = True

    def detect_spots_initial(self, frame):
        """
        Runs ONCE at startup to find where the parking lines are.
        Now handles SEGMENTATION MASKS (polygons) from trained model.
        """
        if self.demo_mode:
            self.spots = self._generate_demo_grid(frame)
        else:
            # Run inference using trained segmentation model
            # conf=0.2 means 20% confidence threshold
            # Lower = more spots detected (might
            #  include false positives)
            # Higher = fewer spots detected (might miss some)
            results = self.model(frame, conf=0.6, iou=0.5)
            
            self.spots = []
            self.spot_masks = []  # Store polygon masks for more accurate checking
            self.spot_classes = []  # Store class IDs
            
            blocked_count = 0
            parking_count = 0
            
            for r in results:
                # Check if we have segmentation masks
                if hasattr(r, 'masks') and r.masks is not None:
                    # SEGMENTATION MODE (polygons) - More accurate!
                    for i, mask in enumerate(r.masks):
                        # Get the bounding box for quick checks
                        bbox = r.boxes[i].xyxy[0].cpu().numpy().astype(int)
                        self.spots.append(bbox)
                        
                        # Store the CLASS ID (0=blocked_space, 1=parking_space)
                        class_id = int(r.boxes[i].cls[0].cpu().numpy())
                        self.spot_classes.append(class_id)
                        
                        if class_id == 0:
                            blocked_count += 1
                        else:
                            parking_count += 1
                        
                        # Store the actual polygon mask for accurate occupancy checking
                        # masks.xy gives us the polygon points
                        if hasattr(mask, 'xy') and len(mask.xy) > 0:
                            polygon = mask.xy[0].astype(int)  # Get first polygon
                            self.spot_masks.append(polygon)
                        else:
                            # Fallback: create rectangle from bbox
                            self.spot_masks.append(self._bbox_to_polygon(bbox))
                    
                    print(f"✅ Detected {len(self.spots)} parking spots via AI (SEGMENTATION)")
                    print(f"   - {parking_count} available parking spaces")
                    print(f"   - {blocked_count} blocked/unavailable spaces")
                    
                else:
                    # FALLBACK: Bounding box mode (if model isn't segmentation)
                    for i, box in enumerate(r.boxes):
                        bbox = box.xyxy[0].cpu().numpy().astype(int)
                        self.spots.append(bbox)
                        self.spot_masks.append(self._bbox_to_polygon(bbox))
                        
                        # Store class ID
                        class_id = int(box.cls[0].cpu().numpy())
                        self.spot_classes.append(class_id)
                        
                        if class_id == 0:
                            blocked_count += 1
                        else:
                            parking_count += 1
                    
                    print(f"✅ Detected {len(self.spots)} parking spots via AI (BBOX mode)")
                    print(f"   - {parking_count} available parking spaces")
                    print(f"   - {blocked_count} blocked/unavailable spaces")

        return self.spots

    def check_occupancy(self, cars):
        """
        Matches Cars to Spots using POLYGON-based checking (more accurate!).
        Returns a list of spot dictionaries:
        [{'bbox': [x1,y1,x2,y2], 'occupied': True/False, 'car_id': 5, 
          'is_blocked': True/False, 'class_id': 0/1}, ...]
        
        Class IDs:
          0 = blocked_space (permanently unavailable - reserved, disabled, etc.)
          1 = parking_space (normal spot that can be occupied by cars)
        """
        spot_statuses = []

        for i, spot in enumerate(self.spots):
            sx1, sy1, sx2, sy2 = spot

            # Get the polygon mask for this spot (if available)
            spot_poly = self.spot_masks[i] if i < len(self.spot_masks) else self._bbox_to_polygon(spot)
            
            # Get the class ID (0=blocked, 1=parking)
            class_id = self.spot_classes[i] if i < len(self.spot_classes) else 1  # Default to parking
            is_blocked_space = (class_id == 0)

            is_occupied = False
            occupying_car_id = None

            # Only check for car occupancy if it's a PARKING space (not blocked)
            if not is_blocked_space:
                for car in cars:
                    # We check if the Car's Bottom-Center is inside the Spot POLYGON
                    cx, cy = car["center_point"]

                    # Check 1: Is the car center inside the spot polygon?
                    if self._is_point_in_polygon((cx, cy), spot_poly):
                        # Check 2: Is the car actually PARKED?
                        if car.get("is_parked", False):
                            is_occupied = True
                            occupying_car_id = car["id"]
                            break  # Found a car, stop checking this spot

            spot_statuses.append({
                "bbox": spot,
                "occupied": is_occupied,
                "car_id": occupying_car_id,
                "is_blocked": is_blocked_space,  # True if permanently blocked
                "class_id": class_id  # 0=blocked_space, 1=parking_space
            })

        return spot_statuses

    def _is_point_in_polygon(self, point, polygon):
        """
        Check if a point (x, y) is inside a polygon using cv2.pointPolygonTest.
        More accurate than simple rectangle check!
        """
        if len(polygon) < 3:
            # Not a valid polygon, fall back to rectangle
            return False
        
        # cv2.pointPolygonTest returns:
        # > 0: point is inside
        # = 0: point is on the edge
        # < 0: point is outside
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0
    
    def _bbox_to_polygon(self, bbox):
        """Convert bounding box [x1,y1,x2,y2] to polygon points"""
        x1, y1, x2, y2 = bbox
        return np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32)

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
            # Also create mask polygons for demo spots
            self.spot_masks.append(self._bbox_to_polygon([x1, y1, x2, y2]))
            # Demo spots are all parking spaces (class 1)
            self.spot_classes.append(1)
        
        return spots
