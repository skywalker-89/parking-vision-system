import os
from ultralytics import YOLO

# 1. GET ABSOLUTE PATH to best.pt
# This ensures Python finds the file no matter where you run the command from
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "best.pt")

print(f"🚀 Loading Custom Model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)


def detect(frame):
    """
    Runs tracking on the frame using the custom model.
    """
    # Run tracker
    results = model.track(frame, persist=True, conf=0.15, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            if box.id is None:
                continue

            # Get the Class ID
            class_id = int(box.cls[0])

            # -----------------------------------------------------------
            # ⚠️ DEBUGGING STEP (Check this in your terminal!)
            # If you see boxes on screen, you can delete this print line.
            # If NO boxes appear, look at the terminal to see what ID your
            # friend's model is using for cars (it's likely 0 now).
            # print(f"DEBUG: Found Object with Class ID: {class_id}")
            # -----------------------------------------------------------

            # CUSTOM MODEL FILTER
            # If your friend trained on just cars, they are likely all Class 0.
            # We accept Class 0 (likely car) and keep 2, 3, 5, 7 just in case.
            if class_id in [0, 2, 3, 5, 7]:
                detections.append(
                    {
                        "bbox": box.xyxy[0].tolist(),
                        "conf": float(box.conf[0]),
                        "cls": class_id,
                        "id": int(box.id[0]),
                    }
                )

    return detections


def is_car_in_roi(frame, bbox):
    """
    Checks if a car is present in the specified Region of Interest (ROI).
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Context Padding: Add significant context. 
    # The spots might be very thin strips, so adding vertical padding helps aspect ratio.
    padding = 40 
    h, w = frame.shape[:2]
    
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    # Crop the spot image
    crop = frame[y1:y2, x1:x2]

    # If crop is too small or invalid, return False
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return False

    # Run Detection on the crop
    # Lowered confidence substantially
    try:
        results = model(crop, conf=0.1, verbose=False)
        
        for r in results:
            if len(r.boxes) > 0:
                # DEBUG: Print the confidence of the first hit just to see in console
                # print(f"DEBUG: Spot Hit! Conf: {r.boxes.conf[0]:.2f}")
                return True
                
        return False
    except Exception as e:
        print(f"⚠️ Detection Error on crop: {e}")
        return False
