from ultralytics import YOLO

# Load the model once when the module is imported
# "yolo11n.pt" will auto-download the first time you run this.
# We use the 'nano' (n) version because it is the fastest for testing.
model = YOLO("yolo11n.pt")


def detect(frame):
    """
    Takes a frame, runs YOLO, and returns a clean list of detections.
    """
    # conf=0.3 means we only keep detections with >30% confidence
    results = model(frame, conf=0.3)

    detections = []

    for r in results:
        for box in r.boxes:
            # Extract standard info: bounding box, confidence, class ID
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            # Filter: We only care about cars (2), motorcycles (3), buses (5), trucks (7)
            # COCO dataset class IDs: https://docs.ultralytics.com/datasets/detect/coco/
            if class_id in [2, 3, 5, 7]:
                detections.append(
                    {"bbox": [x1, y1, x2, y2], "conf": confidence, "cls": class_id}
                )

    return detections
