from ultralytics import YOLO

# Load model once
model = YOLO("yolo11n.pt")


def detect(frame):
    """
    Runs tracking on the frame.
    Returns list of cars with stable IDs.
    """
    # -----------------------------------------------------
    # KEY CHANGE: .track() instead of model()
    # persist=True -> "Remember this car from the last frame"
    # tracker="bytetrack.yaml" -> The algorithm we use
    # -----------------------------------------------------
    results = model.track(frame, persist=True, conf=0.3, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            # ⚠️ CRITICAL: If YOLO can't track an object yet, box.id is None
            if box.id is None:
                continue

            # Filter for vehicles only (Car=2, Motorcycle=3, Bus=5, Truck=7)
            class_id = int(box.cls[0])
            if class_id in [2, 3, 5, 7]:
                detections.append(
                    {
                        "bbox": box.xyxy[0].tolist(),
                        "conf": float(box.conf[0]),
                        "cls": class_id,
                        # NEW: Get the unique ID (e.g., Car #42)
                        "id": int(box.id[0]),
                    }
                )

    return detections
