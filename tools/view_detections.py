"""
YOLO Detection Viewer - Shows what YOLO actually detects
Helps diagnose why some vehicles aren't being detected
"""

import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.detector import detector

# Load video
cap = cv2.VideoCapture('../videos/demo.mp4')

# Get frame 100
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
success, frame = cap.read()

if not success:
    print("Could not read video")
    sys.exit(1)

# Detect vehicles
cars = detector.detect(frame)

print(f"\nYOLO Detection Results (Frame 100):")
print("="*60)
print(f"Total vehicles detected and tracked: {len(cars)}\n")

# Draw all detections
for car in cars:
    x1, y1, x2, y2 = map(int, car['bbox'])
    vehicle_id = car['id']
    confidence = car['conf']
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Label
    label = f"Car #{vehicle_id} ({confidence:.2f})"
    cv2.putText(frame, label, (x1, y1-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Print details
    print(f"Car #{vehicle_id}:")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")
    print()

print("="*60)
print("\nGREEN BOXES = All vehicles YOLO detected and tracked")
print("If you see parked cars WITHOUT green boxes, YOLO missed them\n")

# Add text overlay
cv2.putText(frame, f"YOLO Detected: {len(cars)} vehicles", 
           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

# Display
cv2.imshow('YOLO Detection Viewer - What Does YOLO See?', frame)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
