"""
Zone Diagnostic Tool - Shows anchor points vs zone boundaries
Helps identify why vehicles aren't being detected in certain zones
"""

import cv2
import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.detector import detector
from modules.logic import get_bottom_center
from shapely.geometry import Polygon, Point
import numpy as np

# Load config
with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load video
cap = cv2.VideoCapture('../videos/demo.mp4')

# Get a good frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
success, frame = cap.read()

if not success:
    print("Could not read video")
    sys.exit(1)

# Detect vehicles
cars = detector.detect(frame)

# Draw zones
for zone_config in config['zones']:
    zone_id = zone_config['id']
    coords = zone_config['polygon']
    polygon = Polygon(coords)
    
    pts = np.array(coords, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
    
    # Label zone
    moments = cv2.moments(pts)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        cv2.putText(frame, zone_id, (cx-40, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# Draw vehicles and anchor points
print("\nVehicle Detection Analysis:")
print("="*60)

for car in cars:
    vehicle_id = car['id']
    bbox = car['bbox']
    x1, y1, x2, y2 = map(int, bbox)
    
    # Calculate anchor point
    anchor = get_bottom_center(bbox)
    cx, cy = anchor
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"#{vehicle_id}", (x1, y1-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw anchor point (BIG and BRIGHT)
    cv2.circle(frame, anchor, 15, (0, 0, 255), -1)  # Red dot
    cv2.circle(frame, anchor, 17, (255, 255, 255), 2)  # White outline
    
    # Check which zones this vehicle is in
    print(f"\nCar #{vehicle_id}:")
    print(f"  Anchor point: ({cx}, {cy})")
    
    for zone_config in config['zones']:
        zone_id = zone_config['id']
        polygon = Polygon(zone_config['polygon'])
        point = Point(anchor)
        
        is_inside = polygon.contains(point)
        status = "INSIDE" if is_inside else "OUTSIDE"
        symbol = "✓" if is_inside else "✗"
        
        print(f"  {symbol} {zone_id}: {status}")

print("\n" + "="*60)
print("\nLEGEND:")
print("  Yellow polygons = Your defined zones")
print("  Blue boxes = Detected vehicles")
print("  RED DOTS = Anchor points (where wheels touch ground)")
print("\nIf a red dot is OUTSIDE a yellow zone, that vehicle won't")
print("be counted in that zone. You need to expand the polygon.\n")

# Display
cv2.imshow('Zone Diagnostic - Check Red Dots vs Yellow Zones', frame)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
