"""
Simple tool to find pixel coordinates by clicking on the image.
Click on points and the coordinates will be printed to console.
Press 'q' to quit.
"""

import cv2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Global variables
points = []
zone_count = 0

def mouse_callback(event, x, y, flags, param):
    """Called when mouse is clicked"""
    global points, zone_count
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point {len(points)}: [{x}, {y}]")
        
        # Draw the point
        cv2.circle(param['display'], (x, y), 8, (0, 0, 255), -1)
        cv2.putText(param['display'], str(len(points)), (x+10, y+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw line to previous point
        if len(points) > 1:
            cv2.line(param['display'], tuple(points[-2]), tuple(points[-1]), 
                     (0, 255, 0), 2)
        
        # Close polygon if 4 points
        if len(points) == 4:
            cv2.line(param['display'], tuple(points[-1]), tuple(points[0]), 
                     (0, 255, 0), 2)
            zone_count += 1
            print(f"\nZone {zone_count} complete:")
            print("  - id: \"ZONE_{0}\"".format(zone_count))
            print("    polygon:")
            for pt in points:
                print(f"      - [{pt[0]}, {pt[1]}]")
            print("\nPress 'n' for next zone, or 'q' to quit\n")
        
        cv2.imshow('Click to Find Coordinates', param['display'])

def main():
    global points
    
    # Load the reference image
    img_path = 'parking_layout_reference.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return
    
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not load {img_path}")
        return
    
    display = image.copy()
    
    # Create window and set mouse callback
    cv2.namedWindow('Click to Find Coordinates')
    cv2.setMouseCallback('Click to Find Coordinates', mouse_callback, 
                         {'image': image, 'display': display})
    
    # Instructions
    print("\n" + "="*60)
    print("COORDINATE FINDER TOOL")
    print("="*60)
    print("\nInstructions:")
    print("1. Click 4 corners of a parking slot (clockwise)")
    print("2. After 4 clicks, zone coordinates will be printed")
    print("3. Press 'n' to start next zone (resets points)")
    print("4. Press 'r' to reset current zone")
    print("5. Press 'q' to quit")
    print("\nStart by clicking the TOP-LEFT corner of a parking slot...")
    print("="*60 + "\n")
    
    cv2.imshow('Click to Find Coordinates', display)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nExiting...")
            break
        elif key == ord('n'):
            # Start new zone
            points = []
            display = image.copy()
            cv2.imshow('Click to Find Coordinates', display)
            print("Starting new zone... Click 4 corners")
        elif key == ord('r'):
            # Reset current zone
            points = []
            display = image.copy()
            cv2.imshow('Click to Find Coordinates', display)
            print("Reset. Click 4 corners again")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
