import cv2
import numpy as np
import os

# Define Paths
# Adjust the video filename if needed
VIDEO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videos", "4min_demo2.mov")

points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"✅ Point Selected: ({x}, {y})")
            
            # Draw point
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            
            # Draw lines if we have more than 1 point
            if len(points) > 1:
                cv2.line(param, points[-2], points[-1], (0, 255, 0), 2)
            
            # If 4th point, close the loop
            if len(points) == 4:
                cv2.line(param, points[-1], points[0], (0, 255, 0), 2)
                print("\n🎉 4 Points Selected! Press ANY KEY to finish and generate code.")

            cv2.imshow("Calibration: Click 4 corners of a RECTANGLE (e.g. one lane)", param)

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Error reading video frame.")
        return

    print("\n" + "="*50)
    print("      PARKING SYSTEM CALIBRATOR v1.0")
    print("="*50)
    print("INSTRUCTIONS:")
    print("1. A window will open showing the first frame of your video.")
    print("2. Click exactly 4 POINTS that form a REAL-WORLD RECTANGLE.")
    print("   (e.g., The 4 corners of a single parking lane, or a rectangular marking on the road).")
    print("3. Order matters! Try to click: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left.")
    print("4. Press ANY KEY after selecting 4 points to see the code.")
    print("="*50 + "\n")

    cv2.imshow("Calibration: Click 4 corners of a RECTANGLE (e.g. one lane)", frame)
    cv2.setMouseCallback("Calibration: Click 4 corners of a RECTANGLE (e.g. one lane)", click_event, frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 4:
        print("\n\n✅ COPY THIS CODE into 'modules/parking_logic/perspective.py':\n")
        print("-" * 40)
        print("        self.src_points = np.float32([")
        print(f"            [{points[0][0]}, {points[0][1]}],    # Top-Left")
        print(f"            [{points[1][0]}, {points[1][1]}],    # Top-Right")
        print(f"            [{points[2][0]}, {points[2][1]}],    # Bottom-Right")
        print(f"            [{points[3][0]}, {points[3][1]}]     # Bottom-Left")
        print("        ])")
        print("-" * 40)
        print("\nNOTE: Ensure the order in the code matches (TL, TR, BR, BL).")
        print("If you clicked in a different order, adjust the list manually.")

if __name__ == "__main__":
    main()
