import cv2
import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from modules.parking_logic.perspective import PerspectiveManager
from modules.parking_logic.spot_manager import SpotManager

# Image Path
IMAGE_PATH = os.path.join(PROJECT_ROOT, "config", "reference.jpg")

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image not found at {IMAGE_PATH}")
        return

    # Initialize Managers
    print("🚀 Initializing Managers...")
    perspective = PerspectiveManager()
    spot_manager = SpotManager()

    # Load Image
    print(f"🖼️ Loading Reference Image: {IMAGE_PATH}")
    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        print("❌ Error reading image.")
        return

    # Detect Spots (or load from JSON)
    print("🔍 Loading Spots...")
    spots = spot_manager.detect_spots_initial(frame)
    if not spots:
        print("❌ No spots found. Check spot_manager configuration.")
        return

    print("🎥 Warping Frame to Bird's Eye View...")
    
    # 1. Warp Source Image
    # We use the same matrix but output size 400x800 as defined in PerspectiveManager
    bev_w, bev_h = perspective.bev_w, perspective.bev_h
    warped_frame = cv2.warpPerspective(frame, perspective.matrix, (bev_w, bev_h))

    # 2. Draw Spots on BEV Map
    # We iterate over all spots and project their centers onto the map
    print(f"🎨 Drawing {len(spots)} spots on the map...")
    
    for i, spot in enumerate(spots):
        x1, y1, x2, y2 = spot
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Transform Center to BEV
        # Note: PerspectiveManager.transform_point returns (x, y)
        bev_x, bev_y = perspective.transform_point(cx, cy)
        
        # Draw on Warped Image
        # Circle for center
        cv2.circle(warped_frame, (int(bev_x), int(bev_y)), 4, (0, 255, 0), -1)
        # ID text
        cv2.putText(warped_frame, str(i+1), (int(bev_x)+5, int(bev_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 3. Show Result
    print("\n✅ Visualization Ready!")
    print("👉 If the spots form a neat(ish) grid, calibration is GOOD.")
    print("👉 If spots are scattered, curved, or bunched up, calibration is BAD.")
    print("   (Bad calibration means you need to re-run tools/point_selector.py and pick better corners)")
    
    cv2.imshow("Bird's Eye View Debugger (Press any key to exit)", warped_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
