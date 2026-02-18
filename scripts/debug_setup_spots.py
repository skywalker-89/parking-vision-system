import cv2
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.parking_logic.spot_manager import SpotManager

def main():
    # Paths
    # Note: SpotManager looks for model relative to itself, so "spots.pt" is fine if it's in modules/parking_logic
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ref_image_path = os.path.join(project_root, "config", "reference.jpg")
    output_path = os.path.join(project_root, "debug_spots_output.jpg")

    print(f"Debug: Loading reference image from {ref_image_path}...")
    if not os.path.exists(ref_image_path):
        print(f"Error: Reference image not found at {ref_image_path}")
        return

    frame = cv2.imread(ref_image_path)
    if frame is None:
        print("Error: Failed to load image.")
        return

    print("Debug: Initializing SpotManager...")
    # Initialize with the specific model filename requested by user
    manager = SpotManager(model_filename="spots.pt")

    print("Debug: Running detection...")
    # detect_spots_from_frame will run YOLO, save to json, and update self.spots
    success = manager.detect_spots_from_frame(frame)

    if success:
        print(f"Debug: Detection successful. Found {len(manager.spots)} spots.")
        
        # Visualize
        for spot in manager.spots:
            x1, y1, x2, y2 = map(int, spot)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label
            cv2.putText(frame, "Spot", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(output_path, frame)
        print(f"Debug: Output saved to {output_path}")
    else:
        print("Debug: Detection failed or no spots found.")

if __name__ == "__main__":
    main()
