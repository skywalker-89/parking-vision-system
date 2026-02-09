import cv2
import argparse
from ultralytics import YOLO
import numpy as np

def compare_models(model1_path, model2_path, source, output_path="comparison.mp4"):
    print(f"Loading models: {model1_path}, {model2_path}...")
    model1 = YOLO(model1_path)
    model2 = YOLO(model2_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error opening video source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output writer (Side-by-side)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

    print(f"Processing video... Output: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        res1 = model1(frame, verbose=False)[0]
        res2 = model2(frame, verbose=False)[0]

        # Plot results
        im1 = res1.plot()
        im2 = res2.plot()

        # Add labels
        cv2.putText(im1, "Baseline (Original)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(im2, "Modified (Custom)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Concatenate horizontally
        combined = np.hstack((im1, im2))
        out.write(combined)

    cap.release()
    out.release()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline model (e.g., original_best.pt)")
    parser.add_argument("--modified", type=str, required=True, help="Path to modified model")
    parser.add_argument("--source", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, default="comparison.mp4", help="Output video path")
    args = parser.parse_args()

    compare_models(args.baseline, args.modified, args.source, args.output)
