# YOLOv8 Model Refinement & Optimization Guide

This guide describes how to **modify usage of the YOLOv8 architecture**, **retrain** it on your dataset, and **benchmark** the results to prove improvements in Speed (FPS) or Performance (mAP).

---

## 🛠️ Step 1: Setup & Baseline
First, verify your current performance so we have numbers to beat.

1.  **Locate your files**:
    - `data.yaml`: The file pointing to your training/validation images.
    - `best.pt`: The original trained model weights.
    - `test_video.mp4`: A video for visual testing.

2.  **Run Baseline Validation**:
    Open your terminal/command prompt and run:
    ```bash
    yolo val model=path/to/best.pt data=path/to/data.yaml split=test name=baseline_eval
    ```
    > **Write down**: The mAP50 and mAP50-95 scores.

---

## 🏗️ Step 2: Modify The Architecture
To "modify layers and modules", we change the YAML configuration file that defines the neural network structure.

1.  **Create a Custom Config**:
    Find `yolov8n.yaml` (or whatever size `s`, `m`, `l` you used). Copy it to a new file named `custom_yolov8.yaml`.

2.  **Edit `custom_yolov8.yaml`**:
    Open it in a text editor (Notepad, VS Code).
    
    **Option A: Make it Faster (Lighter Backbone)**
    Change standard Convolutions to Ghost Convolutions (efficient, fewer parameters).
    *Find lines like:* `[-1, 1, Conv, [64, 3, 2]]`
    *Change to:* `[-1, 1, GhostConv, [64, 3, 2]]`
    
    **Option B: Pruning (Remove Layers)**
    In the `head` section, if you only have 1 class (Cars), you don't need complex multi-scale detection if all cars are roughly the same size. You could try removing one of the detection heads.

---

## 🚀 Step 3: Retrain (The Critical Step)
You cannot use the old weights (`best.pt`) with a new architecture. You must retrain from scratch.

Run this command:
```bash
yolo train model=custom_yolov8.yaml data=path/to/data.yaml epochs=50 imgsz=640 name=custom_model_run
```
*Note: We point `model` to the .yaml file, NOT the .pt file.*

Once finished, your new specific weights will be in:
`runs/detect/custom_model_run/weights/best.pt`

---

## 📊 Step 4: Benchmark (Proof of Improvement)
Use the python script below to get the exact FPS (Frames Per Second) and Latency numbers for your report.

**Create a file named `benchmark.py` and paste this code:**

```python
import time
import argparse
from ultralytics import YOLO

# USAGE: python benchmark.py --model path/to/best.pt --source video.mp4

def benchmark(model_path, source, device='cpu'):
    print(f"Loading model: {model_path} on {device}...")
    model = YOLO(model_path)

    # Warmup
    print("Warming up...")
    model.predict(source, save=False, max_det=1, verbose=False)
    
    print(f"Benchmarking...")
    start_time = time.time()
    results = model.predict(source, save=False, verbose=False)
    end_time = time.time()

    total_time = end_time - start_time
    num_frames = len(results)
    fps = num_frames / total_time
    latency = (total_time / num_frames) * 1000

    print("\n" + "="*40)
    print(f"BENCHMARK RESULTS")
    print(f"Model: {model_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Latency: {latency:.2f} ms")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--device", type=str, default="0") # '0' for GPU, 'cpu' for CPU
    args = parser.parse_args()
    benchmark(args.model, args.source, args.device)
```

**Run it for both models:**
```bash
python benchmark.py --model path/to/original_best.pt --source test_video.mp4
python benchmark.py --model runs/detect/custom_model_run/weights/best.pt --source test_video.mp4
```

---

## 🎥 Step 5: Visual Comparison
Create a side-by-side video to visually show the difference.

**Create a file named `visualize.py` and paste this code:**

```python
import cv2
import argparse
from ultralytics import YOLO
import numpy as np

# USAGE: python visualize.py --baseline old.pt --modified new.pt --source video.mp4

def compare(m1_path, m2_path, source):
    m1 = YOLO(m1_path)
    m2 = YOLO(m2_path)
    cap = cv2.VideoCapture(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('comparison.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Run Inference
        r1 = m1(frame, verbose=False)[0].plot()
        r2 = m2(frame, verbose=False)[0].plot()
        
        # Label
        cv2.putText(r1, "Original", (30,50), 1, 3, (0,0,255), 3)
        cv2.putText(r2, "Modified", (30,50), 1, 3, (0,255,0), 3)
        
        # Combine
        out.write(np.hstack((r1, r2)))

    cap.release()
    out.release()
    print("Saved to comparison.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--modified", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    compare(args.baseline, args.modified, args.source)
```

**Run it:**
```bash
python visualize.py --baseline path/to/original_best.pt --modified runs/detect/custom_model_run/weights/best.pt --source test_video.mp4
```

---

## 📋 Final Report Table (Example)

After doing all steps, fill in this table for your professor:

| Metric | Original Model | Custom Modified Model | Improvement |
| :--- | :--- | :--- | :--- |
| **Architecture** | YOLOv8n (Standard) | YOLOv8-Ghost (Custom) | Modified Layers |
| **mAP@50** | 0.94 | 0.93 | -1.0% |
| **FPS (GPU)** | 145 FPS | 185 FPS | +27% (Faster!) |
| **Model Size** | 6.2 MB | 4.1 MB | -33% (Smaller) |

> **Conclusion**: "We modified the YOLOv8 architecture by replacing standard convolutions with Ghost convolutions. This resulted in a model that is **27% faster** with only a negligible drop in accuracy, meeting the goal of optimization."
