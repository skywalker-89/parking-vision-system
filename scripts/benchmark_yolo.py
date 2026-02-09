import time
import argparse
from ultralytics import YOLO
import torch
import numpy as np

def benchmark(model_path, source, device='cpu', imgsz=640):
    print(f"Loading model: {model_path} on {device}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Warmup
    print("Warming up...")
    model.predict(source, save=False, imgsz=imgsz, device=device, verbose=False, max_det=1)
    
    print(f"Benchmarking on {source}...")
    start_time = time.time()
    results = model.predict(source, save=False, imgsz=imgsz, device=device, verbose=False)
    end_time = time.time()

    total_time = end_time - start_time
    num_frames = len(results)
    fps = num_frames / total_time
    latency = (total_time / num_frames) * 1000

    print("\n" + "="*40)
    print(f"BENCHMARK RESULTS ({device})")
    print(f"Model: {model_path}")
    print(f"Frames: {num_frames}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"FPS: {fps:.2f}")
    print(f"Latency: {latency:.2f} ms/frame")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model file (best.pt, best.onnx, etc.)")
    parser.add_argument("--source", type=str, required=True, help="Path to video or image file")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, mps, cuda)")
    args = parser.parse_args()

    benchmark(args.model, args.source, args.device)
