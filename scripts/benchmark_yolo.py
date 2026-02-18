# benchmark_models.py
import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def fmt(x, d=2):
    return "N/A" if x is None else f"{x:.{d}f}"


def is_image(path: str):
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@torch.inference_mode()
def benchmark_one(model_path: str, source: str, device: str, imgsz: int, warmup: int = 20):
    print(f"\nLoading model: {model_path} on device={device}")
    model = YOLO(model_path)

    # ---- Warmup ----
    if is_image(source):
        _ = model.predict(source, save=False, imgsz=imgsz, device=device, verbose=False, max_det=1)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")
        w = 0
        while w < warmup:
            ok, frame = cap.read()
            if not ok:
                break
            _ = model.predict(frame, save=False, imgsz=imgsz, device=device, verbose=False, max_det=1)
            w += 1
        cap.release()

    # ---- Benchmark ----
    if is_image(source):
        t0 = time.time()
        _ = model.predict(source, save=False, imgsz=imgsz, device=device, verbose=False)
        t1 = time.time()
        total = t1 - t0
        frames = 1
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")

        frames = 0
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _ = model.predict(frame, save=False, imgsz=imgsz, device=device, verbose=False)
            frames += 1
        t1 = time.time()
        cap.release()
        total = t1 - t0

    fps = frames / max(total, 1e-9)
    latency_ms = (total / max(frames, 1)) * 1000.0
    return {
        "path": model_path,
        "frames": frames,
        "total_s": total,
        "fps": fps,
        "latency_ms": latency_ms,
    }


def parse_models(kv_list):
    """
    Accept repeated: --models name=path
    """
    models = []
    for item in kv_list:
        if "=" not in item:
            raise ValueError(f'Bad --models value "{item}". Use name=path')
        name, path = item.split("=", 1)
        models.append((name.strip(), path.strip().strip('"')))
    return models


def print_table(rows):
    headers = ["Model", "FPS", "Latency (ms)", "Frames", "Total (s)", "Path"]
    data = []
    for r in rows:
        data.append([
            r["name"],
            f'{r["fps"]:.2f}',
            f'{r["latency_ms"]:.2f}',
            str(r["frames"]),
            f'{r["total_s"]:.2f}',
            r["path"],
        ])

    widths = [len(h) for h in headers]
    for row in data:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))

    def join(row):
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))

    sep = "-+-".join("-" * w for w in widths)

    print("\n=== BENCHMARK COMPARISON ===")
    print(join(headers))
    print(sep)
    for row in data:
        print(join(row))


def save_csv(rows, out_path: str):
    out = Path(out_path)
    with out.open("w", encoding="utf-8") as f:
        f.write("Model,FPS,Latency_ms,Frames,Total_s,Path\n")
        for r in rows:
            f.write(f'{r["name"]},{r["fps"]:.6f},{r["latency_ms"]:.6f},{r["frames"]},{r["total_s"]:.6f},"{r["path"]}"\n')
    print(f"\nSaved CSV -> {out.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="append", required=True, help='Repeat: --models name="path/to/best.pt"')
    ap.add_argument("--source", required=True)
    ap.add_argument("--device", default="cpu")   # "0" for GPU, or "cpu"
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="benchmark_models.csv")
    args = ap.parse_args()

    models = parse_models(args.models)

    rows = []
    for name, path in models:
        print(f"\n=== {name} ===")
        res = benchmark_one(path, args.source, args.device, args.imgsz, warmup=args.warmup)
        res["name"] = name
        rows.append(res)

        print("\n" + "=" * 40)
        print(f"BENCHMARK RESULTS ({args.device})")
        print(f"Model: {name}")
        print(f"Path: {path}")
        print(f"Frames: {res['frames']}")
        print(f"Total Time: {res['total_s']:.2f}s")
        print(f"FPS: {res['fps']:.2f}")
        print(f"Latency: {res['latency_ms']:.2f} ms/frame")
        print("=" * 40)

    print_table(rows)
    save_csv(rows, args.out)


if __name__ == "__main__":
    main()
