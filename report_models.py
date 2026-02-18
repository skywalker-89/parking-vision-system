# report_4models.py
# Compare multiple YOLO models in an Ultralytics-style table:
# Model | size (pixels) | mAPval 50-95 | Speed CPU ONNX (ms) | Speed GPU (ms) | params (M) | FLOPs (B)
#
# Requires (inside your venv):
#   pip install opencv-python onnx onnxruntime thop
#
# Example (PowerShell):
# .\.venv\Scripts\python.exe .\report_4models.py `
#   --models y11n="runs\detect\y11n\weights\best.pt" `
#   --models y11s="runs\detect\y11s\weights\best.pt" `
#   --models y26n="runs\detect\y26n\weights\best.pt" `
#   --models y26s="runs\detect\y26s\weights\best.pt" `
#   --data "data.yaml" --video "videos\demo.mp4" --device 0 --imgsz 640 --split val --out "report_4models.csv"

import argparse
import contextlib
import io
import re
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# -----------------------------
# Formatting helpers
# -----------------------------
def fmt(x, decimals=2):
    if x is None:
        return "N/A"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{decimals}f}"
    return str(x)


def make_ultra_style_table(rows):
    headers = [
        "Model",
        "size (pixels)",
        "mAPval 50-95",
        "Speed CPU ONNX (ms)",
        "Speed GPU (ms)",
        "params (M)",
        "FLOPs (B)",
    ]

    table_rows = []
    for r in rows:
        table_rows.append([
            r["Model"],
            fmt(r["imgsz"], 0),
            fmt(r["map5095"], 3),
            fmt(r["cpu_onnx_ms"], 2),
            fmt(r["gpu_ms"], 2),
            fmt(r["params_m"], 2),
            fmt(r["flops_b"], 2),
        ])

    widths = []
    for i, h in enumerate(headers):
        w = len(h)
        for row in table_rows:
            w = max(w, len(str(row[i])))
        widths.append(w)

    def join_row(row):
        return " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers)))

    sep = "-+-".join("-" * w for w in widths)

    out = []
    out.append("=== REPORT TABLE ===")
    out.append(join_row(headers))
    out.append(sep)
    for row in table_rows:
        out.append(join_row(row))
    return "\n".join(out)


# -----------------------------
# Metrics / speed
# -----------------------------
def get_map_metrics(model: YOLO, data: str, split: str, imgsz: int, device: str):
    metrics = model.val(data=data, split=split, imgsz=imgsz, device=device, verbose=False)
    map50 = float(metrics.box.map50) if hasattr(metrics, "box") else None
    map5095 = float(metrics.box.map) if hasattr(metrics, "box") else None
    return map50, map5095


def bench_latency_ms(model: YOLO, video: str, device: str, warmup: int = 20, max_frames: int = 0):
    """
    End-to-end latency per frame (ms) using Ultralytics predict().
    max_frames=0 => full video
    """
    import torch

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")

    # warmup
    w = 0
    while w < warmup:
        ok, frame = cap.read()
        if not ok:
            break
        with torch.inference_mode():
            _ = model.predict(frame, device=device, verbose=False)
        w += 1

    frames = 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        with torch.inference_mode():
            _ = model.predict(frame, device=device, verbose=False)
        frames += 1
        if max_frames and frames >= max_frames:
            break

    # sync GPU timing (only once at end is usually enough for avg latency)
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    cap.release()

    if frames == 0:
        return None

    total = max(t1 - t0, 1e-9)
    return (total / frames) * 1000.0


# -----------------------------
# Params + FLOPs
# -----------------------------
def _parse_params_gflops_from_info(model: YOLO):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            model.info()
        except Exception:
            pass
    txt = buf.getvalue()

    params = None
    gflops = None

    m_params = re.search(r"([\d,]+)\s+parameters", txt)
    if m_params:
        params = int(m_params.group(1).replace(",", ""))

    m_gflops = re.search(r"([0-9]*\.?[0-9]+)\s+GFLOPs", txt)
    if m_gflops:
        gflops = float(m_gflops.group(1))

    return params, gflops


def get_params_flops_thop(pt_path: str, imgsz: int):
    """
    Returns:
      params_m (float): params in millions
      flops_b (float): FLOPs (B) ~= GFLOPs
    """
    import torch
    from thop import profile

    model = YOLO(pt_path)
    net = model.model

    # params
    params_m = None
    try:
        params_m = sum(p.numel() for p in net.parameters()) / 1e6
    except Exception:
        params_m = None

    # FLOPs
    flops_b = None
    try:
        net = net.to("cpu").eval()
        dummy = torch.zeros(1, 3, imgsz, imgsz)
        with torch.no_grad():
            flops, _ = profile(net, inputs=(dummy,), verbose=False)
        flops_b = float(flops) / 1e9
    except Exception:
        flops_b = None

    return params_m, flops_b


def get_params_flops(pt_path: str, imgsz: int):
    """
    Try:
      1) parse from model.info() (fast)
      2) fallback THOP (reliable)
    """
    model = YOLO(pt_path)

    params, gflops = _parse_params_gflops_from_info(model)

    # params fallback
    if params is None:
        try:
            params = sum(p.numel() for p in model.model.parameters())
        except Exception:
            params = None

    params_m = (params / 1e6) if params else None
    flops_b = gflops  # GFLOPs ~= FLOPs(B)

    if flops_b is None or params_m is None:
        thop_params_m, thop_flops_b = get_params_flops_thop(pt_path, imgsz)
        if params_m is None:
            params_m = thop_params_m
        if flops_b is None:
            flops_b = thop_flops_b

    return params_m, flops_b


# -----------------------------
# ONNX export + CPU speed
# -----------------------------
def export_onnx_if_needed(pt_path: str, imgsz: int):
    """
    Export PT -> ONNX once.
    simplify=False avoids onnxslim/onnxsim problems.
    """
    pt = Path(pt_path)
    onnx_path = pt.with_suffix(".onnx")
    if onnx_path.exists():
        return str(onnx_path)

    m = YOLO(str(pt))
    exported = m.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=False, half=False)
    return str(exported)


# -----------------------------
# Main
# -----------------------------
def parse_models(items):
    """
    items: ["name=path", ...]
    """
    out = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"--models must be like name=path, got: {it}")
        name, path = it.split("=", 1)
        name = name.strip()
        path = path.strip().strip('"').strip("'")
        out.append((name, path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="append", required=True,
                    help='Repeatable. Example: --models y11n="runs\\detect\\y11n\\weights\\best.pt"')
    ap.add_argument("--data", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--device", default="0")  # GPU device for val + GPU speed (e.g. "0") or "cpu"
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = full video")
    ap.add_argument("--out", default="report_4models.csv")
    args = ap.parse_args()

    models = parse_models(args.models)

    rows = []
    for name, pt_path in models:
        print(f"\n=== {name} ===")
        print(f"PT: {pt_path}")

        model = YOLO(pt_path)

        # mAP (GPU)
        map50, map5095 = get_map_metrics(model, args.data, args.split, args.imgsz, args.device)

        # params + FLOPs (B)
        params_m, flops_b = get_params_flops(pt_path, args.imgsz)

        # GPU latency (ms)
        gpu_ms = bench_latency_ms(model, args.video, device=args.device, warmup=args.warmup, max_frames=args.max_frames)

        # CPU ONNX latency (ms)
        onnx_path = export_onnx_if_needed(pt_path, args.imgsz)
        onnx_model = YOLO(onnx_path, task="detect")
        cpu_onnx_ms = bench_latency_ms(onnx_model, args.video, device="cpu", warmup=args.warmup, max_frames=args.max_frames)

        rows.append({
            "Model": name,
            "imgsz": args.imgsz,
            "map50": map50,
            "map5095": map5095,
            "cpu_onnx_ms": cpu_onnx_ms,
            "gpu_ms": gpu_ms,
            "params_m": params_m,
            "flops_b": flops_b,
            "pt_path": pt_path,
            "onnx_path": onnx_path,
        })

    # Print table (Ultralytics-style)
    print()
    print(make_ultra_style_table(rows))

    # Save CSV (includes mAP50 too)
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Model,imgsz,mAP50,mAP50_95,Speed_CPU_ONNX_ms,Speed_GPU_ms,params_M,FLOPs_B,pt_path,onnx_path\n")
        for r in rows:
            f.write(
                f'{r["Model"]},{r["imgsz"]},{fmt(r["map50"],6)},{fmt(r["map5095"],6)},'
                f'{fmt(r["cpu_onnx_ms"],6)},{fmt(r["gpu_ms"],6)},'
                f'{fmt(r["params_m"],6)},{fmt(r["flops_b"],6)},'
                f'"{r["pt_path"]}","{r["onnx_path"]}"\n'
            )

    print(f"\nSaved CSV -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
