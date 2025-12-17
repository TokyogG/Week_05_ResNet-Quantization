#!/usr/bin/env python3
"""
FP32 baseline inference benchmark for Raspberry Pi 5 (CPU).

- Models: resnet18, mobilenet_v2 (torchvision)
- Measures: mean latency, p50/p95, throughput, RSS memory
- Input: optional image file, otherwise random tensor
- Output: prints summary + optionally appends to CSV

Usage examples:
  python3 fp32_baseline.py --model resnet18 --iters 200 --warmup 20
  python3 fp32_baseline.py --model mobilenet_v2 --image ./test.jpg
  python3 fp32_baseline.py --model resnet18 --threads 1 --csv results_fp32.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class BenchResult:
    model: str
    device: str
    threads: int
    iters: int
    warmup: int
    input_shape: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float
    rss_mb: Optional[float]


def _percentile(sorted_vals: List[float], p: float) -> float:
    """Compute percentile p in [0,100] for a sorted list."""
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def get_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def load_model(name: str) -> torch.nn.Module:
    from torchvision.models import resnet18, mobilenet_v2

    name = name.lower().strip()
    if name == "resnet18":
        model = resnet18(weights="DEFAULT")
    elif name in ("mobilenet_v2", "mobilenetv2"):
        model = mobilenet_v2(weights="DEFAULT")
    else:
        raise ValueError(f"Unsupported model: {name}")
    model.eval()
    return model


def load_input(image_path: Optional[str], size: int = 224) -> torch.Tensor:
    # Standard ImageNet preprocessing
    transform = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    if image_path:
        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0)  # NCHW
    else:
        # Random but correctly-shaped tensor (acts as synthetic baseline)
        x = torch.randn(1, 3, size, size)

    return x


@torch.inference_mode()
def run_benchmark(
    model: torch.nn.Module,
    x: torch.Tensor,
    iters: int,
    warmup: int,
    device: torch.device,
) -> Tuple[List[float], torch.Tensor]:
    model = model.to(device)
    x = x.to(device)

    # Warmup
    for _ in range(warmup):
        _ = model(x)

    times_ms: List[float] = []
    y_last = None

    # Timed loop
    for _ in range(iters):
        t0 = time.perf_counter()
        y_last = model(x)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return times_ms, y_last


def maybe_append_csv(csv_path: Optional[str], result: BenchResult) -> None:
    if not csv_path:
        return

    is_new = not os.path.exists(csv_path)
    header = [
        "model", "device", "threads", "iters", "warmup", "input_shape",
        "mean_ms", "p50_ms", "p95_ms", "fps", "rss_mb"
    ]
    row = [
        result.model, result.device, result.threads, result.iters, result.warmup, result.input_shape,
        f"{result.mean_ms:.4f}", f"{result.p50_ms:.4f}", f"{result.p95_ms:.4f}", f"{result.fps:.4f}",
        "" if result.rss_mb is None else f"{result.rss_mb:.2f}",
    ]

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v2"])
    ap.add_argument("--image", type=str, default=None, help="Optional image path for realistic input.")
    ap.add_argument("--size", type=int, default=224, help="Input spatial size (default 224).")
    ap.add_argument("--iters", type=int, default=200, help="Timed iterations.")
    ap.add_argument("--warmup", type=int, default=20, help="Warmup iterations (not timed).")
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads; 0 = leave default.")
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV path to append results.")
    args = ap.parse_args()

    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)

    device = torch.device("cpu")
    model = load_model(args.model)
    x = load_input(args.image, size=args.size)

    rss_before = get_rss_mb()
    times_ms, y_last = run_benchmark(model, x, args.iters, args.warmup, device)
    rss_after = get_rss_mb()

    times_sorted = sorted(times_ms)
    mean_ms = statistics.mean(times_ms)
    p50_ms = _percentile(times_sorted, 50)
    p95_ms = _percentile(times_sorted, 95)
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")

    rss_mb = None
    if rss_before is not None and rss_after is not None:
        # report peak-ish as "after" (simple, consistent)
        rss_mb = max(rss_before, rss_after)

    result = BenchResult(
        model=args.model,
        device=str(device),
        threads=torch.get_num_threads(),
        iters=args.iters,
        warmup=args.warmup,
        input_shape=str(tuple(x.shape)),
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        fps=fps,
        rss_mb=rss_mb,
    )

    print("\n=== FP32 Baseline Benchmark ===")
    print(f"Model:        {result.model}")
    print(f"Device:       {result.device}")
    print(f"Threads:      {result.threads}")
    print(f"Input shape:  {result.input_shape}")
    print(f"Iters:        {result.iters} (+ warmup {result.warmup})")
    print(f"Mean:         {result.mean_ms:.3f} ms")
    print(f"P50:          {result.p50_ms:.3f} ms")
    print(f"P95:          {result.p95_ms:.3f} ms")
    print(f"Throughput:   {result.fps:.2f} FPS")
    if result.rss_mb is None:
        print("RSS:          (psutil not installed; pip install psutil)")
    else:
        print(f"RSS:          {result.rss_mb:.2f} MB")

    # Optional: sanity print output shape
    if hasattr(y_last, "shape"):
        print(f"Output shape: {tuple(y_last.shape)}")

    maybe_append_csv(args.csv, result)
    if args.csv:
        print(f"\nAppended results to: {args.csv}")


if __name__ == "__main__":
    main()