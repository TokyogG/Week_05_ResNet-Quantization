#!/usr/bin/env python3
"""
Benchmark TorchScript FP32 vs INT8 PTQ artifacts.

Outputs: mean / p50 / p95 ms and FPS, plus model file sizes.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch


def _percentile(sorted_vals: List[float], p: float) -> float:
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


def size_mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024)


@torch.inference_mode()
def run(model: torch.jit.ScriptModule, x: torch.Tensor, iters: int, warmup: int) -> List[float]:
    for _ in range(warmup):
        _ = model(x)

    times: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def summarize(times: List[float]) -> Tuple[float, float, float, float]:
    s = sorted(times)
    mean = statistics.mean(times)
    p50 = _percentile(s, 50)
    p95 = _percentile(s, 95)
    fps = 1000.0 / mean if mean > 0 else float("inf")
    return mean, p50, p95, fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True, help="Path to FP32 TorchScript .pt")
    ap.add_argument("--int8", required=True, help="Path to INT8 TorchScript .pt")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    fp32_path = Path(args.fp32)
    int8_path = Path(args.int8)

    x = torch.randn(1, 3, args.size, args.size)

    fp32 = torch.jit.load(str(fp32_path))
    int8 = torch.jit.load(str(int8_path))
    fp32.eval()
    int8.eval()

    t_fp32 = run(fp32, x, args.iters, args.warmup)
    t_int8 = run(int8, x, args.iters, args.warmup)

    m1, p50_1, p95_1, fps1 = summarize(t_fp32)
    m2, p50_2, p95_2, fps2 = summarize(t_int8)

    print("\n=== TorchScript Benchmark ===")
    print(f"Threads: {args.threads}")
    print(f"Input:   (1,3,{args.size},{args.size})\n")

    print("FP32")
    print(f"  File: {fp32_path} ({size_mb(fp32_path):.2f} MB)")
    print(f"  Mean: {m1:.3f} ms | P50: {p50_1:.3f} ms | P95: {p95_1:.3f} ms | FPS: {fps1:.2f}")

    print("\nINT8 (PTQ)")
    print(f"  File: {int8_path} ({size_mb(int8_path):.2f} MB)")
    print(f"  Mean: {m2:.3f} ms | P50: {p50_2:.3f} ms | P95: {p95_2:.3f} ms | FPS: {fps2:.2f}")

    speedup = m1 / m2 if m2 > 0 else float("inf")
    print(f"\nSpeedup (FP32/INT8): {speedup:.2f}x")


if __name__ == "__main__":
    main()