#!/usr/bin/env python3
"""
Day04: Raspberry Pi 5 inference benchmarks with locked execution policy support.

- Benchmarks FP32 vs INT8 (PTQ eager) for torchvision quantization-ready models.
- Produces stable latency stats (mean, p50, p90, p95, p99) + throughput.
- Saves results to JSON + CSV in out_dir/ (default: results/)
- Captures basic device snapshot: temp, throttling, governor, freq (best-effort).

Usage:
  python3 bench_pi5_inference.py --model mobilenet_v2 --threads 1
  python3 bench_pi5_inference.py --model resnet18 --threads 1 --engine qnnpack
"""

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2
from torchvision.models.quantization import resnet18 as q_resnet18


# ----------------------------
# Helpers: system snapshot
# ----------------------------
def _run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return out
    except Exception:
        return None


def _read_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def get_pi_snapshot() -> Dict[str, object]:
    snap: Dict[str, object] = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", "unknown"),
        "cpu_governors": {},
        "cpu_cur_freq_khz": {},
        "vcgencmd_temp": None,
        "vcgencmd_throttled": None,
    }

    # vcgencmd is common on Raspberry Pi OS; best-effort on Ubuntu
    snap["vcgencmd_temp"] = _run_cmd(["vcgencmd", "measure_temp"])
    snap["vcgencmd_throttled"] = _run_cmd(["vcgencmd", "get_throttled"])

    # governor / freq (best-effort)
    cpu_root = "/sys/devices/system/cpu"
    if os.path.isdir(cpu_root):
        for cpu in sorted([d for d in os.listdir(cpu_root) if d.startswith("cpu") and d[3:].isdigit()]):
            gov = _read_file(f"{cpu_root}/{cpu}/cpufreq/scaling_governor")
            cur = _read_file(f"{cpu_root}/{cpu}/cpufreq/scaling_cur_freq")
            if gov is not None:
                snap["cpu_governors"][cpu] = gov
            if cur is not None:
                snap["cpu_cur_freq_khz"][cpu] = cur

    return snap


def set_cpu_affinity(core: Optional[int]) -> None:
    if core is None:
        return
    try:
        os.sched_setaffinity(0, {int(core)})
    except Exception:
        # Not fatal; wrapper script (taskset) can still do this externally.
        pass


# ----------------------------
# Stats
# ----------------------------
def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


@dataclass
class BenchResult:
    model: str
    engine: str
    threads_intraop: int
    threads_interop: int
    pin_core: Optional[int]
    iters: int
    warmup: int
    calib_steps: int
    input_shape: Tuple[int, int, int, int]

    fp32_mean_ms: float
    fp32_p50_ms: float
    fp32_p90_ms: float
    fp32_p95_ms: float
    fp32_p99_ms: float
    fp32_fps: float

    int8_mean_ms: float
    int8_p50_ms: float
    int8_p90_ms: float
    int8_p95_ms: float
    int8_p99_ms: float
    int8_fps: float

    speedup_fp32_over_int8: float

    snapshot: Dict[str, object]


@torch.inference_mode()
def bench(model: torch.nn.Module, x: torch.Tensor, iters: int = 200, warmup: int = 20) -> Tuple[float, float, float, float, float, float]:
    # Warmup (exclude from stats)
    for _ in range(warmup):
        _ = model(x)

    lat_ms: List[float] = []

    # Time total as well (throughput stability)
    t_total0 = time.perf_counter_ns()
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _ = model(x)
        t1 = time.perf_counter_ns()
        lat_ms.append((t1 - t0) / 1e6)
    t_total1 = time.perf_counter_ns()

    s = sorted(lat_ms)
    mean = statistics.mean(lat_ms)
    p50 = percentile(s, 50)
    p90 = percentile(s, 90)
    p95 = percentile(s, 95)
    p99 = percentile(s, 99)

    total_s = (t_total1 - t_total0) / 1e9
    fps = (iters / total_s) if total_s > 0 else float("nan")
    return mean, p50, p90, p95, p99, fps


def build_fp32(model_name: str) -> torch.nn.Module:
    if model_name == "mobilenet_v2":
        return q_mobilenet_v2(weights="DEFAULT", quantize=False).eval()
    return q_resnet18(weights="DEFAULT", quantize=False).eval()


def ptq_int8(fp32: torch.nn.Module, x: torch.Tensor, calib_steps: int, engine: str) -> torch.nn.Module:
    torch.backends.quantized.engine = engine
    if hasattr(fp32, "fuse_model"):
        fp32.fuse_model()

    fp32.qconfig = torch.quantization.get_default_qconfig(torch.backends.quantized.engine)
    prepared = torch.quantization.prepare(fp32, inplace=False)

    for _ in range(calib_steps):
        _ = prepared(x)

    int8 = torch.quantization.convert(prepared, inplace=False).eval()
    return int8


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_csv(path: str, row: Dict[str, object]) -> None:
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mobilenet_v2", "resnet18"], required=True)
    ap.add_argument("--threads", type=int, default=1, help="intra-op threads (torch.set_num_threads)")
    ap.add_argument("--interop_threads", type=int, default=1, help="inter-op threads (torch.set_num_interop_threads)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--calib_steps", type=int, default=50)
    ap.add_argument("--engine", type=str, default="qnnpack", choices=["qnnpack", "onednn"])
    ap.add_argument("--pin_core", type=int, default=None, help="Best-effort CPU affinity (also use taskset in wrapper)")
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--tag", type=str, default="", help="Optional tag appended to output filenames")
    args = ap.parse_args()

    # Make runs more repeatable
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    set_cpu_affinity(args.pin_core)

    # Fixed input tensor (determinism is not critical for latency, but keep it stable)
    x = torch.randn(1, 3, 224, 224)

    fp32 = build_fp32(args.model)
    int8 = ptq_int8(fp32, x, args.calib_steps, args.engine)

    snap = get_pi_snapshot()

    fp32_stats = bench(fp32, x, args.iters, args.warmup)
    int8_stats = bench(int8, x, args.iters, args.warmup)

    (m1, p50_1, p90_1, p95_1, p99_1, fps1) = fp32_stats
    (m2, p50_2, p90_2, p95_2, p99_2, fps2) = int8_stats

    speedup = (m1 / m2) if m2 > 0 else float("inf")

    result = BenchResult(
        model=args.model,
        engine=args.engine,
        threads_intraop=args.threads,
        threads_interop=args.interop_threads,
        pin_core=args.pin_core,
        iters=args.iters,
        warmup=args.warmup,
        calib_steps=args.calib_steps,
        input_shape=(1, 3, 224, 224),
        fp32_mean_ms=m1,
        fp32_p50_ms=p50_1,
        fp32_p90_ms=p90_1,
        fp32_p95_ms=p95_1,
        fp32_p99_ms=p99_1,
        fp32_fps=fps1,
        int8_mean_ms=m2,
        int8_p50_ms=p50_2,
        int8_p90_ms=p90_2,
        int8_p95_ms=p95_2,
        int8_p99_ms=p99_2,
        int8_fps=fps2,
        speedup_fp32_over_int8=speedup,
        snapshot=snap,
    )

    ensure_dir(args.out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    base = f"pi5_{args.model}_{args.engine}_t{args.threads}_i{args.iters}_w{args.warmup}_c{args.calib_steps}{tag}_{ts}"

    json_path = os.path.join(args.out_dir, f"{base}.json")
    csv_path = os.path.join(args.out_dir, "pi5_benchmarks.csv")

    write_json(json_path, asdict(result))
    append_csv(csv_path, {k: v for k, v in asdict(result).items() if k != "snapshot"})

    print("\n=== Pi 5 Eager PTQ Benchmark (Locked Policy Friendly) ===")
    print(f"Model:   {args.model}")
    print(f"Engine:  {args.engine}")
    print(f"Threads: intra={args.threads} interop={args.interop_threads}")
    print(f"PinCore: {args.pin_core}")
    print(f"Input:   (1,3,224,224)")
    print(f"Saved:   {json_path}")
    print(f"CSV:     {csv_path}\n")

    print("FP32")
    print(f"  Mean: {m1:.3f} ms | P50: {p50_1:.3f} | P90: {p90_1:.3f} | P95: {p95_1:.3f} | P99: {p99_1:.3f} | FPS: {fps1:.2f}")

    print("\nINT8 (PTQ)")
    print(f"  Mean: {m2:.3f} ms | P50: {p50_2:.3f} | P90: {p90_2:.3f} | P95: {p95_2:.3f} | P99: {p99_2:.3f} | FPS: {fps2:.2f}")

    print(f"\nSpeedup (FP32/INT8): {speedup:.2f}x")


if __name__ == "__main__":
    main()