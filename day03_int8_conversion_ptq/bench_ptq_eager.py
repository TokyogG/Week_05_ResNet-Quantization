#!/usr/bin/env python3
"""
Day03: Eager-mode PTQ benchmark on Raspberry Pi (reliable).
Benchmarks FP32 vs INT8 for torchvision quantization-ready models.

Usage:
  python3 bench_ptq_eager.py --model mobilenet_v2 --threads 1
  python3 bench_ptq_eager.py --model resnet18 --threads 1
"""

import argparse, time, statistics, os
import torch

from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2
from torchvision.models.quantization import resnet18 as q_resnet18

def percentile(sorted_vals, p):
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

@torch.inference_mode()
def bench(model, x, iters=200, warmup=20):
    for _ in range(warmup):
        _ = model(x)
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter()
        _ = model(x)
        t1=time.perf_counter()
        ts.append((t1-t0)*1000)
    s=sorted(ts)
    mean=statistics.mean(ts)
    p50=percentile(s,50)
    p95=percentile(s,95)
    fps=1000.0/mean
    return mean,p50,p95,fps

def build_fp32(model_name):
    if model_name=="mobilenet_v2":
        m=q_mobilenet_v2(weights="DEFAULT", quantize=False).eval()
    else:
        m=q_resnet18(weights="DEFAULT", quantize=False).eval()
    return m

def ptq_int8(fp32, x, calib_steps, engine):
    torch.backends.quantized.engine = engine
    if hasattr(fp32, "fuse_model"):
        fp32.fuse_model()
    fp32.qconfig = torch.quantization.get_default_qconfig(torch.backends.quantized.engine)
    prepared = torch.quantization.prepare(fp32, inplace=False)

    for _ in range(calib_steps):
        _ = prepared(x)

    int8 = torch.quantization.convert(prepared, inplace=False).eval()
    return int8

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mobilenet_v2","resnet18"], required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--calib_steps", type=int, default=50)
    ap.add_argument("--engine", type=str, default="qnnpack", choices=["qnnpack","onednn"])
    args=ap.parse_args()

    torch.set_num_threads(args.threads)
    x=torch.randn(1,3,224,224)

    fp32=build_fp32(args.model)
    int8=ptq_int8(fp32, x, args.calib_steps, args.engine)

    m1,p50_1,p95_1,fps1=bench(fp32,x,args.iters,args.warmup)
    m2,p50_2,p95_2,fps2=bench(int8,x,args.iters,args.warmup)

    print("\n=== Eager PTQ Benchmark ===")
    print(f"Model:   {args.model}")
    print(f"Engine:  {args.engine}")
    print(f"Threads: {args.threads}")
    print(f"Input:   (1,3,224,224)\n")

    print("FP32")
    print(f"  Mean: {m1:.3f} ms | P50: {p50_1:.3f} ms | P95: {p95_1:.3f} ms | FPS: {fps1:.2f}")

    print("\nINT8 (PTQ)")
    print(f"  Mean: {m2:.3f} ms | P50: {p50_2:.3f} ms | P95: {p95_2:.3f} ms | FPS: {fps2:.2f}")

    print(f"\nSpeedup (FP32/INT8): {(m1/m2):.2f}x")

if __name__=="__main__":
    main()
