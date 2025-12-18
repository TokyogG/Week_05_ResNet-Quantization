#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) to INT8 using PyTorch eager mode.

- Quantizes: mobilenet_v2, resnet18 (torchvision)
- Uses: fbgemm backend (x86), qnnpack backend (ARM)
- On Raspberry Pi (ARM), QNNPACK is typically the intended backend.
- Produces TorchScript models:
    artifacts/<model>_fp32.pt
    artifacts/<model>_int8_ptq.pt

This is a Day-03 "clean" PTQ pipeline:
- Small calibration loop
- No dataset dependency (can calibrate on random or images)
- Deterministic, reproducible artifacts
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image


def pick_backend() -> str:
    # ARM usually uses qnnpack, x86 uses fbgemm
    # On Pi 5: choose qnnpack by default.
    if "qnnpack" in torch.backends.quantized.supported_engines:
        return "qnnpack"
    return torch.backends.quantized.supported_engines[0]


def load_model(name: str) -> torch.nn.Module:
    name = name.lower().strip()

    # Use quantization-ready torchvision models (have QuantStub/DeQuantStub)
    from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2
    from torchvision.models.quantization import resnet18 as q_resnet18

    if name == "mobilenet_v2":
        m = q_mobilenet_v2(weights="DEFAULT", quantize=False)
    elif name == "resnet18":
        m = q_resnet18(weights="DEFAULT", quantize=False)
    else:
        raise ValueError(f"Unsupported model: {name}")

    m.eval()
    return m


def load_input(image_path: Optional[str], size: int = 224) -> torch.Tensor:
    tfm = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    if image_path:
        img = Image.open(image_path).convert("RGB")
        x = tfm(img).unsqueeze(0)
    else:
        # calibration doesn't need labels for PTQ;
        # random is ok for timing + pipeline sanity,
        # but real images are better for accuracy stability.
        x = torch.randn(1, 3, size, size)

    return x


@torch.inference_mode()
def calibrate(model: torch.nn.Module, x: torch.Tensor, steps: int) -> None:
    for _ in range(steps):
        _ = model(x)


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


@torch.inference_mode()
def sanity_compare(fp32: torch.nn.Module, int8: torch.nn.Module, x: torch.Tensor) -> None:
    y_fp = fp32(x)
    y_q = int8(x)

    # Compare top-1 class agreement as a quick sanity check
    top_fp = int(torch.argmax(y_fp, dim=1).item())
    top_q = int(torch.argmax(y_q, dim=1).item())

    print(f"Sanity top-1 (FP32): {top_fp}")
    print(f"Sanity top-1 (INT8): {top_q}")
    print("Top-1 match?", "YES" if top_fp == top_q else "NO (not necessarily bad without real images)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mobilenet_v2", "resnet18"], required=True)
    ap.add_argument("--image", type=str, default=None, help="Optional image for calibration + sanity.")
    ap.add_argument("--calib_steps", type=int, default=50, help="Calibration forward passes.")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--outdir", type=str, default="day03_int8_conversion_ptq/artifacts")
    args = ap.parse_args()

    backend = pick_backend()
    torch.backends.quantized.engine = backend
    print(f"Quant backend: {torch.backends.quantized.engine}")
    print(f"Supported engines: {torch.backends.quantized.supported_engines}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load model + input
    fp32 = load_model(args.model)
    x = load_input(args.image, size=args.size)

    # Save FP32 TorchScript
    fp32_ts_path = outdir / f"{args.model}_fp32.pt"
    fp32_ts = torch.jit.trace(fp32, x)
    fp32_ts.save(str(fp32_ts_path))

    # Prepare for PTQ
    # 1) (Optional) Fusion skipped for Day 03.
    # Some models require specific fuse patterns; an empty fuse list errors.
    model_for_ptq = fp32

    # Fuse (quantization-ready models expose fuse_model())
    if hasattr(model_for_ptq, "fuse_model"):
        model_for_ptq.fuse_model()

    # 2) Set qconfig and prepare
    model_for_ptq.qconfig = torch.quantization.get_default_qconfig(torch.backends.quantized.engine)
    prepared = torch.quantization.prepare(model_for_ptq, inplace=False)

    # 3) Calibrate
    t0 = time.perf_counter()
    calibrate(prepared, x, steps=args.calib_steps)
    t1 = time.perf_counter()
    print(f"Calibration: {args.calib_steps} steps in {(t1 - t0)*1000:.1f} ms")

    # 4) Convert to INT8
    int8_model = torch.quantization.convert(prepared, inplace=False)
    int8_model.eval()

    # Save INT8 TorchScript
    int8_ts_path = outdir / f"{args.model}_int8_ptq.pt"
    # int8_ts = torch.jit.trace(int8_model, x)
    int8_ts = torch.jit.script(int8_model)
    int8_ts.save(str(int8_ts_path))

    print("\n=== Artifacts ===")
    print(f"FP32: {fp32_ts_path} ({model_size_mb(fp32_ts_path):.2f} MB)")
    print(f"INT8: {int8_ts_path} ({model_size_mb(int8_ts_path):.2f} MB)")

    print("\n=== Sanity Check ===")
    sanity_compare(fp32, int8_model, x)


if __name__ == "__main__":
    main()