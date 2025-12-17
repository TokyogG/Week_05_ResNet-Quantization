# Week 05 — Day 01: Quantization Fundamentals

## Goal
Understand *why* quantization exists, *what* it does mathematically, and *how* it affects edge inference pipelines.

## Topics
- FP32 vs FP16 vs INT8
- Scale and zero-point
- Calibration
- PTQ vs QAT

## Diagrams
FP32 Inference
─────────────
Sensor → Normalize → FP32 Tensor → FP32 MAC → FP32 Output
                    (slow, big, power-hungry)

INT8 Inference
─────────────
Sensor → Normalize → Quantize → INT8 Tensor
                              ↓
                        INT8 MAC (fast)
                              ↓
                      Dequantize → Output

PTQ vs QAT (Critical Distinction)

PTQ
FP32 Model → Calibration → INT8 Model

QAT
FP32 Model → Fake-Quant Training → INT8 Model

## Key Takeaways

Edge AI systems are always constrained by latency, compute, memory, and power.
Because of these constraints, quantization is not optional — it’s a requirement for real deployments.

In practice, this usually means moving from FP32 to INT8.
Reducing precision from 32 bits to 8 bits dramatically lowers memory usage and compute cost, which directly reduces latency and power consumption. The tradeoff is reduced numerical precision, which can introduce error if not handled carefully.

Quantization is math, not magic.
When compressing 32 bits of information into 8 bits, precision is lost by definition. The goal is to control where that error goes, not eliminate it.

This is where calibration matters.
Calibration data is used to determine the scale and zero-point that map floating-point values into the INT8 range. For quantization to work well, the calibration data must reflect the real input distribution. Poor calibration leads to clipping, saturation, and distribution shift — all of which hurt accuracy.

Quantization can be applied after training using Post-Training Quantization (PTQ), or during training using Quantization-Aware Training (QAT). PTQ is faster and often sufficient, while QAT is used when accuracy loss must be minimized.

As an Edge AI engineer, these are system-level tradeoffs we make:  balancing accuracy, latency, memory, and power based on the hardware and deployment goals.

The key idea is that quantization is a system design decision, not just a model optimization.