# Week_05_ResNet-Quantization

## Day 01 — Quantization Fundamentals (Theory First)

Focus

Day 01 establishes the theoretical foundation required to deploy real neural networks on constrained edge hardware.
Rather than starting with code, this day focuses on why quantization is necessary and what tradeoffs it introduces.

Topics Covered

    Edge constraints: latency, compute, memory, and power

    FP32 vs FP16 vs INT8 numeric formats

    Quantization math: scale and zero-point

    Calibration and input distributions

    Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)

Key Outcome

By the end of Day 01, the system architecture built in Week 04 is understood through a quantization-first lens, ensuring that real models introduced later in the week can be integrated without redesigning the execution pipeline.

No code is written on Day 01. The deliverable is a clear mental model that guides all quantization and benchmarking work that follows.

### Day 02 — FP32 Baseline (Pi 5 CPU)

| Model | Precision | Mean Latency | P95 Latency | FPS | RSS Memory |
|------|----------|--------------|-------------|-----|------------|
| ResNet-18 | FP32 | 77.5 ms | 78.1 ms | 12.9 | 456 MB |
| MobileNet-V2 | FP32 | 71.8 ms | 73.1 ms | 13.9 | 421 MB |

These results establish the baseline that INT8 quantization must improve upon in Day 03.
