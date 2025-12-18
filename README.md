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

FP32 Baseline vs INT8 Expectations

ResNet-18 and MobileNet-V2 represent two different design philosophies. ResNet-18 is a general-purpose vision backbone optimized for accuracy and deep feature extraction, while MobileNet-V2 is explicitly designed for mobile and edge environments using depthwise separable convolutions.

Despite MobileNet-V2 being “edge-friendly” by architecture, FP32 benchmarks on the Raspberry Pi 5 show that both models remain slow and memory-heavy when executed without quantization. This highlights an important edge-AI principle: architectural efficiency alone is insufficient without matching numerical precision to hardware constraints.

Under INT8 quantization, ResNet-18 is expected to benefit more in absolute terms due to its higher baseline compute and memory footprint. MobileNet-V2 typically remains efficient, but its heavy use of depthwise convolutions makes it more sensitive to quantization noise and calibration quality.

This comparison demonstrates that quantization is not merely a model optimization step, but a system-level design decision that determines whether real neural networks are viable on constrained edge hardware.

## Day 03 — INT8 Post-Training Quantization (PTQ) on Pi 5 (CPU)

Eager-mode PTQ (QNNPACK) was evaluated for two CNNs:

- **MobileNet-V2** (depthwise/pointwise, edge-first)
- **ResNet-18** (residual blocks, general-purpose backbone)

### Results (Threads=1, Input=1×3×224×224)

| Model | Backend | FP32 Mean | INT8 Mean | Speedup |
|------|---------|----------:|----------:|--------:|
| MobileNet-V2 | qnnpack | 63.204 ms | 46.720 ms | **1.35×** |
| ResNet-18 | qnnpack | 75.150 ms | 213.187 ms | **0.35×** |

### Interpretation

INT8 quantization reduced MobileNet-V2 latency as expected, but ResNet-18 became slower under this eager PTQ + backend combination. This demonstrates an important edge engineering principle:

**Quantization is not guaranteed to speed up every model on CPU.** Performance depends on the backend (QNNPACK vs oneDNN), graph structure (depthwise vs residual adds), and how many quantize/dequantize boundaries are introduced.

This motivates (a) backend selection experiments and/or graph-mode quantization for CPU, and (b) accelerator deployment (Hailo) where INT8 execution is the native fast path.
