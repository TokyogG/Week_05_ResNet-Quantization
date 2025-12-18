Week 05 — Quantization & Real Models (Pi 5 + Hailo)
Day 01 — Quantization Fundamentals (Theory First)
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

No code is written on Day 01.
The deliverable is a clear mental model that guides all quantization and benchmarking work that follows.

Day 02 — FP32 Baseline (Pi 5 CPU)
Baseline Measurements
Model	Precision	Mean Latency	P95 Latency	FPS	RSS Memory
ResNet-18	FP32	77.5 ms	78.1 ms	12.9	456 MB
MobileNet-V2	FP32	71.8 ms	73.1 ms	13.9	421 MB

These results establish the unoptimized CPU baseline that INT8 quantization must improve upon in Day 03.

FP32 Baseline vs INT8 Expectations

ResNet-18 and MobileNet-V2 represent two different design philosophies:

ResNet-18 is a general-purpose vision backbone optimized for accuracy and deep feature extraction.

MobileNet-V2 is explicitly designed for mobile and edge environments using depthwise separable convolutions.

Despite MobileNet-V2 being “edge-friendly” by architecture, FP32 benchmarks on the Raspberry Pi 5 show that both models remain slow and memory-heavy when executed without quantization. This highlights a key edge-AI principle:

Architectural efficiency alone is insufficient without matching numerical precision to hardware constraints.

Under INT8 quantization, ResNet-18 is expected to benefit more in absolute terms due to its higher baseline compute and memory footprint. MobileNet-V2 typically remains efficient, but its heavy use of depthwise convolutions makes it more sensitive to quantization noise and calibration quality.

This comparison frames quantization not as a model tweak, but as a system-level design decision.

Day 03 — INT8 Post-Training Quantization (PTQ) on Pi 5 (CPU)
Setup

Eager-mode PTQ was evaluated using the two quantization backends supported by this PyTorch build:

QNNPACK — ARM-focused quantization backend

oneDNN — primarily optimized for x86 (INT8 often assumes VNNI-class instructions)

Benchmarks were performed in eager execution mode to reflect actual kernel performance on this platform.

Results

(Threads = 1, Input = 1×3×224×224)

Model	Backend	FP32 Mean	INT8 Mean	Speedup
MobileNet-V2	qnnpack	63.204 ms	46.720 ms	1.35×
ResNet-18	qnnpack	75.150 ms	213.187 ms	0.35×
ResNet-18	onednn	80.322 ms	849.841 ms	0.09×
Interpretation

MobileNet-V2 benefits from INT8 PTQ on Pi 5 CPU using QNNPACK, showing reduced latency and improved throughput.

ResNet-18 becomes slower under eager PTQ on this platform. Its residual structure introduces additional quantize/dequantize boundaries and operator overhead that outweigh INT8 gains.

oneDNN INT8 performs poorly on this ARM system and appears unsuitable without x86-class vector acceleration.

Conclusion

INT8 quantization is not guaranteed to improve performance on CPU.
On ARM platforms, performance depends strongly on backend choice, model structure, and graph execution characteristics.

These results motivate accelerator-based deployment (Hailo), where INT8 execution is the native fast path and a broader range of model architectures can benefit consistently.

Overall Assessment

Week 05 demonstrates that quantization is a necessary but not sufficient condition for edge performance. Correct numerical precision, backend selection, and hardware capabilities must align for INT8 to deliver real gains.

✅ Add this to README.md
Day 04 — Pi 5 Inference Benchmarks (Locked Execution Policy)
Focus

Day 04 transitions from functional benchmarking to reproducible, system-level benchmarking on edge hardware.

While Day 02 and Day 03 measured raw FP32 and INT8 performance, those measurements were still subject to OS scheduling, CPU frequency scaling, and background system noise. Day 04 introduces a locked execution policy to ensure that inference benchmarks are stable, repeatable, and defensible.

This mirrors how real embedded AI systems are evaluated in production and during performance validation.

Locked Execution Policy

Before benchmarking, the Raspberry Pi 5 execution environment was explicitly constrained:

CPU governor set to performance

CPU frequency locked at maximum (2.4 GHz)

Deep idle states disabled (where available)

Inference pinned to a single CPU core

Torch intra-op and inter-op threads fixed

Thermal state monitored to avoid throttling

Warmup iterations excluded from timing statistics

All inference runs were executed under identical conditions, producing consistent p50, p95, and p99 latency measurements.

Benchmark Methodology

Execution mode: PyTorch eager inference

Backend: QNNPACK (ARM-optimized)

Threads: 1

Input: 1×3×224×224

Metrics collected:

Mean latency

p50 / p90 / p95 / p99 latency

Throughput (FPS)

Results persisted to CSV and JSON for reproducibility

Results (MobileNet-V2, QNNPACK, Locked Policy)
Precision	Mean Latency	p95 Latency	p99 Latency	FPS
FP32	58.49 ms	58.92 ms	59.78 ms	17.10
INT8 (PTQ)	46.71 ms	47.02 ms	47.06 ms	21.41

Observed Speedup (FP32 → INT8): ~1.25×

Latency distributions were tightly clustered, indicating minimal scheduler or frequency-related jitter once the execution environment was locked.

Interpretation

These results confirm several important edge-AI principles:

Python can be used for reliable benchmarking when the operating system controls determinism

Quantized INT8 execution on CPU provides measurable gains, but improvements remain modest

Even with a fully optimized CPU execution environment, general-purpose processors remain latency-limited for real-time inference

Most importantly, Day 04 establishes a credible CPU baseline against which accelerator-based execution (Hailo) can be evaluated.

Key Takeaway

Day 04 demonstrates that quantization alone is not enough.
Even under ideal CPU conditions, INT8 inference on general-purpose processors delivers incremental gains, not transformational ones.

This motivates the transition to dedicated inference accelerators, where INT8 execution is the native fast path rather than an optimization layer.

Day 04 therefore serves as the final CPU benchmark gate before introducing Hailo-based deployment in the following phase of the bootcamp.

Day 05 — Week 05 Wrap-Up & Transition to Accelerators
Purpose

Day 05 consolidates the findings from Week 05 and frames the transition from CPU-based inference to dedicated edge accelerators.

Rather than introducing new tooling, this day focuses on system-level understanding: what quantization can and cannot solve on general-purpose CPUs, and why accelerators such as Hailo are necessary for real-time edge AI workloads.

Key Lessons from Week 05

Quantization is mandatory, but not sufficient
INT8 quantization reduces memory footprint and can improve latency, but gains on CPU are highly model- and backend-dependent.

Backend choice matters as much as precision
On ARM platforms, QNNPACK enables modest INT8 gains, while x86-oriented backends (e.g., oneDNN INT8) are ineffective without appropriate vector hardware.

Determinism is a system property
Reliable benchmarking required explicit control of CPU frequency, scheduling, threading, and thermal state. Once the execution environment was locked, Python proved sufficient as an orchestration layer.

CPU optimization has diminishing returns
Even under ideal conditions, optimized INT8 inference on the Pi 5 CPU delivers incremental improvements, not step-change performance.

Why Accelerators Are Required

Week 05 demonstrates that general-purpose CPUs are fundamentally limited for sustained, low-latency edge inference. While quantization improves efficiency, it does not eliminate:

instruction overhead

cache contention

operator scheduling costs

CPU power inefficiency at scale

Dedicated accelerators invert this trade-off by making INT8 execution the native fast path, rather than an optimization layered on top of FP32 execution.

Transition to Next Phase

With CPU baselines fully characterized and quantization behavior understood, the bootcamp transitions to accelerator-based inference.

The next phase introduces:

ONNX export and graph-level deployment

Hailo model compilation and runtime execution

CPU vs NPU latency, throughput, and utilization comparisons

Real-time inference pipelines decoupled from Python execution

Week 05 therefore serves as the quantitative and conceptual foundation for Hailo-based deployment in the following weeks.