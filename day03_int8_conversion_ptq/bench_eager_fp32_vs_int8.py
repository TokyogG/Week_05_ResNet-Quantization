import time, statistics, torch
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2

torch.set_num_threads(1)
torch.backends.quantized.engine = "qnnpack"

fp32 = q_mobilenet_v2(weights="DEFAULT", quantize=False).eval()
x = torch.randn(1,3,224,224)

# Prepare/convert PTQ (same as your export script but minimal)
fp32.fuse_model()
fp32.qconfig = torch.quantization.get_default_qconfig(torch.backends.quantized.engine)
prepared = torch.quantization.prepare(fp32, inplace=False)

with torch.inference_mode():
    for _ in range(50):
        prepared(x)

int8 = torch.quantization.convert(prepared, inplace=False).eval()

def bench(m, iters=200, warmup=20):
    with torch.inference_mode():
        for _ in range(warmup):
            _ = m(x)
        ts=[]
        for _ in range(iters):
            t0=time.perf_counter()
            _ = m(x)
            t1=time.perf_counter()
            ts.append((t1-t0)*1000)
    return statistics.mean(ts)

print("FP32 mean ms:", bench(fp32))
print("INT8 mean ms:", bench(int8))