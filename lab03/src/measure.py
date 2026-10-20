import sys
import time
import torch
from torchinfo import summary
from thop import profile

sys.path.insert(0, "lab03/src")

from models import MLP, CNN, AttentionModel

torch.set_num_threads(1)
device = torch.device("cpu")

models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel(),
}

def measure_latency(model, batch_size, runs):
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28)

    with torch.no_grad():
        for _ in range(5):
            model(x)

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return times[len(times) // 2]


for name, model in models.items():
    print(f"\n===== {name} =====")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    summary(model, input_size=(1, 1, 28, 28), verbose=0)

    example = torch.randn(1, 1, 28, 28)
    macs, _ = profile(model, inputs=(example,), verbose=False)
    print(f"THOP MACs: {macs:,}")
    print(f"Approximate FLOPs (2 x MACs): {2 * macs:,}")

    for batch_size, runs in [(1, 100), (128, 20)]:
        latency = measure_latency(model, batch_size, runs)
        print(
            f"Batch {batch_size} median latency: "
            f"{latency:.3f} ms"
        )