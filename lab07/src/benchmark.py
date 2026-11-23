
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab07" / "src"))

from models import OversizedCNN


DEVICE = torch.device("cpu")
WARMUP_RUNS = 20
BENCHMARK_RUNS = 200


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def calculate_sparsity(model):
    total_weights = 0
    zero_weights = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights = module.weight
            total_weights += weights.numel()
            zero_weights += torch.count_nonzero(weights == 0).item()

    return 100 * zero_weights / total_weights


def measure_latency(model):
    model.eval()
    sample = torch.randn(1, 1, 28, 28, device=DEVICE)

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model(sample)

        start = time.perf_counter()

        for _ in range(BENCHMARK_RUNS):
            model(sample)

        end = time.perf_counter()

    return (end - start) * 1000 / BENCHMARK_RUNS


def measure_model(model_path):
    model = OversizedCNN().to(DEVICE)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    parameters = count_parameters(model)
    sparsity = calculate_sparsity(model)
    latency = measure_latency(model)

    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    return {
        "parameters": parameters,
        "sparsity": sparsity,
        "size_mb": model_size_mb,
        "latency_ms": latency
    }


def main():
    checkpoints = ROOT / "lab07" / "checkpoints"
    results_dir = ROOT / "lab07" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("Baseline dense", "baseline_cnn.pt")
    ]

    for method in ["global", "per_layer"]:
        for percentage in [25, 50, 75, 90, 95]:
            experiments.append((
                f"{method} {percentage}%",
                f"{method}_pruned_{percentage}.pt"
            ))

    results = []

    print("DENSE VS SPARSE MODEL BENCHMARK")
    print("--------------------------------")

    for label, filename in experiments:
        model_path = checkpoints / filename

        if not model_path.exists():
            print(f"Skipping missing checkpoint: {filename}")
            continue

        metrics = measure_model(model_path)
        results.append((label, metrics))

        print(
            f"{label:20} | "
            f"Parameters: {metrics['parameters']:,} | "
            f"Sparsity: {metrics['sparsity']:.2f}% | "
            f"Size: {metrics['size_mb']:.2f} MB | "
            f"Latency: {metrics['latency_ms']:.3f} ms"
        )

    results_file = results_dir / "benchmark_results.txt"

    with open(results_file, "w") as file:
        file.write("Dense vs Sparse Model Benchmark\n")
        file.write("--------------------------------\n")
        file.write(
            "Model | Parameters | Sparsity (%) | "
            "Size (MB) | Latency (ms)\n"
        )

        for label, metrics in results:
            file.write(
                f"{label} | "
                f"{metrics['parameters']} | "
                f"{metrics['sparsity']:.2f} | "
                f"{metrics['size_mb']:.2f} | "
                f"{metrics['latency_ms']:.3f}\n"
            )

    print("\nBenchmark completed.")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
