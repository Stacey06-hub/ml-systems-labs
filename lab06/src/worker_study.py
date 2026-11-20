
import sys
import time
import yaml
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab03" / "src"))

from data import load_data


def main():
    config_path = ROOT / "lab06" / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Load the original Fashion-MNIST training dataset
    train_loader, _, _ = load_data()

    train_dataset = train_loader.dataset
    batch_size = config["training"]["batch_size"]

    worker_counts = [0, 1, 2, 4, 8]
    results = []

    for workers in worker_counts:
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers
        )

        # Warm-up
        iterator = iter(loader)
        for _ in range(3):
            next(iterator)

        start = time.perf_counter()
        samples = 0

        for images, labels in loader:
            samples += images.size(0)

        elapsed = time.perf_counter() - start
        throughput = samples / elapsed

        results.append({
            "num_workers": workers,
            "time": elapsed,
            "throughput": throughput
        })

        print(
            f"Workers: {workers} | "
            f"Time: {elapsed:.2f}s | "
            f"Throughput: {throughput:.2f} samples/s"
        )

    results_dir = ROOT / "lab06" / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save the results table
    report_path = results_dir / "worker_study.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("Lab 6 - Data Loading Worker Study\n\n")
        file.write("Workers | Time (s) | Throughput (samples/s)\n")

        for result in results:
            file.write(
                f"{result['num_workers']} | "
                f"{result['time']:.2f} | "
                f"{result['throughput']:.2f}\n"
            )

    # Plot throughput for each worker setting
    workers = [r["num_workers"] for r in results]
    throughputs = [r["throughput"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(workers, throughputs)
    plt.xlabel("Number of workers")
    plt.ylabel("Throughput (samples/second)")
    plt.title("Fashion-MNIST Data Loading Performance")
    plt.xticks(workers)
    plt.tight_layout()

    plot_path = plots_dir / "worker_throughput.png"
    plt.savefig(plot_path)
    plt.close()

    print("\nResults saved to:", report_path)
    print("Plot saved to:", plot_path)


if __name__ == "__main__":
    main()
