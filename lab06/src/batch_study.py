
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lab03.src.models import CNN
from lab06.src.train import load_config, load_data, set_seed


def measure_peak_ram(stop_event, peak_memory):
    process = psutil.Process()

    while not stop_event.is_set():
        memory_mb = process.memory_info().rss / (1024 ** 2)
        peak_memory[0] = max(peak_memory[0], memory_mb)
        time.sleep(0.01)


def run_experiment(batch_size, config, train_dataset, test_loader):
    set_seed(config["training"]["seed"])

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    model = CNN()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    peak_memory = [psutil.Process().memory_info().rss / (1024 ** 2)]
    stop_event = threading.Event()

    monitor = threading.Thread(
        target=measure_peak_ram,
        args=(stop_event, peak_memory)
    )

    start_time = time.perf_counter()
    monitor.start()

    for epoch in range(5):
        model.train()

        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    stop_event.set()
    monitor.join()

    elapsed = time.perf_counter() - start_time
    throughput = (len(train_dataset) * 5) / elapsed

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return throughput, peak_memory[0], accuracy


def main():
    config = load_config()
    set_seed(config["training"]["seed"])

    train_loader, _, test_loader = load_data(config)
    train_dataset = train_loader.dataset

    batch_sizes = [32, 64, 128, 256, 512]

    results = []

    for batch_size in batch_sizes:
        print(f"\nTesting batch size {batch_size}")

        throughput, peak_ram, accuracy = run_experiment(
            batch_size,
            config,
            train_dataset,
            test_loader
        )

        results.append((batch_size, throughput, peak_ram, accuracy))

        print(f"Throughput: {throughput:.2f} samples/second")
        print(f"Peak RAM: {peak_ram:.2f} MB")
        print(f"Test accuracy: {accuracy:.4f}")

    results_dir = ROOT / "lab06" / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(
        results_dir / "batch_study.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write("Batch Size Study (5 epochs)\n")
        file.write("Peak RAM is process RSS, measured in MB.\n\n")

        for batch_size, throughput, peak_ram, accuracy in results:
            file.write(
                f"Batch size: {batch_size}, "
                f"Throughput: {throughput:.2f} samples/s, "
                f"Peak RAM: {peak_ram:.2f} MB, "
                f"Test accuracy: {accuracy:.4f}\n"
            )

    sizes = [r[0] for r in results]
    throughputs = [r[1] for r in results]
    memories = [r[2] for r in results]
    accuracies = [r[3] for r in results]

    plt.figure()
    plt.plot(sizes, throughputs, marker="o")
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (samples/second)")
    plt.title("Batch Size vs Throughput")
    plt.savefig(plots_dir / "batch_throughput.png")
    plt.close()

    plt.figure()
    plt.plot(sizes, memories, marker="o")
    plt.xlabel("Batch size")
    plt.ylabel("Peak process RAM (MB)")
    plt.title("Batch Size vs Peak RAM")
    plt.savefig(plots_dir / "batch_memory.png")
    plt.close()

    plt.figure()
    plt.plot(sizes, accuracies, marker="o")
    plt.xlabel("Batch size")
    plt.ylabel("Test accuracy")
    plt.title("Batch Size vs Accuracy")
    plt.savefig(plots_dir / "batch_accuracy.png")
    plt.close()

    print("\nResults saved to lab06/results/batch_study.txt")


if __name__ == "__main__":
    main()
