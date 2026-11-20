
import sys
import time
import yaml
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader

# Reuse the original Lab 3 model and Lab 6 data loader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab03" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import CNN
from train import load_config, load_data, set_seed


def main():
    config = load_config()
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _ = load_data(config)

    model = CNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    # Warm-up before collecting the measurements
    model.train()
    warmup_loader = iter(train_loader)

    for _ in range(3):
        images, labels = next(warmup_loader)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    # Profile one epoch
    results = {
        "data_loading": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0
    }

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU
        ],
        record_shapes=True,
        profile_memory=True
    )

    profiler.start()

    data_start = time.perf_counter()

    for images, labels in train_loader:
        data_end = time.perf_counter()
        results["data_loading"] += data_end - data_start

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        forward_start = time.perf_counter()
        outputs = model(images)
        loss = criterion(outputs, labels)
        forward_end = time.perf_counter()

        loss.backward()
        backward_end = time.perf_counter()

        optimizer.step()
        optimizer_end = time.perf_counter()

        results["forward"] += forward_end - forward_start
        results["backward"] += backward_end - forward_end
        results["optimizer"] += optimizer_end - backward_end

        data_start = time.perf_counter()

    profiler.stop()

    total_time = sum(results.values())

    output_dir = ROOT / "lab06" / "results" / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "profile_breakdown.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("Lab 6 - Training Profile\n\n")
        file.write("Time breakdown for one training epoch\n\n")

        for name, duration in results.items():
            percentage = duration / total_time * 100
            line = (
                f"{name}: {duration:.4f} seconds "
                f"({percentage:.2f}%)"
            )
            print(line)
            file.write(line + "\n")

        file.write(f"\nTotal measured time: {total_time:.4f} seconds\n")

    # Save the detailed PyTorch profiler output
    profiler_path = output_dir / "profiler_trace.json"
    profiler.export_chrome_trace(str(profiler_path))

    print("\nProfile report saved to:", report_path)
    print("Profiler trace saved to:", profiler_path)


if __name__ == "__main__":
    main()
