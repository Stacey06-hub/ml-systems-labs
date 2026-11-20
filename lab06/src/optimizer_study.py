
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab03" / "src"))

from models import CNN
from data import load_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    seed = 42
    epochs = 10
    learning_rate = 0.001

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_base, val_base, test_base = load_data()

    train_loader = DataLoader(
        train_base.dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        val_base.dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_base.dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    optimizer_names = ["Adam", "SGD"]
    scheduler_names = [
        "StepLR",
        "CosineAnnealingLR",
        "ReduceLROnPlateau"
    ]

    results = []

    for optimizer_name in optimizer_names:
        for scheduler_name in scheduler_names:
            print(
                f"\nRunning {optimizer_name} + {scheduler_name}"
            )

            set_seed(seed)
            model = CNN().to(device)
            criterion = nn.CrossEntropyLoss()

            if optimizer_name == "Adam":
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate
                )
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=learning_rate,
                    momentum=0.9
                )

            if scheduler_name == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=3,
                    gamma=0.5
                )
            elif scheduler_name == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=2
                )

            start_time = time.perf_counter()
            history = []

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)

                train_loss = running_loss / len(train_loader.dataset)

                val_accuracy = evaluate(
                    model, val_loader, device
                )

                history.append(val_accuracy)

                if scheduler_name == "ReduceLROnPlateau":
                    scheduler.step(train_loss)
                else:
                    scheduler.step()

                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- Loss: {train_loss:.4f} "
                    f"- Validation accuracy: {val_accuracy:.4f} "
                    f"- LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

            elapsed = time.perf_counter() - start_time
            test_accuracy = evaluate(model, test_loader, device)

            results.append({
                "optimizer": optimizer_name,
                "scheduler": scheduler_name,
                "test_accuracy": test_accuracy,
                "time": elapsed,
                "history": history
            })

            print(
                f"Finished {optimizer_name} + {scheduler_name} "
                f"- Test accuracy: {test_accuracy:.4f}"
            )

    results_dir = ROOT / "lab06" / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "optimizer_study.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("Lab 6 - Optimizer and Scheduler Study\n\n")
        file.write("Training budget: 10 epochs\n\n")
        file.write(
            "Optimizer | Scheduler | Test accuracy | Time (seconds)\n"
        )

        for result in results:
            file.write(
                f"{result['optimizer']} | "
                f"{result['scheduler']} | "
                f"{result['test_accuracy']:.4f} | "
                f"{result['time']:.2f}\n"
            )

    plt.figure(figsize=(10, 6))

    for result in results:
        label = (
            result["optimizer"] + " + " + result["scheduler"]
        )
        plt.plot(
            range(1, epochs + 1),
            result["history"],
            marker="o",
            label=label
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Optimizer and Learning Rate Scheduler Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "optimizer_scheduler_comparison.png")
    plt.close()

    print("\nResults saved to:", report_path)
    print(
        "Plot saved to:",
        plots_dir / "optimizer_scheduler_comparison.png"
    )


if __name__ == "__main__":
    main()
