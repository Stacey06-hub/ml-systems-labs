
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Define project folders
ROOT = Path(__file__).resolve().parents[2]
LAB03_SRC = ROOT / "lab03" / "src"
LAB07_SRC = ROOT / "lab07" / "src"

# Add both folders to Python's import path.
# Lab 7 must come first so its models.py is imported.
sys.path.insert(0, str(LAB03_SRC))
sys.path.insert(0, str(LAB07_SRC))

from data import load_data
from models import OversizedCNN


def main():
    torch.manual_seed(42)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load Fashion-MNIST datasets
    train_loader, val_loader, test_loader = load_data()

    # Create the baseline CNN
    model = OversizedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    best_val_accuracy = 0.0
    history = []

    # Train the model
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        average_loss = total_loss / len(train_loader)

        # Validate the model
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        history.append({
            "epoch": epoch + 1,
            "loss": average_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            checkpoint_dir = ROOT / "lab07" / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                model.state_dict(),
                checkpoint_dir / "baseline_cnn.pt"
            )

    # Load the best checkpoint
    model_path = ROOT / "lab07" / "checkpoints" / "baseline_cnn.pt"

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    # Evaluate on the test dataset
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total

    # Measure single-image inference latency
    sample = torch.randn(1, 1, 28, 28).to(device)

    with torch.no_grad():
        for _ in range(10):
            model(sample)

        start_time = time.perf_counter()

        for _ in range(100):
            model(sample)

        end_time = time.perf_counter()

    latency_ms = ((end_time - start_time) / 100) * 1000

    # Calculate model metrics
    parameters = sum(p.numel() for p in model.parameters())
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    # Display results
    print("\nBASELINE RESULTS")
    print("-------------------------")
    print(f"Parameters: {parameters:,}")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Inference latency: {latency_ms:.3f} ms")

    # Save results
    results_dir = ROOT / "lab07" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "baseline_results.txt", "w") as file:
        file.write("Baseline CNN Results\n")
        file.write("-------------------------\n")
        file.write(f"Parameters: {parameters:,}\n")
        file.write(f"Test accuracy: {test_accuracy:.2f}%\n")
        file.write(f"Model size: {model_size_mb:.2f} MB\n")
        file.write(f"Inference latency: {latency_ms:.3f} ms\n\n")
        file.write("Training history:\n")

        for item in history:
            file.write(str(item) + "\n")


if __name__ == "__main__":
    main()
