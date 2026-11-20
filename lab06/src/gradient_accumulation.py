
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lab03.src.models import CNN
from lab06.src.train import load_config, load_data, set_seed


def run_experiment(
    batch_size,
    accumulation_steps,
    config,
    train_dataset,
    test_loader
):
    set_seed(config["training"]["seed"])

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = CNN()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(5):
        model.train()
        total_loss = 0.0
        processed = 0
        optimizer.zero_grad()

        usable_batches = (
            (len(loader) // accumulation_steps) * accumulation_steps
        )

        for batch_index, (images, labels) in enumerate(loader):
            if batch_index >= usable_batches:
                break

            outputs = model(images)
            loss = criterion(outputs, labels)

            (loss / accumulation_steps).backward()

            if (batch_index + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * images.size(0)
            processed += images.size(0)

        average_loss = total_loss / processed
        history.append(average_loss)

        print(
            f"Epoch {epoch + 1}/5 - "
            f"Loss: {average_loss:.4f}"
        )

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print(f"Test accuracy: {accuracy:.4f}")

    return history, accuracy


def main():
    config = load_config()

    # load_data returns three values, not four.
    train_loader, _, test_loader = load_data(config)

    # Get the training dataset from the training loader.
    train_dataset = train_loader.dataset

    print("\nStandard batch size 256")

    standard_history, standard_accuracy = run_experiment(
        256,
        1,
        config,
        train_dataset,
        test_loader
    )

    print("\nBatch size 64 with accumulation 4")

    accumulated_history, accumulated_accuracy = run_experiment(
        64,
        4,
        config,
        train_dataset,
        test_loader
    )

    results_dir = ROOT / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "gradient_accumulation.txt"

    with open(results_path, "w", encoding="utf-8") as file:
        file.write("Standard batch size 256\n")
        file.write(f"Test accuracy: {standard_accuracy:.4f}\n")
        file.write(f"Epoch losses: {standard_history}\n\n")

        file.write("Batch size 64, accumulation steps 4\n")
        file.write(f"Test accuracy: {accumulated_accuracy:.4f}\n")
        file.write(f"Epoch losses: {accumulated_history}\n")

    print("\nResults saved to lab06/results/gradient_accumulation.txt")


if __name__ == "__main__":
    main()
