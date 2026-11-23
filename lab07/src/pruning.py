
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

ROOT = Path(__file__).resolve().parents[2]
LAB03_SRC = ROOT / "lab03" / "src"
LAB07_SRC = ROOT / "lab07" / "src"

sys.path.insert(0, str(LAB03_SRC))
sys.path.insert(0, str(LAB07_SRC))

from data import load_data
from models import OversizedCNN


SPARSITY_LEVELS = [0, 0.25, 0.50, 0.75, 0.90, 0.95]
FINE_TUNE_EPOCHS = 3


def get_prunable_layers(model):
    return [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]


def calculate_sparsity(model):
    total = 0
    zeros = 0

    for module, name in get_prunable_layers(model):
        weights = getattr(module, name)
        total += weights.numel()
        zeros += torch.sum(weights == 0).item()

    return 100.0 * zeros / total if total else 0.0


def apply_pruning(model, amount, method):
    layers = get_prunable_layers(model)

    if amount == 0:
        return model

    if method == "global":
        prune.global_unstructured(
            layers,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
    else:
        for module, name in layers:
            prune.l1_unstructured(module, name=name, amount=amount)

    return model


def fine_tune(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(FINE_TUNE_EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        # Reapply masks after each epoch to guarantee zeros stay zero.
        for module, name in get_prunable_layers(model):
            if prune.is_pruned(module):
                module.weight.data.mul_(module.weight_mask)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_accuracy = 100.0 * correct / total

        print(
            f"    Fine-tune epoch {epoch + 1}/{FINE_TUNE_EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation accuracy: {val_accuracy:.2f}%"
        )

    return model


def make_pruning_permanent(model):
    for module, name in get_prunable_layers(model):
        if prune.is_pruned(module):
            prune.remove(module, name)

    return model


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    print("Loading Fashion-MNIST...")
    train_loader, val_loader, _ = load_data()

    results_dir = ROOT / "lab07" / "results"
    checkpoints_dir = ROOT / "lab07" / "checkpoints"

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "pruning_results.txt"

    with open(results_file, "w") as file:
        file.write("Magnitude Pruning and Fine-Tuning Results\n")
        file.write("-----------------------------------------\n")

        for method in ["global", "per_layer"]:
            print(f"\nPruning method: {method}")
            file.write(f"\nMethod: {method}\n")

            for amount in SPARSITY_LEVELS:
                print(f"\nTarget sparsity: {amount * 100:.0f}%")

                model = OversizedCNN().to(device)

                apply_pruning(model, amount, method)

                before_finetune = calculate_sparsity(model)
                print(f"Actual sparsity before fine-tuning: {before_finetune:.2f}%")

                model = fine_tune(
                    model,
                    train_loader,
                    val_loader,
                    device
                )

                after_finetune = calculate_sparsity(model)
                print(f"Actual sparsity after fine-tuning: {after_finetune:.2f}%")

                model = make_pruning_permanent(model)

                checkpoint_path = (
                    checkpoints_dir /
                    f"{method}_pruned_{int(amount * 100)}.pt"
                )

                torch.save(model.state_dict(), checkpoint_path)

                result = (
                    f"Target: {amount * 100:.0f}% | "
                    f"Before fine-tuning: {before_finetune:.2f}% | "
                    f"After fine-tuning: {after_finetune:.2f}%"
                )

                file.write(result + "\n")
                file.flush()

    print("\nPruning and fine-tuning experiments completed.")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
