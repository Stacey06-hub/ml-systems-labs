
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab07" / "src"))

from models import OversizedCNN


PRUNING_LEVELS = [0.10, 0.25, 0.50]


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def prune_filters(model, amount):
    """Remove the lowest-magnitude output filters from each convolution."""

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=1,
                dim=0
            )

            # Make the pruning permanent
            prune.remove(module, "weight")


def count_zero_filters(model):
    """Count convolutional filters that contain only zero weights."""

    total_filters = 0
    zero_filters = 0

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            for filter_weights in module.weight:
                total_filters += 1

                if torch.count_nonzero(filter_weights).item() == 0:
                    zero_filters += 1

    return total_filters, zero_filters


def main():
    torch.manual_seed(42)

    results_dir = ROOT / "lab07" / "results"
    checkpoints_dir = ROOT / "lab07" / "checkpoints"

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "structured_pruning_results.txt"

    print("STRUCTURED PRUNING EXPERIMENT")
    print("-----------------------------")

    with open(results_file, "w") as file:
        file.write("Structured Pruning Results\n")
        file.write("--------------------------\n")

        for amount in PRUNING_LEVELS:
            model = OversizedCNN()

            original_parameters = count_parameters(model)

            prune_filters(model, amount)

            total_filters, zero_filters = count_zero_filters(model)

            remaining_parameters = count_parameters(model)

            result = (
                f"Target filter pruning: {amount * 100:.0f}% | "
                f"Zero filters: {zero_filters}/{total_filters} | "
                f"Parameters: {original_parameters:,} | "
                f"Parameters after masking: {remaining_parameters:,}"
            )

            print(result)
            file.write(result + "\n")

            checkpoint_path = (
                checkpoints_dir /
                f"structured_pruned_{int(amount * 100)}.pt"
            )

            torch.save(model.state_dict(), checkpoint_path)

    print("\nStructured pruning completed.")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
