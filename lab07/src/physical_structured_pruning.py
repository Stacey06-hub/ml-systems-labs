import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "lab03" / "src"))
sys.path.insert(0, str(ROOT / "lab07" / "src"))

from data import load_data
from models import OversizedCNN


DEVICE = torch.device("cpu")

CHECKPOINT = ROOT / "lab07" / "checkpoints" / "baseline_cnn.pt"
OUTPUT_DIR = ROOT / "lab07" / "checkpoints"
RESULTS_FILE = (
    ROOT / "lab07" / "results"
    / "physical_structured_pruning_results.txt"
)


class CompactCNN(nn.Module):

    def __init__(self, channels):
        super().__init__()

        c1, c2, c3 = channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 7 * 7, 192),
            nn.ReLU(),
            nn.Linear(192, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_baseline():

    model = OversizedCNN().to(DEVICE)

    state = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model


def select_filters(weight, removal_rate):

    scores = weight.abs().sum(dim=(1, 2, 3))

    total = weight.shape[0]

    keep_count = max(
        1,
        round(total * (1 - removal_rate))
    )

    indices = torch.argsort(
        scores,
        descending=True
    )[:keep_count]

    return indices.sort().values


def copy_convolution(source, target, output_indices,
                     input_indices=None):

    with torch.no_grad():

        weights = source.weight.index_select(
            0, output_indices
        )

        if input_indices is not None:
            weights = weights.index_select(
                1, input_indices
            )

        target.weight.copy_(weights)

        if source.bias is not None:
            target.bias.copy_(
                source.bias.index_select(
                    0, output_indices
                )
            )


def build_compact_model(original, removal_rate):

    conv1 = original.features[0]
    conv2 = original.features[2]
    conv3 = original.features[5]

    # Select the filters with the largest L1 scores.
    keep1 = select_filters(
        conv1.weight, removal_rate
    )

    keep2 = select_filters(
        conv2.weight, removal_rate
    )

    keep3 = select_filters(
        conv3.weight, removal_rate
    )

    channels = (
        len(keep1),
        len(keep2),
        len(keep3)
    )

    compact = CompactCNN(channels).to(DEVICE)

    # Copy retained filters and corresponding input channels.
    copy_convolution(
        conv1,
        compact.features[0],
        keep1
    )

    copy_convolution(
        conv2,
        compact.features[2],
        keep2,
        keep1
    )

    copy_convolution(
        conv3,
        compact.features[5],
        keep3,
        keep2
    )

    # Select the fully connected input weights that remain.
    spatial_size = 7 * 7

    fc_indices = (
        keep3[:, None] * spatial_size
        + torch.arange(spatial_size)[None, :]
    ).reshape(-1)

    with torch.no_grad():

        compact.classifier[1].weight.copy_(
            original.classifier[1].weight.index_select(
                1, fc_indices
            )
        )

        compact.classifier[1].bias.copy_(
            original.classifier[1].bias
        )

        compact.classifier[3].weight.copy_(
            original.classifier[3].weight
        )

        compact.classifier[3].bias.copy_(
            original.classifier[3].bias
        )

    return compact, channels


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def evaluate(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100 * correct / total


def measure_latency(model):

    model.eval()

    sample = torch.randn(
        1, 1, 28, 28,
        device=DEVICE
    )

    with torch.no_grad():

        for _ in range(20):
            model(sample)

        start = time.perf_counter()

        for _ in range(200):
            model(sample)

        end = time.perf_counter()

    return (end - start) / 200 * 1000


def main():

    print("PHYSICAL STRUCTURED PRUNING")
    print("---------------------------")

    _, _, test_loader = load_data()

    original = load_baseline()

    baseline_parameters = count_parameters(original)

    baseline_accuracy = evaluate(
        original, test_loader
    )

    baseline_latency = measure_latency(original)

    baseline_size = CHECKPOINT.stat().st_size / (
        1024 * 1024
    )

    print(
        f"Baseline parameters: {baseline_parameters:,}"
    )

    print(
        f"Baseline accuracy: {baseline_accuracy:.2f}%"
    )

    print(
        f"Baseline size: {baseline_size:.2f} MB"
    )

    print(
        f"Baseline latency: {baseline_latency:.3f} ms"
    )

    results = [
        "PHYSICAL STRUCTURED PRUNING RESULTS",
        "-----------------------------------",
        f"Baseline parameters: {baseline_parameters}",
        f"Baseline accuracy: {baseline_accuracy:.2f}%",
        f"Baseline size: {baseline_size:.2f} MB",
        f"Baseline latency: {baseline_latency:.3f} ms",
        ""
    ]

    for rate in [0.10, 0.25, 0.50]:

        model, channels = build_compact_model(
            original, rate
        )

        parameters = count_parameters(model)

        accuracy = evaluate(
            model, test_loader
        )

        latency = measure_latency(model)

        filename = (
            f"physical_structured_{int(rate * 100)}.pt"
        )

        checkpoint_path = OUTPUT_DIR / filename

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        size_mb = checkpoint_path.stat().st_size / (
            1024 * 1024
        )

        reduction = (
            1 - parameters / baseline_parameters
        ) * 100

        print(f"\nFilter removal: {rate * 100:.0f}%")
        print(f"Remaining channels: {channels}")
        print(f"Parameters: {parameters:,}")
        print(f"Parameter reduction: {reduction:.2f}%")
        print(f"Model size: {size_mb:.2f} MB")
        print(f"Test accuracy: {accuracy:.2f}%")
        print(f"Inference latency: {latency:.3f} ms")

        results.extend([
            f"Filter removal: {rate * 100:.0f}%",
            f"Remaining channels: {channels}",
            f"Parameters: {parameters}",
            f"Parameter reduction: {reduction:.2f}%",
            f"Model size: {size_mb:.2f} MB",
            f"Test accuracy: {accuracy:.2f}%",
            f"Inference latency: {latency:.3f} ms",
            ""
        ])

    RESULTS_FILE.write_text(
        "\n".join(results),
        encoding="utf-8"
    )

    print("\nExperiment completed.")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()