
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lab03.src.models import CNN
from lab06.src.train import load_config, load_data, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interrupt-at-3", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config()
    set_seed(config["training"]["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    train_loader, _, _ = load_data(config)

    model = CNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = ROOT / "lab06" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "recovery_checkpoint.pt"
    results_dir = ROOT / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    history = []
    start_epoch = 0
    total_epochs = 5

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "No recovery checkpoint found. Run --interrupt-at-3 first."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]

        print(f"Resuming from epoch {start_epoch + 1}")

    elif args.interrupt_at_3:
        print("Starting training. It will stop after epoch 3.")

    else:
        print("Choose --interrupt-at-3 or --resume.")
        return

    end_epoch = 3 if args.interrupt_at_3 else total_epochs

    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        average_loss = total_loss / total_samples
        history.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{total_epochs} "
            f"- Training loss: {average_loss:.4f}"
        )

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }, checkpoint_path)

    if args.interrupt_at_3:
        print("\nTraining interrupted after epoch 3.")
        print("Checkpoint saved:", checkpoint_path)
        print("Run with --resume to continue.")
        return

    print("\nTraining completed after resuming.")

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(history) + 1),
        history,
        marker="o",
        label="Training loss"
    )
    plt.axvline(
        x=3,
        linestyle="--",
        label="Resume point (after epoch 3)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Checkpoint Recovery Training")
    plt.xticks(range(1, total_epochs + 1))
    plt.legend()
    plt.grid(True)

    plot_path = results_dir / "checkpoint_recovery.png"
    plt.savefig(plot_path)
    plt.close()

    with open(
        results_dir / "checkpoint_recovery.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write("Checkpoint recovery results\n")
        file.write(f"Epoch losses: {history}\n")
        file.write(f"Resume point: after epoch 3\n")

    print("Loss plot saved:", plot_path)
    print("Results saved:", results_dir / "checkpoint_recovery.txt")


if __name__ == "__main__":
    main()
