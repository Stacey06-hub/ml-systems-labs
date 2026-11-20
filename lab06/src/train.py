
import os
import random
import yaml
import mlflow
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from pathlib import Path
import sys

# Use the original CNN from Lab 3
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lab03" / "src"))
from models import CNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (config["data"]["normalization_mean"],),
            (config["data"]["normalization_std"],)
        )
    ])

    train_full = datasets.FashionMNIST(
        root=config["data"]["directory"],
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root=config["data"]["directory"],
        train=False,
        download=True,
        transform=transform
    )

    train_data, val_data = random_split(
        train_full,
        [
            config["data"]["train_size"],
            config["data"]["validation_size"]
        ],
        generator=torch.Generator().manual_seed(config["training"]["seed"])
    )

    batch_size = config["training"]["batch_size"]
    workers = config["training"]["num_workers"]

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )

    return train_loader, val_loader, test_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    config = load_config()

    seed = config["training"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = load_data(config)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    checkpoint_dir = Path(config["checkpoint"]["directory"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / config["checkpoint"]["filename"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    best_accuracy = 0.0

    with mlflow.start_run():
        mlflow.log_params({
            "model": config["model"]["name"],
            "seed": seed,
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "optimizer": config["training"]["optimizer"],
            "num_workers": config["training"]["num_workers"]
        })

        for epoch in range(config["training"]["epochs"]):
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

            val_loss, val_accuracy = evaluate(
                model, val_loader, criterion, device
            )

            print(
                f"Epoch {epoch + 1}/{config['training']['epochs']} "
                f"- Train loss: {train_loss:.4f} "
                f"- Validation loss: {val_loss:.4f} "
                f"- Validation accuracy: {val_accuracy:.4f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
            mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
            mlflow.log_metric(
                "val_accuracy", val_accuracy, step=epoch + 1
            )

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_accuracy": best_accuracy,
                    "config": config
                }, checkpoint_path)

        # Test the best saved model
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, device
        )

        print(f"\nTest loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print("Best checkpoint saved at:", checkpoint_path)

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("best_val_accuracy", best_accuracy)

        mlflow.log_artifact(str(checkpoint_path))


if __name__ == "__main__":
    main()
