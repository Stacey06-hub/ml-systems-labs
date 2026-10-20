import torch
import torch.nn as nn
from models import MLP, CNN, AttentionModel
from data import load_data

DEVICE = torch.device("cpu")
EPOCHS = 5
LR = 1e-3


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_model(name, model, train_loader, val_loader, test_loader):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_accuracy = evaluate(model, val_loader)
        print(
            f"{name} | Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss {total_loss / len(train_loader):.4f} "
            f"| Validation accuracy {val_accuracy:.4f}"
        )

    test_accuracy = evaluate(model, test_loader)
    print(f"{name} | Test accuracy: {test_accuracy:.4f}\n")
    return test_accuracy


def main():
    train_loader, val_loader, test_loader = load_data()

    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}")
        results[name] = train_model(
            name, model, train_loader, val_loader, test_loader
        )

    print("\nFinal test accuracy:")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.4f}")


if __name__ == "__main__":
    main()