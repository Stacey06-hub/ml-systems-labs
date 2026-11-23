
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "lab03" / "src"))
sys.path.insert(0, str(ROOT / "lab07" / "src"))

from data import load_data
from models import OversizedCNN


# Distillation settings
TEMPERATURE = 4.0
ALPHA = 0.5
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")


class StudentCNN(nn.Module):
    """Compact CNN student model."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 48),
            nn.ReLU(),
            nn.Linear(48, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_student(model, train_loader, distilled=False, teacher=None):
    """Train a student normally or using knowledge distillation."""

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if teacher is not None:
        teacher.eval()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            student_outputs = model(images)

            hard_loss = criterion(student_outputs, labels)

            if distilled and teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(images)

                soft_targets = F.softmax(
                    teacher_outputs / TEMPERATURE,
                    dim=1
                )

                soft_student = F.log_softmax(
                    student_outputs / TEMPERATURE,
                    dim=1
                )

                soft_loss = F.kl_div(
                    soft_student,
                    soft_targets,
                    reduction="batchmean"
                ) * (TEMPERATURE ** 2)

                loss = (
                    ALPHA * soft_loss
                    + (1 - ALPHA) * hard_loss
                )
            else:
                loss = hard_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predictions = student_outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = 100.0 * correct / total
        average_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Training accuracy: {accuracy:.2f}%"
        )


def evaluate(model, test_loader):
    """Calculate test accuracy."""

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def main():
    torch.manual_seed(42)

    print("Loading Fashion-MNIST...")
    train_loader, val_loader, test_loader = load_data()

    checkpoint_dir = ROOT / "lab07" / "checkpoints"
    results_dir = ROOT / "lab07" / "results"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained teacher
    print("\nLoading teacher model...")

    teacher = OversizedCNN().to(DEVICE)

    teacher_path = checkpoint_dir / "baseline_cnn.pt"

    teacher.load_state_dict(
        torch.load(
            teacher_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    teacher.eval()

    teacher_parameters = count_parameters(teacher)

    print(f"Teacher parameters: {teacher_parameters:,}")
    print(f"Teacher test accuracy: {evaluate(teacher, test_loader):.2f}%")

    # Train standalone student
    print("\nTRAINING STANDALONE STUDENT")
    torch.manual_seed(42)

    standalone_student = StudentCNN().to(DEVICE)

    print(
        f"Student parameters: "
        f"{count_parameters(standalone_student):,}"
    )

    train_student(
        standalone_student,
        train_loader
    )

    standalone_accuracy = evaluate(
        standalone_student,
        test_loader
    )

    standalone_path = checkpoint_dir / "student_standalone.pt"

    torch.save(
        standalone_student.state_dict(),
        standalone_path
    )

    print(
        f"Standalone student test accuracy: "
        f"{standalone_accuracy:.2f}%"
    )

    # Train distilled student
    print("\nTRAINING DISTILLED STUDENT")
    torch.manual_seed(42)

    distilled_student = StudentCNN().to(DEVICE)

    train_student(
        distilled_student,
        train_loader,
        distilled=True,
        teacher=teacher
    )

    distilled_accuracy = evaluate(
        distilled_student,
        test_loader
    )

    distilled_path = checkpoint_dir / "student_distilled.pt"

    torch.save(
        distilled_student.state_dict(),
        distilled_path
    )

    print(
        f"Distilled student test accuracy: "
        f"{distilled_accuracy:.2f}%"
    )

    # Calculate model sizes
    teacher_size = teacher_path.stat().st_size / (1024 * 1024)
    standalone_size = standalone_path.stat().st_size / (1024 * 1024)
    distilled_size = distilled_path.stat().st_size / (1024 * 1024)

    student_parameters = count_parameters(standalone_student)

    # Display comparison
    print("\nKNOWLEDGE DISTILLATION RESULTS")
    print("--------------------------------")
    print(f"Teacher parameters: {teacher_parameters:,}")
    print(f"Student parameters: {student_parameters:,}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Alpha: {ALPHA}")

    print("\nModel comparison:")
    print(
        f"Teacher: {teacher_size:.2f} MB | "
        f"Accuracy: {evaluate(teacher, test_loader):.2f}%"
    )
    print(
        f"Standalone student: {standalone_size:.2f} MB | "
        f"Accuracy: {standalone_accuracy:.2f}%"
    )
    print(
        f"Distilled student: {distilled_size:.2f} MB | "
        f"Accuracy: {distilled_accuracy:.2f}%"
    )

    # Save results
    results_file = results_dir / "distillation_results.txt"

    with open(results_file, "w") as file:
        file.write("Knowledge Distillation Results\n")
        file.write("------------------------------\n")
        file.write(f"Temperature: {TEMPERATURE}\n")
        file.write(f"Alpha: {ALPHA}\n")
        file.write(f"Training epochs: {EPOCHS}\n\n")

        file.write(f"Teacher parameters: {teacher_parameters:,}\n")
        file.write(f"Student parameters: {student_parameters:,}\n\n")

        file.write(
            f"Teacher accuracy: "
            f"{evaluate(teacher, test_loader):.2f}%\n"
        )
        file.write(
            f"Standalone student accuracy: "
            f"{standalone_accuracy:.2f}%\n"
        )
        file.write(
            f"Distilled student accuracy: "
            f"{distilled_accuracy:.2f}%\n\n"
        )

        file.write(f"Teacher size: {teacher_size:.2f} MB\n")
        file.write(f"Standalone student size: {standalone_size:.2f} MB\n")
        file.write(f"Distilled student size: {distilled_size:.2f} MB\n")

    print(f"\nResults saved to: {results_file}")
    print("Knowledge distillation completed.")


if __name__ == "__main__":
    main()
