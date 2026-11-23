
import torch
import torch.nn as nn


class OversizedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 192),
            nn.ReLU(),
            nn.Linear(192, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = OversizedCNN()
    parameters = sum(p.numel() for p in model.parameters())

    print("Model: OversizedCNN")
    print(f"Total parameters: {parameters:,}")
