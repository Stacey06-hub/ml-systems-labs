
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(42)

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def train_numpy():
    W1 = np.random.randn(2, 8) * 0.1
    b1 = np.zeros((1, 8))
    W2 = np.random.randn(8, 1) * 0.1
    b2 = np.zeros((1, 1))
    lr = 0.1

    for _ in range(100):
        z1 = X_train @ W1 + b1
        h = sigmoid(z1)
        z2 = h @ W2 + b2
        pred = sigmoid(z2)

        dz2 = (pred - y_train.reshape(-1, 1)) / len(X_train)
        dW2 = h.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        dz1 = (dz2 @ W2.T) * h * (1 - h)
        dW1 = X_train.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return W1, W2

class MoonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Sigmoid(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_torch():
    model = MoonModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCELoss()

    x = torch.tensor(X_train, dtype=torch.float32)
    target = torch.tensor(
        y_train.reshape(-1, 1), dtype=torch.float32
    )

    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        optimizer.step()

    return model

numpy_times = []
torch_times = []

for _ in range(5):
    start = time.perf_counter()
    train_numpy()
    numpy_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    train_torch()
    torch_times.append(time.perf_counter() - start)

print("NumPy median time:", np.median(numpy_times))
print("PyTorch median time:", np.median(torch_times))
print("Runs: 5")
print("Epochs: 100")
print("Architecture: 2 -> 8 -> 1")
