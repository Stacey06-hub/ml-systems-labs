
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Shared initialization so both implementations start identically.
rng = np.random.default_rng(42)
initial_W1 = rng.normal(0, np.sqrt(1 / 2), (2, 8))
initial_b1 = np.zeros((1, 8))
initial_W2 = rng.normal(0, np.sqrt(1 / 8), (8, 1))
initial_b2 = np.zeros((1, 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def train_numpy():
    W1 = initial_W1.copy()
    b1 = initial_b1.copy()
    W2 = initial_W2.copy()
    b2 = initial_b2.copy()

    lr = 0.1
    targets = y_train.reshape(-1, 1)

    for _ in range(100):
        z1 = X_train @ W1 + b1
        h = sigmoid(z1)
        z2 = h @ W2 + b2
        pred = sigmoid(z2)

        dz2 = (pred - targets) / len(X_train)
        dW2 = h.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        dz1 = (dz2 @ W2.T) * h * (1 - h)
        dW1 = X_train.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    test_pred = sigmoid(sigmoid(X_test @ W1 + b1) @ W2 + b2)
    test_y = y_test.reshape(-1, 1)

    loss = -np.mean(
        test_y * np.log(test_pred + 1e-8)
        + (1 - test_y) * np.log(1 - test_pred + 1e-8)
    )
    accuracy = np.mean((test_pred >= 0.5) == test_y)

    return loss, accuracy


def train_torch():
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.Sigmoid(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(initial_W1.T, dtype=torch.float32))
        model[0].bias.copy_(torch.tensor(initial_b1.ravel(), dtype=torch.float32))
        model[2].weight.copy_(torch.tensor(initial_W2.T, dtype=torch.float32))
        model[2].bias.copy_(torch.tensor(initial_b2.ravel(), dtype=torch.float32))

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCELoss()

    x = torch.tensor(X_train, dtype=torch.float32)
    target = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32)
        test_y = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)
        predictions = model(test_x)
        test_loss = loss_fn(predictions, test_y).item()
        accuracy = ((predictions >= 0.5) == test_y).float().mean().item()

    return test_loss, accuracy


numpy_times = []
torch_times = []

for _ in range(5):
    start = time.perf_counter()
    numpy_result = train_numpy()
    numpy_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    torch_result = train_torch()
    torch_times.append(time.perf_counter() - start)

print("NUMPY")
print("Median training time:", np.median(numpy_times))
print("Test loss:", numpy_result[0])
print("Test accuracy:", numpy_result[1])

print("\nPYTORCH")
print("Median training time:", np.median(torch_times))
print("Test loss:", torch_result[0])
print("Test accuracy:", torch_result[1])

print("\nRuns: 5")
print("Epochs: 100")
print("Architecture: 2 -> 8 -> 1")
print("Hidden activation: Sigmoid")
print("Output activation: Sigmoid")
print("Initialization: Shared Xavier-style weights")

with open("lab05/results/training_results.txt", "w") as f:
    f.write("Lab 5 make_moons training results\n")
    f.write(f"NumPy median time: {np.median(numpy_times)}\n")
    f.write(f"NumPy test loss: {numpy_result[0]}\n")
    f.write(f"NumPy test accuracy: {numpy_result[1]}\n")
    f.write(f"PyTorch median time: {np.median(torch_times)}\n")
    f.write(f"PyTorch test loss: {torch_result[0]}\n")
    f.write(f"PyTorch test accuracy: {torch_result[1]}\n")
    f.write("Runs: 5\nEpochs: 100\n")
    f.write("Architecture: 2 -> 8 -> 1\n")
    f.write("Hidden activation: Sigmoid\n")
    f.write("Output activation: Sigmoid\n")
    f.write("Initialization: Shared Xavier-style weights\n")
