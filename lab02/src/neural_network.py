
import random
import platform
import time
import statistics
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Reproducibility and paths
# --------------------------------------------------
random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Dataset
# --------------------------------------------------
def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target
    Y = np.eye(10, dtype=np.float64)[y]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, Y_train, Y_test


# --------------------------------------------------
# Network initialization and activations
# --------------------------------------------------
def initialize_parameters(width, seed=42):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 0.01, (64, width)),
        "b1": np.zeros(width),
        "W2": rng.normal(0, 0.01, (width, width)),
        "b2": np.zeros(width),
        "W3": rng.normal(0, 0.01, (width, 10)),
        "b3": np.zeros(10),
    }


def relu(x):
    return np.maximum(0.0, x)


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# --------------------------------------------------
# Forward propagation and loss
# --------------------------------------------------
def forward_pass(X, params):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)

    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)

    z3 = a2 @ params["W3"] + params["b3"]
    probabilities = softmax(z3)

    cache = {
        "X": X,
        "z1": z1, "a1": a1,
        "z2": z2, "a2": a2,
        "probabilities": probabilities,
    }
    return probabilities, cache


def cross_entropy_loss(probabilities, Y):
    eps = 1e-12
    probabilities = np.clip(probabilities, eps, 1.0)
    return -np.mean(np.sum(Y * np.log(probabilities), axis=1))


# --------------------------------------------------
# Backpropagation
# --------------------------------------------------
def backward_pass(Y, params, cache):
    X = cache["X"]
    a1 = cache["a1"]
    a2 = cache["a2"]
    z1 = cache["z1"]
    z2 = cache["z2"]
    probabilities = cache["probabilities"]

    n = X.shape[0]

    dz3 = (probabilities - Y) / n
    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0)

    da2 = dz3 @ params["W3"].T
    dz2 = da2 * (z2 > 0)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {
        "W1": dW1, "b1": db1,
        "W2": dW2, "b2": db2,
        "W3": dW3, "b3": db3,
    }


# --------------------------------------------------
# Numerical gradient check
# --------------------------------------------------
def gradient_check():
    print("\n=== Numerical gradient check ===")

    rng = np.random.default_rng(123)
    X = rng.random((8, 64))
    Y = np.eye(10)[rng.integers(0, 10, size=8)]
    params = initialize_parameters(16, seed=7)

    probabilities, cache = forward_pass(X, params)
    analytical = backward_pass(Y, params, cache)

    eps = 1e-5
    checks = []

    # Check 10 randomly selected weights across weight matrices.
    candidates = [
        (name, idx)
        for name in ("W1", "W2", "W3")
        for idx in np.ndindex(params[name].shape)
    ]
    selected = rng.choice(len(candidates), size=10, replace=False)

    for selected_index in selected:
        name, idx = candidates[selected_index]
        original = params[name][idx]

        params[name][idx] = original + eps
        plus_p, _ = forward_pass(X, params)
        plus_loss = cross_entropy_loss(plus_p, Y)

        params[name][idx] = original - eps
        minus_p, _ = forward_pass(X, params)
        minus_loss = cross_entropy_loss(minus_p, Y)

        params[name][idx] = original

        numerical = (plus_loss - minus_loss) / (2 * eps)
        analytic = analytical[name][idx]
        denominator = max(1e-12, abs(numerical) + abs(analytic))
        relative_error = abs(numerical - analytic) / denominator

        checks.append(relative_error)
        print(
            f"{name}{idx}: analytical={analytic:.8e}, "
            f"numerical={numerical:.8e}, "
            f"relative error={relative_error:.3e}"
        )

    passed = sum(error < 1e-6 for error in checks)
    print(f"Gradient check: {passed}/10 passed (need at least 9)")
    return passed >= 9


# --------------------------------------------------
# Training and evaluation
# --------------------------------------------------
def train_model(X, Y, width, learning_rate, epochs=50, batch_size=32):
    params = initialize_parameters(width, seed=42)
    rng = np.random.default_rng(42)
    losses = []

    for epoch in range(epochs):
        indices = rng.permutation(len(X))
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        for start in range(0, len(X), batch_size):
            xb = X_shuffled[start:start + batch_size]
            yb = Y_shuffled[start:start + batch_size]

            probabilities, cache = forward_pass(xb, params)
            gradients = backward_pass(yb, params, cache)

            for name in params:
                params[name] -= learning_rate * gradients[name]

        probabilities, _ = forward_pass(X, params)
        epoch_loss = cross_entropy_loss(probabilities, Y)
        losses.append(epoch_loss)

        print(
            f"width={width:2d}, lr={learning_rate:g}, "
            f"epoch={epoch + 1:02d}/{epochs}, loss={epoch_loss:.5f}"
        )

    return params, losses


def accuracy(X, Y, params):
    probabilities, _ = forward_pass(X, params)
    predicted = np.argmax(probabilities, axis=1)
    actual = np.argmax(Y, axis=1)
    return np.mean(predicted == actual)


# --------------------------------------------------
# Timing
# --------------------------------------------------
def time_forward_backward(X, Y, params, repeats=100):
    # Warm-up
    probabilities, cache = forward_pass(X, params)
    backward_pass(Y, params, cache)

    forward_times = []
    backward_times = []

    for _ in range(repeats):
        start = time.perf_counter()
        probabilities, cache = forward_pass(X, params)
        forward_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        backward_pass(Y, params, cache)
        backward_times.append(time.perf_counter() - start)

    return (
        statistics.median(forward_times) * 1000,
        statistics.median(backward_times) * 1000,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    print("Python:", platform.python_version())
    print("NumPy:", np.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("scikit-learn:", sklearn.__version__)

    X_train, X_test, Y_train, Y_test = load_data()
    print("Training shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # Forward-pass sanity checks
    for width in (16, 64):
        params = initialize_parameters(width)
        probabilities, _ = forward_pass(X_train[:32], params)
        loss = cross_entropy_loss(probabilities, Y_train[:32])
        print(
            f"Width {width}: output={probabilities.shape}, "
            f"row sum={probabilities[0].sum():.12f}, "
            f"initial loss={loss:.6f}"
        )

    if not gradient_check():
        raise RuntimeError("Gradient check failed; fix gradients before training.")

    # Run every required width / learning-rate combination
    learning_rates = (0.001, 0.01, 0.1)
    widths = (16, 64)
    all_losses = {}
    rows = []

    for width in widths:
        for lr in learning_rates:
            print(f"\nTraining width={width}, learning rate={lr}")
            params, losses = train_model(
                X_train, Y_train, width, lr,
                epochs=50, batch_size=32,
            )

            test_accuracy = accuracy(X_test, Y_test, params)
            forward_ms, backward_ms = time_forward_backward(
                X_train[:32], Y_train[:32], params, repeats=100
            )

            key = f"W={width}, lr={lr}"
            all_losses[key] = losses
            rows.append({
                "width": width,
                "learning_rate": lr,
                "final_train_loss": losses[-1],
                "test_accuracy": test_accuracy,
                "forward_median_ms": forward_ms,
                "backward_median_ms": backward_ms,
            })

            print(f"Test accuracy: {test_accuracy:.4f}")
            print(f"Forward median: {forward_ms:.6f} ms")
            print(f"Backward median: {backward_ms:.6f} ms")

    # Save loss curves
    plt.figure(figsize=(10, 6))
    for label, losses in all_losses.items():
        plt.plot(range(1, len(losses) + 1), losses, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("Digits Neural Network: Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
    plt.close()

    # Save numerical results
    import csv

    csv_path = RESULTS_DIR / "training_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Final results ===")
    for row in rows:
        print(row)

    print("\nSaved loss curves to:", RESULTS_DIR / "loss_curves.png")
    print("Saved results to:", csv_path)
