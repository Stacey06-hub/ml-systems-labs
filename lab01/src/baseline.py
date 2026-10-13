import sys
import time
import random
import platform
import threading
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import joblib
import psutil
import torch

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "lab01" / "results"
MODELS_DIR = RESULTS_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def measure_peak_memory(function):
    """Return function result and approximate peak process memory in MB."""
    process = psutil.Process()
    baseline_mb = process.memory_info().rss / (1024 ** 2)
    peak_mb = [baseline_mb]
    stop_event = threading.Event()

    def monitor_memory():
        while not stop_event.is_set():
            try:
                current_mb = process.memory_info().rss / (1024 ** 2)
                peak_mb[0] = max(peak_mb[0], current_mb)
            except psutil.Error:
                pass
            stop_event.wait(0.01)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()

    try:
        result = function()
    finally:
        stop_event.set()
        monitor.join()

        current_mb = process.memory_info().rss / (1024 ** 2)
        peak_mb[0] = max(peak_mb[0], current_mb)

    additional_peak_mb = max(0.0, peak_mb[0] - baseline_mb)

    return result, peak_mb[0], additional_peak_mb


def main():
    # Environment information
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("Joblib:", joblib.__version__)
    print("PyTorch:", torch.__version__)
    print()

    # Load dataset and create stratified 70/30 split
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    sample = X_test[:1]

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            random_state=42,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    accuracy_results = []
    performance_results = []

    for name, template in models.items():
        print(f"Measuring {name}...", flush=True)

        # One warm-up training run
        warmup_model = copy.deepcopy(template)
        warmup_model.fit(X_train, y_train)

        # Five measured training runs
        training_times = []
        training_peak_memories = []

        for run_number in range(5):
            model = copy.deepcopy(template)

            start = time.perf_counter()
            _, peak_mb, _ = measure_peak_memory(
                lambda: model.fit(X_train, y_train)
            )
            elapsed = time.perf_counter() - start

            training_times.append(elapsed)
            training_peak_memories.append(peak_mb)

        training_median_seconds = float(np.median(training_times))
        training_peak_memory_mb = float(
            np.median(training_peak_memories)
        )

        # Fit final model for evaluation and saving
        final_model = copy.deepcopy(template)
        final_model.fit(X_train, y_train)

        accuracy = final_model.score(X_test, y_test)

        # Warm-up inference
        final_model.predict(sample)

        # 100 single-sample inference measurements
        inference_times_ms = []

        for _ in range(100):
            start = time.perf_counter()
            final_model.predict(sample)
            inference_times_ms.append(
                (time.perf_counter() - start) * 1000
            )

        inference_median_ms = float(np.median(inference_times_ms))

        # Approximate process memory during inference
        _, inference_peak_memory_mb, _ = measure_peak_memory(
            lambda: [
                final_model.predict(sample)
                for _ in range(100)
            ]
        )

        # Save the model and measure its serialized file size
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(final_model, model_path)

        model_size_bytes = model_path.stat().st_size
        model_size_kb = model_size_bytes / 1024

        accuracy_results.append({
            "model": name,
            "test_accuracy": round(accuracy, 4),
        })

        performance_results.append({
            "model": name,
            "training_median_seconds": round(
                training_median_seconds, 6
            ),
            "inference_median_ms": round(
                inference_median_ms, 6
            ),
            "model_size_bytes": model_size_bytes,
            "model_size_kb": round(model_size_kb, 3),
            "training_peak_memory_mb": round(
                training_peak_memory_mb, 3
            ),
            "inference_peak_memory_mb": round(
                inference_peak_memory_mb, 3
            ),
        })

        print(f"  Test accuracy: {accuracy:.4f}")
        print(
            f"  Median training time: "
            f"{training_median_seconds:.6f} seconds"
        )
        print(
            f"  Median single-sample inference: "
            f"{inference_median_ms:.6f} ms"
        )
        print(
            f"  Model size: {model_size_bytes} bytes "
            f"({model_size_kb:.3f} KB)"
        )
        print(
            f"  Peak training process memory: "
            f"{training_peak_memory_mb:.3f} MB"
        )
        print(
            f"  Peak inference process memory: "
            f"{inference_peak_memory_mb:.3f} MB"
        )
        print(f"  Saved model: {model_path}")
        print()

    # Save CSV results
    accuracy_path = RESULTS_DIR / "baseline_accuracy.csv"
    performance_path = RESULTS_DIR / "performance_results.csv"

    pd.DataFrame(accuracy_results).to_csv(
        accuracy_path, index=False
    )
    pd.DataFrame(performance_results).to_csv(
        performance_path, index=False
    )

    print("Measurements completed successfully.")
    print("Accuracy CSV:", accuracy_path)
    print("Performance CSV:", performance_path)


if __name__ == "__main__":
    main()