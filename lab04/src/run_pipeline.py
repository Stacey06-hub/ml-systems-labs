
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "lab04" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

test_size = config["split"]["test_size"]
random_state = config["split"]["random_state"]

if test_size != 0.3:
    raise ValueError("Lab 4 requires a 70/30 train/test split.")

print("Configuration loaded successfully.")
print(f"Test size: {test_size}")
print(f"Random state: {random_state}")

scripts = [
    "lab04/src/load_data.py",
    "lab04/src/validate_data.py",
    "lab04/src/check_ranges.py",
    "lab04/src/split_data.py",
    "lab04/src/preprocess_data.py",
    "lab04/src/train_model.py",
    "lab04/src/compare_models.py",
    "lab04/src/label_audit.py",
    "lab04/src/hash_data.py",
]

for script in scripts:
    print(f"\nRunning {script}...")
    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        check=False
    )

    if result.returncode != 0:
        print(f"ERROR: {script} failed.")
        sys.exit(result.returncode)

print("\nPipeline finished!")
