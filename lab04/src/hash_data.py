
import hashlib
import os
import pandas as pd
import joblib

from preprocess_data import OutlierClipper


def main():
    data = pd.read_csv("data/adult_raw.csv")

    X = data.drop(columns=["class"]).copy()

    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] + 16)

    preprocessor = joblib.load(
        "lab04/results/preprocessor.joblib"
    )

    processed = preprocessor.transform(X)

    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    result = hashlib.sha256(processed.tobytes()).hexdigest()

    os.makedirs("lab04/results", exist_ok=True)

    with open("lab04/results/processed_data_hash.txt", "w") as file:
        file.write(result + "\n")

    print("Processed dataset SHA256:", result)
    print("Hash saved successfully!")


if __name__ == "__main__":
    main()
