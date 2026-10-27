
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("data/adult_raw.csv")

    # Separate the target label from the features
    X = df.drop(columns=["class"])
    y = df["class"]

    # Split before fitting any preprocessing steps
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    print("TRAINING FEATURES:", X_train.shape)
    print("TEST FEATURES:", X_test.shape)
    print("\nTRAINING LABEL DISTRIBUTION")
    print(y_train.value_counts(normalize=True).round(3))
    print("\nTEST LABEL DISTRIBUTION")
    print(y_test.value_counts(normalize=True).round(3))

    print("\nTRAINING ROWS:", len(X_train))
    print("TEST ROWS:", len(X_test))

if __name__ == "__main__":
    main()
