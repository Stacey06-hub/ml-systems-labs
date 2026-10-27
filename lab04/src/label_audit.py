
import pandas as pd


def main():
    df = pd.read_csv("data/adult_raw.csv")

    print("INCOME LABEL COUNTS")
    print(df["class"].value_counts())

    print("\nMISSING LABELS:", df["class"].isna().sum())
    print("UNIQUE LABELS:", df["class"].unique())

    # Check whether identical features have different income labels
    features = df.drop(columns=["class"])
    check = features.copy()
    check["class"] = df["class"]

    label_counts = check.groupby(
        list(features.columns),
        dropna=False
    )["class"].nunique()

    conflicts = label_counts[label_counts > 1]

    print("\nCONFLICTING DUPLICATE FEATURE GROUPS:", len(conflicts))

    if len(conflicts) > 0:
        print("Warning: Some identical feature rows have different labels.")
    else:
        print("No conflicting duplicate labels found.")

    # Save a reproducible sample of 50 rows for manual review
    audit = df.sample(n=50, random_state=42)
    audit.to_csv("lab04/results/label_audit_50.csv", index=False)

    print("\nSaved 50 rows to lab04/results/label_audit_50.csv")


if __name__ == "__main__":
    main()
