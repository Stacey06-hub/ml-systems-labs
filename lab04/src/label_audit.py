import pandas as pd

def main():
    df = pd.read_csv("data/adult_raw.csv")

    print("INCOME LABEL COUNTS")
    print(df["class"].value_counts())

    print("\nMISSING LABELS:", df["class"].isna().sum())
    print("UNIQUE LABELS:", df["class"].unique())

    audit = df.sample(n=50, random_state=42)

    audit.to_csv("lab04/results/label_audit_50.csv", index=False)

    print("\nSaved 50 rows to lab04/results/label_audit_50.csv")

if __name__ == "__main__":
    main()