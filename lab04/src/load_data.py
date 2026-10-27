from sklearn.datasets import fetch_openml
import os

def main():
    print("Downloading the Adult dataset...")

    data = fetch_openml(
        name="adult",
        version=2,
        as_frame=True
    )

    df = data.frame

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/adult_raw.csv", index=False)

    print("Dataset downloaded successfully!")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Saved to data/adult_raw.csv")

if __name__ == "__main__":
    main()