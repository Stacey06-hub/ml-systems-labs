
import pandas as pd

def main():
    df = pd.read_csv("data/adult_raw.csv")

    print("NUMERICAL COLUMN SUMMARY")
    print(df.describe())

    print("\nCATEGORICAL VALUE COUNTS")
    for column in df.select_dtypes(include="object").columns:
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False).head(10))

if __name__ == "__main__":
    main()
