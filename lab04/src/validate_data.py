import pandas as pd

def main():
    df = pd.read_csv("data/adult_raw.csv")

    print("DATASET SHAPE")
    print(df.shape)

    print("\nCOLUMN NAMES")
    print(df.columns.tolist())

    print("\nDATA TYPES")
    print(df.dtypes)

    print("\nMISSING VALUES")
    print(df.isnull().sum())

    print("\nDUPLICATE ROWS")
    print(df.duplicated().sum())

    print("\nFIRST 5 ROWS")
    print(df.head())

if __name__ == "__main__":
    main()