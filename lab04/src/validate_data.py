
import pandas as pd

DATA_PATH = "data/adult_raw.csv"

EXPECTED_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "class",
]

EXPECTED_TYPES = {
    "age": "int64",
    "workclass": "object",
    "fnlwgt": "int64",
    "education": "object",
    "education-num": "int64",
    "marital-status": "object",
    "occupation": "object",
    "relationship": "object",
    "race": "object",
    "sex": "object",
    "capital-gain": "int64",
    "capital-loss": "int64",
    "hours-per-week": "int64",
    "native-country": "object",
    "class": "object",
}

NUMERIC_RANGES = {
    "age": (17, 90),
    "fnlwgt": (0, None),
    "education-num": (1, 16),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
    "hours-per-week": (1, 99),
}

VALID_INCOME_LABELS = {"<=50K", ">50K"}


def main():
    df = pd.read_csv(
        DATA_PATH,
        na_values=["?"],
        keep_default_na=True,
    )

    print("DATASET SHAPE")
    print(df.shape)

    print("\nSCHEMA CHECK")
    if df.columns.tolist() == EXPECTED_COLUMNS:
        print("PASS: Column names and order are correct.")
    else:
        print("FAIL: Column names or order do not match.")
        print("Expected:", EXPECTED_COLUMNS)
        print("Actual:", df.columns.tolist())

    print("\nDATA TYPES")
    print(df.dtypes)

    print("\nDATA TYPE CHECK")
    for column, expected_type in EXPECTED_TYPES.items():
        actual_type = str(df[column].dtype)
        status = "PASS" if actual_type == expected_type else "FAIL"
        print(
            f"{column}: {status} "
            f"(expected {expected_type}, got {actual_type})"
        )

    print("\nMISSING VALUES")
    print(df.isnull().sum())

    print("\nNUMERICAL RANGE CHECKS")
    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        values = df[column]
        invalid = values.notna()

        if minimum is not None:
            invalid &= values < minimum

        if maximum is not None:
            invalid |= values.notna() & (values > maximum)

        count = int(invalid.sum())
        print(f"{column}: {count} range violations")

    print("\nDUPLICATE ROWS")
    print(int(df.duplicated().sum()))

    print("\nINCOME LABEL CHECK")
    invalid_labels = (
        df["class"].notna()
        & ~df["class"].isin(VALID_INCOME_LABELS)
    )
    print("Missing income labels:", int(df["class"].isna().sum()))
    print("Invalid income labels:", int(invalid_labels.sum()))

    print("\nFIRST 5 ROWS")
    print(df.head())


if __name__ == "__main__":
    main()
