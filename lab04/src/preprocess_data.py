import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_ = X.quantile(0.01)
        self.upper_ = X.quantile(0.99)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.clip(self.lower_, self.upper_, axis=1).to_numpy()


def main():
    df = pd.read_csv("data/adult_raw.csv")

    X = df.drop(columns=["class"]).copy()
    y = df["class"]

    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Remove exact duplicate feature rows from training data only.
    train_data = X_train.copy()
    train_data["class"] = y_train
    train_data = train_data.drop_duplicates()

    X_train = train_data.drop(columns=["class"])
    y_train = train_data["class"]

    numeric_columns = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", OutlierClipper()),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    joblib.dump(
        preprocessor,
        "lab04/results/preprocessor.joblib"
    )

    print("Preprocessing completed!")
    print("Training rows after duplicate removal:", len(X_train))
    print("Test rows:", len(X_test))
    print("Training processed shape:", X_train_processed.shape)
    print("Test processed shape:", X_test_processed.shape)
    print("Saved preprocessor to lab04/results/preprocessor.joblib")


if __name__ == "__main__":
    main()