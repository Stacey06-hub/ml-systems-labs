
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from preprocess_data import OutlierClipper


def build_model(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

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
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])


def main():
    df = pd.read_csv("data/adult_raw.csv")

    X = df.drop(columns=["class"]).copy()
    y = df["class"].copy()

    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = (
        X["hours-per-week"] / (X["age"] - 16)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Remove duplicates from training data only
    train_data = X_train.copy()
    train_data["class"] = y_train
    train_data = train_data.drop_duplicates()

    y_train = train_data["class"]
    X_train = train_data.drop(columns=["class"])

    original_columns = [
        col for col in X.columns
        if col not in ["capital_net", "hours_per_age"]
    ]

    simple_model = build_model(X_train[original_columns])
    simple_model.fit(X_train[original_columns], y_train)
    simple_predictions = simple_model.predict(X_test[original_columns])

    feature_model = build_model(X_train)
    feature_model.fit(X_train, y_train)
    feature_predictions = feature_model.predict(X_test)

    print("RESULTS")
    print("Simple model accuracy:",
          round(accuracy_score(y_test, simple_predictions), 4))
    print("Simple model F1:",
          round(f1_score(y_test, simple_predictions, pos_label=">50K"), 4))

    print("\nModel with new features accuracy:",
          round(accuracy_score(y_test, feature_predictions), 4))
    print("Model with new features F1:",
          round(f1_score(y_test, feature_predictions, pos_label=">50K"), 4))


if __name__ == "__main__":
    main()
