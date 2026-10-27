
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from preprocess_data import OutlierClipper


def main():
    df = pd.read_csv("data/adult_raw.csv")

    X = df.drop(columns=["class"]).copy()
    y = df["class"].copy()

    # Create required engineered features
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = (
        X["hours-per-week"] / (X["age"] - 16)
    )

    # Split into training and test sets before preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Remove duplicate training rows only
    train_data = X_train.copy()
    train_data["class"] = y_train
    train_data = train_data.drop_duplicates()

    X_train = train_data.drop(columns=["class"])
    y_train = train_data["class"]

    # Identify numeric and categorical columns
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    # Fit imputation, clipping and scaling on training data only
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

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    # Train and evaluate
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("MODEL EVALUATION")
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )
    print(
        "F1 score for >50K:",
        round(
            f1_score(y_test, predictions, pos_label=">50K"),
            4
        )
    )

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    joblib.dump(model, "lab04/results/adult_model.joblib")
    print("Saved model to lab04/results/adult_model.joblib")


if __name__ == "__main__":
    main()
