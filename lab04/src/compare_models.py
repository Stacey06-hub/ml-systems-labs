
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def main():
    df = pd.read_csv("data/adult_raw.csv")

    X = df.drop(columns=["class"]).copy()
    y = df["class"]

    # Feature engineering: create a new age group feature
    X["age_group"] = pd.cut(
        X["age"],
        bins=[16, 25, 45, 65, 100],
        labels=["young", "adult", "senior", "older"]
    ).astype(str)

    X_simple = X.drop(columns=["age_group"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_simple, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_fe = X_train.copy()
    X_test_fe = X_test.copy()

    X_train_fe["age_group"] = pd.cut(
        X_train_fe["age"],
        bins=[16, 25, 45, 65, 100],
        labels=["young", "adult", "senior", "older"]
    ).astype(str)

    X_test_fe["age_group"] = pd.cut(
        X_test_fe["age"],
        bins=[16, 25, 45, 65, 100],
        labels=["young", "adult", "senior", "older"]
    ).astype(str)

    def build_model(data):
        numeric = data.select_dtypes(include="number").columns.tolist()
        categorical = data.select_dtypes(exclude="number").columns.tolist()

        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical)
        ])

        return Pipeline([
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ])

    simple_model = build_model(X_train)
    simple_model.fit(X_train, y_train)
    simple_predictions = simple_model.predict(X_test)

    feature_model = build_model(X_train_fe)
    feature_model.fit(X_train_fe, y_train)
    feature_predictions = feature_model.predict(X_test_fe)

    print("RESULTS")
    print("Simple model accuracy:", round(
        accuracy_score(y_test, simple_predictions), 4))
    print("Simple model F1:", round(
        f1_score(y_test, simple_predictions, pos_label=">50K"), 4))

    print("\nModel with age group feature accuracy:", round(
        accuracy_score(y_test, feature_predictions), 4))
    print("Model with age group feature F1:", round(
        f1_score(y_test, feature_predictions, pos_label=">50K"), 4))

if __name__ == "__main__":
    main()
