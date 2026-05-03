from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from src.sample_data import create_sample_titanic

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = BASE_DIR / "models" / "titanic_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]
TARGET_COLUMN = "Survived"


def _extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", str(name))
    return match.group(1).strip() if match else "Unknown"


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Name" in df.columns:
        df["Title"] = df["Name"].astype(str).apply(_extract_title)
    else:
        df["Title"] = "Unknown"
    return df


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return create_sample_titanic()


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=150, random_state=42)),
        ]
    )


def train() -> None:
    df = load_data()
    df = _prepare_dataframe(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Data must include '{TARGET_COLUMN}' column.")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    METADATA_PATH.write_text(
        json.dumps(
            {
                "model_type": "RandomForestClassifier",
                "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                "data_source": str(DATA_PATH) if DATA_PATH.exists() else "sample dataset",
            },
            indent=2,
        )
    )

    print("Training complete.")
    print(f"Model saved to {MODEL_PATH}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    train()
