from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .sample_data import create_sample_dataset
except ImportError:
    from sample_data import create_sample_dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "laptop_price.csv"
MODEL_PATH = PROJECT_DIR / "models" / "laptop_price_model.joblib"
METRICS_PATH = PROJECT_DIR / "models" / "metrics.json"
METADATA_PATH = PROJECT_DIR / "models" / "metadata.json"

KAGGLE_FEATURES = [
    "Company",
    "TypeName",
    "Inches",
    "Resolution",
    "Cpu",
    "Ram",
    "Memory",
    "Gpu",
    "OpSys",
    "Weight",
]


def load_training_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        create_sample_dataset(data_path)
        print(f"Created sample dataset at {data_path}")
        print("Replace it with your Kaggle laptop price dataset at this path for the final model.")
    return pd.read_csv(data_path)


def pick_target(df: pd.DataFrame) -> str:
    for target in ("Price_euros", "price_euros", "price", "Price"):
        if target in df.columns:
            return target
    raise ValueError("No target column found. Expected Price_euros, price_euros, price, or Price.")


def pick_features(df: pd.DataFrame, target: str) -> list[str]:
    preferred = [col for col in KAGGLE_FEATURES if col in df.columns]
    if preferred:
        return preferred
    excluded = {target, "Id", "LaptopID"}
    return [col for col in df.columns if col not in excluded]


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in x.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }


def train(data_path: Path = DATA_PATH) -> tuple[Pipeline, dict[str, Any]]:
    df = load_training_data(data_path)
    target = pick_target(df)
    features = pick_features(df, target)
    train_df = df[features + [target]].dropna(subset=[target]).copy()

    x = train_df[features]
    y = train_df[target].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    preprocessor = build_preprocessor(x_train)

    candidates = {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        results[name] = evaluate_model(pipeline, x_test, y_test)
        fitted_models[name] = pipeline

    best_name = min(results, key=lambda item: results[item]["rmse"])
    best_model = fitted_models[best_name]
    metrics = {
        "best_model": best_name,
        "rows": int(len(train_df)),
        "features": features,
        "target": target,
        "model_results": results,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "features": features,
                "target": target,
                "model_path": str(MODEL_PATH.relative_to(PROJECT_DIR)),
                "expected_schema": {
                    col: str(dtype) for col, dtype in x[features].dtypes.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_model, metrics


if __name__ == "__main__":
    _, metrics = train()
    print(json.dumps(metrics, indent=2))
