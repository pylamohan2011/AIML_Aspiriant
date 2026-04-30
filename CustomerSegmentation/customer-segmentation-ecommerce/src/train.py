from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .sample_data import create_sample_dataset
except ImportError:  # Allows running as: python src/train.py
    from sample_data import create_sample_dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_CSV_PATH = DATA_DIR / "online_retail.csv"
DEFAULT_EXCEL_PATH = DATA_DIR / "online_retail.xlsx"
MODEL_PATH = PROJECT_DIR / "models" / "customer_segmentation_model.joblib"
METRICS_PATH = PROJECT_DIR / "models" / "metrics.json"
METADATA_PATH = PROJECT_DIR / "models" / "metadata.json"

FEATURES = [
    "recency_days",
    "frequency",
    "total_quantity",
    "total_spend",
    "average_order_value",
    "unique_products",
]


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    if data_path:
        path = Path(data_path)
    elif DEFAULT_EXCEL_PATH.exists():
        path = DEFAULT_EXCEL_PATH
    elif DEFAULT_CSV_PATH.exists():
        path = DEFAULT_CSV_PATH
    else:
        path = create_sample_dataset(DEFAULT_CSV_PATH)
        print(f"Created sample Kaggle-like Online Retail dataset at {path}")
        print("Replace it with your Kaggle CSV/XLSX for the final model.")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]
    aliases = {
        "Invoice": "InvoiceNo",
        "InvoiceNo.": "InvoiceNo",
        "Invoice Date": "InvoiceDate",
        "Customer Id": "CustomerID",
        "Customer ID": "CustomerID",
        "Price": "UnitPrice",
        "Unit Price": "UnitPrice",
        "Sales": "LineTotal",
    }
    normalized = normalized.rename(columns={col: aliases.get(col, col) for col in normalized.columns})

    required = {"InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    missing = required.difference(normalized.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return normalized


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    transactions = normalize_columns(df)
    transactions = transactions.dropna(subset=["CustomerID", "InvoiceDate"]).copy()
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"], errors="coerce")
    transactions["Quantity"] = pd.to_numeric(transactions["Quantity"], errors="coerce")
    transactions["UnitPrice"] = pd.to_numeric(transactions["UnitPrice"], errors="coerce")
    transactions = transactions.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])
    transactions = transactions[(transactions["Quantity"] > 0) & (transactions["UnitPrice"] > 0)]

    if "LineTotal" not in transactions.columns:
        transactions["LineTotal"] = transactions["Quantity"] * transactions["UnitPrice"]
    else:
        transactions["LineTotal"] = pd.to_numeric(transactions["LineTotal"], errors="coerce")

    if transactions.empty:
        raise ValueError("No usable positive transaction rows found.")

    snapshot_date = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)
    product_col = "StockCode" if "StockCode" in transactions.columns else "Description"
    aggregations = {
        "InvoiceDate": lambda values: (snapshot_date - values.max()).days,
        "InvoiceNo": pd.Series.nunique,
        "Quantity": "sum",
        "LineTotal": "sum",
    }
    if product_col in transactions.columns:
        aggregations[product_col] = pd.Series.nunique

    customer_df = transactions.groupby("CustomerID").agg(aggregations).reset_index()
    rename_map = {
        "InvoiceDate": "recency_days",
        "InvoiceNo": "frequency",
        "Quantity": "total_quantity",
        "LineTotal": "total_spend",
        product_col: "unique_products",
    }
    customer_df = customer_df.rename(columns=rename_map)
    if "unique_products" not in customer_df.columns:
        customer_df["unique_products"] = 1
    customer_df["average_order_value"] = customer_df["total_spend"] / customer_df["frequency"].clip(lower=1)
    return customer_df[["CustomerID"] + FEATURES].dropna().copy()


def make_segment_labels(customer_df: pd.DataFrame, labels: np.ndarray) -> dict[str, str]:
    labeled = customer_df.copy()
    labeled["cluster"] = labels
    centroids = labeled.groupby("cluster")[FEATURES].mean()

    monetary_rank = centroids["total_spend"].rank(pct=True)
    frequency_rank = centroids["frequency"].rank(pct=True)
    recency_rank = centroids["recency_days"].rank(pct=True, ascending=False)
    aov_rank = centroids["average_order_value"].rank(pct=True)
    scores = (monetary_rank + frequency_rank + recency_rank + aov_rank) / 4

    ordered_clusters = scores.sort_values(ascending=False).index.tolist()
    label_names = [
        "Champions",
        "Loyal Customers",
        "Big Spenders",
        "Occasional Buyers",
        "At Risk",
        "Low Value",
    ]
    return {str(cluster): label_names[min(index, len(label_names) - 1)] for index, cluster in enumerate(ordered_clusters)}


def evaluate_candidates(x: pd.DataFrame) -> tuple[Pipeline, int, dict[str, dict[str, float]], np.ndarray]:
    max_k = min(6, len(x) - 1)
    if max_k < 2:
        raise ValueError("Need at least three customers to build segmentation clusters.")

    results: dict[str, dict[str, float]] = {}
    fitted: dict[int, tuple[Pipeline, np.ndarray]] = {}
    for k in range(2, max_k + 1):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KMeans(n_clusters=k, random_state=42, n_init=20)),
            ]
        )
        labels = pipeline.fit_predict(x)
        results[str(k)] = {
            "silhouette": round(float(silhouette_score(x, labels)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(x, labels)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(x, labels)), 4),
            "inertia": round(float(pipeline.named_steps["model"].inertia_), 4),
        }
        fitted[k] = (pipeline, labels)

    best_k = max(results, key=lambda key: results[key]["silhouette"])
    best_model, best_labels = fitted[int(best_k)]
    return best_model, int(best_k), results, best_labels


def train(data_path: Optional[Path] = None) -> tuple[Pipeline, dict[str, object]]:
    raw_df = load_raw_data(data_path)
    customer_df = build_customer_features(raw_df)
    if len(customer_df) < 10:
        raise ValueError("Need at least 10 customer records for reliable segmentation.")

    x = customer_df[FEATURES]
    model, best_k, model_results, labels = evaluate_candidates(x)
    segment_map = make_segment_labels(customer_df, labels)
    customer_df["cluster"] = labels
    customer_df["segment"] = customer_df["cluster"].astype(str).map(segment_map)

    segment_counts = customer_df["segment"].value_counts().to_dict()
    metrics = {
        "best_k": best_k,
        "rows": int(len(customer_df)),
        "features": FEATURES,
        "model_results": model_results,
        "segment_counts": {str(key): int(value) for key, value in segment_counts.items()},
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "features": FEATURES,
                "segment_map": segment_map,
                "model_path": str(MODEL_PATH.relative_to(PROJECT_DIR)),
                "example_customer": customer_df[FEATURES].median(numeric_only=True).to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model, metrics


if __name__ == "__main__":
    _, training_metrics = train()
    print(json.dumps(training_metrics, indent=2))
