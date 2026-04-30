from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .sample_data import create_sample_dataset
except ImportError:  # Allows running as: python src/train.py
    from sample_data import create_sample_dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_CSV_PATH = DATA_DIR / "stock_prices.csv"
DEFAULT_EXCEL_PATH = DATA_DIR / "stock_prices.xlsx"
MODEL_PATH = PROJECT_DIR / "models" / "stock_price_model.joblib"
METRICS_PATH = PROJECT_DIR / "models" / "metrics.json"
METADATA_PATH = PROJECT_DIR / "models" / "metadata.json"

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "lag_1",
    "lag_2",
    "lag_5",
    "ma_5",
    "ma_10",
    "return_1",
    "volatility_5",
]

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def download_yahoo_data(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance or provide data/stock_prices.csv or .xlsx.") from exc

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No Yahoo Finance data returned for ticker {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df.reset_index()


def load_training_data(
    data_path: Optional[Path] = None,
    ticker: str = "AAPL",
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    if data_path:
        path = Path(data_path)
    elif DEFAULT_EXCEL_PATH.exists():
        path = DEFAULT_EXCEL_PATH
    elif DEFAULT_CSV_PATH.exists():
        path = DEFAULT_CSV_PATH
    else:
        path = DEFAULT_CSV_PATH
        try:
            df = download_yahoo_data(ticker=ticker, start=start, end=end)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Downloaded Yahoo Finance data for {ticker} to {path}")
            return df
        except Exception as exc:
            create_sample_dataset(path)
            print(f"Yahoo/data load failed: {exc}")
            print(f"Created sample stock dataset at {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]
    aliases = {
        "Adj Close": "Close",
        "AdjClose": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "date": "Date",
    }
    normalized = normalized.rename(columns={col: aliases.get(col, col) for col in normalized.columns})
    missing = REQUIRED_COLUMNS.difference(normalized.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "Date" in normalized.columns:
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        normalized = normalized.sort_values("Date")
    return normalized


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = normalize_columns(df)
    for col in REQUIRED_COLUMNS:
        featured[col] = pd.to_numeric(featured[col], errors="coerce")

    featured["lag_1"] = featured["Close"].shift(1)
    featured["lag_2"] = featured["Close"].shift(2)
    featured["lag_5"] = featured["Close"].shift(5)
    featured["ma_5"] = featured["Close"].rolling(window=5).mean()
    featured["ma_10"] = featured["Close"].rolling(window=10).mean()
    featured["return_1"] = featured["Close"].pct_change()
    featured["volatility_5"] = featured["return_1"].rolling(window=5).std()
    featured["target_next_close"] = featured["Close"].shift(-1)
    return featured.dropna(subset=FEATURES + ["target_next_close"]).copy()


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    return {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "mape_percent": round(float(mape), 4),
    }


def train(
    data_path: Optional[Path] = None,
    ticker: str = "AAPL",
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> tuple[Pipeline, dict[str, object]]:
    raw_df = load_training_data(data_path=data_path, ticker=ticker, start=start, end=end)
    model_df = add_time_series_features(raw_df)
    if len(model_df) < 60:
        raise ValueError("Need at least 60 usable rows after feature engineering.")

    split_index = int(len(model_df) * 0.8)
    x = model_df[FEATURES]
    y = model_df["target_next_close"]
    x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    candidates = {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        ),
    }

    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        results[name] = evaluate_model(model, x_test, y_test)
        fitted_models[name] = model

    best_name = min(results, key=lambda item: results[item]["rmse"])
    best_model = fitted_models[best_name]
    metrics = {
        "best_model": best_name,
        "ticker": ticker,
        "rows": int(len(model_df)),
        "features": FEATURES,
        "target": "target_next_close",
        "model_results": results,
        "last_close": round(float(model_df["Close"].iloc[-1]), 4),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "ticker": ticker,
                "features": FEATURES,
                "target": "target_next_close",
                "model_path": str(MODEL_PATH.relative_to(PROJECT_DIR)),
                "latest_feature_row": model_df[FEATURES].iloc[-1].to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_model, metrics


if __name__ == "__main__":
    _, training_metrics = train()
    print(json.dumps(training_metrics, indent=2))
