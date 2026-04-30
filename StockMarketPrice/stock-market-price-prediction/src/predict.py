from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from .train import FEATURES, METADATA_PATH, MODEL_PATH, add_time_series_features, download_yahoo_data


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {"features": FEATURES}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def predict_next_close(features: dict[str, Any]) -> float:
    model = load_model()
    ordered = load_metadata().get("features") or FEATURES
    row = {col: features.get(col) for col in ordered}
    prediction = model.predict(pd.DataFrame([row]))[0]
    return round(float(prediction), 4)


def predict_from_yahoo(ticker: str, start: str = "2020-01-01", end: Optional[str] = None) -> dict[str, float]:
    raw_df = download_yahoo_data(ticker=ticker, start=start, end=end)
    featured = add_time_series_features(raw_df)
    latest_features = featured[FEATURES].iloc[-1].to_dict()
    predicted = predict_next_close(latest_features)
    last_close = float(featured["Close"].iloc[-1])
    return {
        "last_close": round(last_close, 4),
        "predicted_next_close": predicted,
        "predicted_change": round(predicted - last_close, 4),
        "predicted_change_percent": round(((predicted - last_close) / last_close) * 100, 4),
    }
