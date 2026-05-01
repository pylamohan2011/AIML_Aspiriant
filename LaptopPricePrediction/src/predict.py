from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "laptop_price_model.joblib"
METADATA_PATH = PROJECT_DIR / "models" / "metadata.json"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {"features": []}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def predict_price(features: dict[str, Any]) -> float:
    model = load_model()
    metadata = load_metadata()
    ordered_features = metadata.get("features") or list(features)
    row = {col: features.get(col) for col in ordered_features}
    prediction = model.predict(pd.DataFrame([row]))[0]
    return round(float(prediction), 2)
