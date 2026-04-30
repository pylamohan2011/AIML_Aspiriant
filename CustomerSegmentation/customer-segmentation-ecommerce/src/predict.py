from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .train import FEATURES, METADATA_PATH, MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {"features": FEATURES, "segment_map": {}}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def predict_segment(features: dict[str, Any]) -> dict[str, Any]:
    metadata = load_metadata()
    ordered_features = metadata.get("features") or FEATURES
    row = {col: features.get(col) for col in ordered_features}
    model = load_model()
    cluster = int(model.predict(pd.DataFrame([row]))[0])
    segment = metadata.get("segment_map", {}).get(str(cluster), f"Cluster {cluster}")
    return {"cluster": cluster, "segment": segment}
