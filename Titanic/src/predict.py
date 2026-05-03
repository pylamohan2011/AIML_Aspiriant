from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "titanic_model.joblib"


def _ensure_model() -> None:
    if not MODEL_PATH.exists():
        from src.train import train

        train()


import re


def _normalize_payload(payload: dict[str, object]) -> dict[str, object]:
    name = str(payload.get("Name", "Unknown"))
    title_match = re.search(r",\s*([^\.]+)\.", name)
    title = title_match.group(1).strip() if title_match else "Unknown"
    return {
        "Pclass": int(payload["Pclass"]),
        "Sex": str(payload["Sex"]).lower(),
        "Age": float(payload["Age"]),
        "SibSp": int(payload["SibSp"]),
        "Parch": int(payload["Parch"]),
        "Fare": float(payload["Fare"]),
        "Embarked": str(payload["Embarked"]).upper(),
        "Title": title,
    }


def predict_survival(payload: dict[str, object]) -> dict[str, object]:
    _ensure_model()
    model = joblib.load(MODEL_PATH)
    normalized = _normalize_payload(payload)
    input_df = pd.DataFrame([normalized])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return {
        "survived": bool(prediction),
        "survival_probability": float(probability),
        "model_version": MODEL_PATH.name,
    }
