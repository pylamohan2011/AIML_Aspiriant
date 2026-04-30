from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_next_close
from src.train import MODEL_PATH, train


app = FastAPI(title="Stock Market Price Prediction API", version="1.0.0")


class StockFeatures(BaseModel):
    Open: float = Field(190.0, gt=0)
    High: float = Field(195.0, gt=0)
    Low: float = Field(188.0, gt=0)
    Close: float = Field(192.0, gt=0)
    Volume: float = Field(60_000_000, ge=0)
    lag_1: float = Field(191.0, gt=0)
    lag_2: float = Field(190.0, gt=0)
    lag_5: float = Field(188.0, gt=0)
    ma_5: float = Field(190.5, gt=0)
    ma_10: float = Field(189.5, gt=0)
    return_1: float = 0.005
    volatility_5: float = Field(0.018, ge=0)


@app.on_event("startup")
def ensure_model() -> None:
    if not MODEL_PATH.exists():
        train()


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "Stock price model API is running"}


@app.post("/predict")
def predict(features: StockFeatures) -> dict[str, Any]:
    payload = features.model_dump() if hasattr(features, "model_dump") else features.dict()
    return {"predicted_next_close": predict_next_close(payload), "currency": "USD"}
