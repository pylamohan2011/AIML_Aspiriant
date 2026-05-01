from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_price
from src.train import MODEL_PATH, train

app = FastAPI(title="Laptop Price Prediction API", version="1.0.0")


class LaptopFeatures(BaseModel):
    Company: str = Field("Dell", min_length=1)
    TypeName: str = Field("Notebook", min_length=1)
    Inches: float = Field(15.6, gt=0)
    Resolution: str = Field("1920x1080", min_length=1)
    Cpu: str = Field("Intel Core i5", min_length=1)
    Ram: int = Field(8, ge=1)
    Memory: str = Field("256GB SSD", min_length=1)
    Gpu: str = Field("Intel", min_length=1)
    OpSys: str = Field("Windows", min_length=1)
    Weight: float = Field(1.8, gt=0)


@app.on_event("startup")
def ensure_model() -> None:
    if not MODEL_PATH.exists():
        train()


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "Laptop price model API is running"}


@app.post("/predict")
def predict(features: LaptopFeatures) -> dict[str, Any]:
    payload = features.model_dump() if hasattr(features, "model_dump") else features.dict()
    return {
        "predicted_price_euros": predict_price(payload),
        "currency": "EUR",
    }
