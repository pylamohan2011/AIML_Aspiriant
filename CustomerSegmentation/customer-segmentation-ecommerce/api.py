from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_segment
from src.train import MODEL_PATH, train


app = FastAPI(title="E-commerce Customer Segmentation API", version="1.0.0")


class CustomerFeatures(BaseModel):
    recency_days: float = Field(30, ge=0)
    frequency: float = Field(8, ge=1)
    total_quantity: float = Field(160, ge=0)
    total_spend: float = Field(2500, ge=0)
    average_order_value: float = Field(312.5, ge=0)
    unique_products: float = Field(20, ge=1)


@app.on_event("startup")
def ensure_model() -> None:
    if not MODEL_PATH.exists():
        train()


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "Customer segmentation API is running"}


@app.post("/predict")
def predict(features: CustomerFeatures) -> dict[str, Any]:
    payload = features.model_dump() if hasattr(features, "model_dump") else features.dict()
    return predict_segment(payload)
