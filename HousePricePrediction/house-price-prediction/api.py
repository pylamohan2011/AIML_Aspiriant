from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_price


app = FastAPI(title="House Price Prediction API", version="1.0.0")


class HouseFeatures(BaseModel):
    OverallQual: int = Field(7, ge=1, le=10)
    GrLivArea: float = Field(1500, gt=0)
    GarageCars: int = Field(2, ge=0, le=5)
    TotalBsmtSF: float = Field(900, ge=0)
    FirstFlrSF: float = Field(900, ge=0, alias="1stFlrSF")
    FullBath: int = Field(2, ge=0, le=5)
    BedroomAbvGr: int = Field(3, ge=0, le=10)
    YearBuilt: int = Field(2000, ge=1800, le=2030)
    YearRemodAdd: int = Field(2010, ge=1800, le=2030)
    LotArea: float = Field(9000, gt=0)
    Neighborhood: str = "CollgCr"
    HouseStyle: str = "1Story"

    class Config:
        populate_by_name = True


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "House price model API is running"}


@app.post("/predict")
def predict(features: HouseFeatures) -> dict[str, Any]:
    if hasattr(features, "model_dump"):
        payload = features.model_dump(by_alias=True)
    else:
        payload = features.dict(by_alias=True)
    return {"predicted_price": predict_price(payload), "currency": "USD"}
