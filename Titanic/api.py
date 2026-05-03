from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_survival

app = FastAPI(title="Titanic Survival Prediction API", version="1.0.0")


class PassengerFeatures(BaseModel):
    Pclass: int = Field(3, ge=1, le=3)
    Sex: str = Field("male")
    Age: float = Field(30.0, ge=0)
    SibSp: int = Field(0, ge=0)
    Parch: int = Field(0, ge=0)
    Fare: float = Field(7.25, ge=0)
    Embarked: str = Field("S")
    Name: str = Field("Braund, Mr. Owen Harris")


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "Titanic survival prediction API is running"}


@app.post("/predict")
def predict(features: PassengerFeatures) -> dict[str, Any]:
    payload = features.model_dump()
    prediction = predict_survival(payload)
    return {
        "survived": bool(prediction["survived"]),
        "survival_probability": float(prediction["survival_probability"]),
        "model_version": prediction["model_version"],
    }
