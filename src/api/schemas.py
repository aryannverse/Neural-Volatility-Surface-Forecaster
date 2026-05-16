from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SurfaceResponse(BaseModel):
    ticker: str
    timestamp: datetime
    strike_axis: list[float]
    expiry_axis: list[float]
    iv_grid: list[list[float]]


class ForecastResponse(BaseModel):
    ticker: str
    model_name: str
    horizon: int
    created_at: datetime
    prediction_grid: list[list[float]]


class TrainRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=16)
    model_name: str = Field(default="transformer")
    lookback: int = Field(default=20, ge=5, le=120)
    horizon: int = Field(default=1, ge=1, le=20)
    epochs: int = Field(default=30, ge=1, le=500)
    batch_size: int = Field(default=32, ge=4, le=512)


class TrainResponse(BaseModel):
    ticker: str
    model_name: str
    metrics: dict[str, float]
    best_val_loss: float
    checkpoint: str
