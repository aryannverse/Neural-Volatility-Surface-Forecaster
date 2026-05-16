from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RawChainRecord(Base):
    __tablename__ = "raw_chain_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    expiration: Mapped[datetime] = mapped_column(DateTime, index=True)
    option_type: Mapped[str] = mapped_column(String(8))
    strike: Mapped[float] = mapped_column(Float)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    mid_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    iv: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    open_interest: Mapped[int | None] = mapped_column(Integer, nullable=True)


class SurfaceRecord(Base):
    __tablename__ = "surface_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    strike_axis: Mapped[str] = mapped_column(Text)  # JSON text
    expiry_axis: Mapped[str] = mapped_column(Text)
    iv_grid: Mapped[str] = mapped_column(Text)


class PredictionRecord(Base):
    __tablename__ = "prediction_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    horizon: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    prediction_grid: Mapped[str] = mapped_column(Text)


class MetricRecord(Base):
    __tablename__ = "metric_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    run_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    rmse: Mapped[float] = mapped_column(Float)
    mae: Mapped[float] = mapped_column(Float)
    directional_skew_accuracy: Mapped[float] = mapped_column(Float)
    surface_similarity: Mapped[float] = mapped_column(Float)
