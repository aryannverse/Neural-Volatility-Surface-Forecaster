from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import MetricRecord, PredictionRecord, RawChainRecord, SurfaceRecord


class ForecastRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def save_raw_chain(self, chain: pd.DataFrame) -> None:
        rows = []
        for _, r in chain.iterrows():
            rows.append(
                RawChainRecord(
                    ticker=str(r["ticker"]),
                    timestamp=pd.Timestamp(r["timestamp"]).to_pydatetime(),
                    expiration=pd.Timestamp(r["expiry"]).to_pydatetime(),
                    option_type=str(r["option_type"]),
                    strike=float(r["strike"]),
                    bid=float(r["bid"]) if pd.notna(r["bid"]) else None,
                    ask=float(r["ask"]) if pd.notna(r["ask"]) else None,
                    mid_price=float(r["mid_price"]) if pd.notna(r["mid_price"]) else None,
                    iv=float(r["iv"]) if pd.notna(r["iv"]) else None,
                    volume=int(r["volume"]) if pd.notna(r.get("volume")) else None,
                    open_interest=int(r["open_interest"]) if pd.notna(r.get("open_interest")) else None,
                )
            )
        with self.session_factory() as s:
            s.add_all(rows)
            s.commit()

    def save_surface(
        self,
        ticker: str,
        timestamp: datetime,
        strike_axis: np.ndarray,
        expiry_axis: np.ndarray,
        iv_grid: np.ndarray,
    ) -> None:
        payload = SurfaceRecord(
            ticker=ticker,
            timestamp=timestamp,
            strike_axis=json.dumps(strike_axis.tolist()),
            expiry_axis=json.dumps(expiry_axis.tolist()),
            iv_grid=json.dumps(iv_grid.tolist()),
        )
        with self.session_factory() as s:
            s.add(payload)
            s.commit()

    def latest_surface(self, ticker: str) -> dict | None:
        with self.session_factory() as s:
            stmt = (
                select(SurfaceRecord)
                .where(SurfaceRecord.ticker == ticker.upper())
                .order_by(desc(SurfaceRecord.timestamp))
                .limit(1)
            )
            row = s.scalar(stmt)
            if row is None:
                return None
            return {
                "ticker": row.ticker,
                "timestamp": row.timestamp,
                "strike_axis": json.loads(row.strike_axis),
                "expiry_axis": json.loads(row.expiry_axis),
                "iv_grid": json.loads(row.iv_grid),
            }

    def surface_history(self, ticker: str, limit: int = 100) -> list[dict]:
        with self.session_factory() as s:
            stmt = (
                select(SurfaceRecord)
                .where(SurfaceRecord.ticker == ticker.upper())
                .order_by(desc(SurfaceRecord.timestamp))
                .limit(limit)
            )
            rows = s.scalars(stmt).all()
            return [
                {
                    "ticker": r.ticker,
                    "timestamp": r.timestamp,
                    "strike_axis": json.loads(r.strike_axis),
                    "expiry_axis": json.loads(r.expiry_axis),
                    "iv_grid": json.loads(r.iv_grid),
                }
                for r in rows
            ]

    def save_prediction(self, ticker: str, model_name: str, horizon: int, prediction_grid: np.ndarray) -> None:
        record = PredictionRecord(
            ticker=ticker.upper(),
            model_name=model_name,
            horizon=horizon,
            prediction_grid=json.dumps(prediction_grid.tolist()),
        )
        with self.session_factory() as s:
            s.add(record)
            s.commit()

    def latest_prediction(self, ticker: str) -> dict | None:
        with self.session_factory() as s:
            stmt = (
                select(PredictionRecord)
                .where(PredictionRecord.ticker == ticker.upper())
                .order_by(desc(PredictionRecord.created_at))
                .limit(1)
            )
            row = s.scalar(stmt)
            if row is None:
                return None
            return {
                "ticker": row.ticker,
                "model_name": row.model_name,
                "horizon": row.horizon,
                "created_at": row.created_at,
                "prediction_grid": json.loads(row.prediction_grid),
            }

    def save_metrics(self, ticker: str, model_name: str, metrics: dict[str, float]) -> None:
        row = MetricRecord(
            ticker=ticker.upper(),
            model_name=model_name,
            rmse=float(metrics["rmse"]),
            mae=float(metrics["mae"]),
            directional_skew_accuracy=float(metrics["directional_skew_accuracy"]),
            surface_similarity=float(metrics["surface_similarity"]),
        )
        with self.session_factory() as s:
            s.add(row)
            s.commit()
