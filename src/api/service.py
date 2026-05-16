from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from src.datasets.surface_dataset import DatasetConfig, create_dataloaders
from src.database.repository import ForecastRepository
from src.ingestion.pipeline import OptionsIngestionPipeline
from src.iv_surface.builder import SurfaceBuilder
from src.models.base import ModelConfig
from src.models.factory import build_model
from src.training.trainer import TrainConfig, Trainer


@dataclass
class TrainResult:
    metrics: dict[str, float]
    best_val_loss: float
    checkpoint: str


class ForecastService:
    def __init__(
        self,
        ingestion: OptionsIngestionPipeline,
        surface_builder: SurfaceBuilder,
        repository: ForecastRepository,
    ):
        self.ingestion = ingestion
        self.surface_builder = surface_builder
        self.repository = repository

    def refresh_current_surface(self, ticker: str) -> dict:
        chain = self.ingestion.pull_snapshot(ticker=ticker)
        snap = self.surface_builder.build_surface(chain)
        self.repository.save_raw_chain(chain)
        self.repository.save_surface(
            ticker=snap.ticker,
            timestamp=snap.timestamp,
            strike_axis=snap.strike_axis,
            expiry_axis=snap.expiry_axis,
            iv_grid=snap.iv_grid,
        )
        return {
            "ticker": snap.ticker,
            "timestamp": snap.timestamp,
            "strike_axis": snap.strike_axis.tolist(),
            "expiry_axis": snap.expiry_axis.tolist(),
            "iv_grid": snap.iv_grid.tolist(),
        }

    def train_model(
        self,
        ticker: str,
        model_name: str,
        lookback: int,
        horizon: int,
        epochs: int,
        batch_size: int,
    ) -> TrainResult:
        history = self.repository.surface_history(ticker=ticker, limit=2000)
        if len(history) < lookback + horizon + 10:
            raise ValueError(
                f"Need at least {lookback + horizon + 10} stored surfaces for training, have {len(history)}."
            )
        history_sorted = sorted(history, key=lambda x: x["timestamp"])
        surfaces = np.asarray([h["iv_grid"] for h in history_sorted], dtype=np.float32)

        ds_cfg = DatasetConfig(lookback=lookback, horizon=horizon, batch_size=batch_size)
        train_loader, val_loader, test_loader = create_dataloaders(surfaces, ds_cfg)

        mc = ModelConfig(
            lookback=lookback,
            expiry_dim=surfaces.shape[1],
            strike_dim=surfaces.shape[2],
        )
        model = build_model(model_name, mc)
        trainer = Trainer(model, train_loader, val_loader, TrainConfig(epochs=epochs))
        fit_result = trainer.fit()
        metrics = trainer.evaluate(test_loader)
        self.repository.save_metrics(ticker=ticker, model_name=model_name, metrics=metrics)

        # Create a live forecast from latest lookback window.
        x = surfaces[-lookback:]
        pred = model(torch.from_numpy(x[None, ...]).to(next(model.parameters()).device)).detach().cpu().numpy()[0]
        self.repository.save_prediction(ticker=ticker, model_name=model_name, horizon=horizon, prediction_grid=pred)
        return TrainResult(
            metrics=metrics,
            best_val_loss=float(fit_result["best_val_loss"]),
            checkpoint=str(fit_result["checkpoint"]),
        )
