from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import regression_metrics
from src.training.losses import SurfaceLoss, SurfaceLossConfig
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    early_stopping_patience: int = 8
    checkpoint_dir: str = "artifacts/models"
    experiment_name: str = "neural_vol_surface"
    track_with_mlflow: bool = False
    grad_clip: float = 1.0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig | None = None,
        loss_cfg: SurfaceLossConfig | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg or TrainConfig()
        self.loss_fn = SurfaceLoss(loss_cfg)
        self.optimizer = AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
        self.checkpoint_dir = Path(self.cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.cfg.device)

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        losses: list[float] = []

        for x, y in tqdm(loader, disable=False):
            x = x.to(self.cfg.device)
            y = y.to(self.cfg.device)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                pred = self.model(x)
                loss = self.loss_fn(pred, y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses))

    def fit(self) -> dict[str, Any]:
        best_val = float("inf")
        patience = 0
        best_ckpt = self.checkpoint_dir / "best.pt"
        history: list[dict[str, float]] = []

        if self.cfg.track_with_mlflow and mlflow is not None:
            mlflow.set_experiment(self.cfg.experiment_name)
            mlflow.start_run()

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step(val_loss)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            history.append(row)
            LOGGER.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

            if self.cfg.track_with_mlflow and mlflow is not None:
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save(self.model.state_dict(), best_ckpt)
            else:
                patience += 1
                if patience >= self.cfg.early_stopping_patience:
                    LOGGER.info("Early stopping triggered.")
                    break

        self.model.load_state_dict(torch.load(best_ckpt, map_location=self.cfg.device))
        if self.cfg.track_with_mlflow and mlflow is not None:
            mlflow.end_run()

        return {"history": history, "best_val_loss": best_val, "checkpoint": str(best_ckpt)}

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        preds: list[np.ndarray] = []
        trues: list[np.ndarray] = []
        for x, y in test_loader:
            x = x.to(self.cfg.device)
            pred = self.model(x).cpu().numpy()
            preds.append(pred)
            trues.append(y.numpy())
        pred_arr = np.concatenate(preds, axis=0)
        true_arr = np.concatenate(trues, axis=0)
        return regression_metrics(true_arr, pred_arr)
