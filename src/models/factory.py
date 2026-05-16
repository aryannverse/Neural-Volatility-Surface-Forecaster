from __future__ import annotations

from src.models.autoencoder_forecaster import AutoencoderForecaster
from src.models.base import ModelConfig
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.conv3d_model import Conv3DSurfaceForecaster
from src.models.gru_model import GRUSurfaceForecaster
from src.models.lstm_model import LSTMSurfaceForecaster
from src.models.transformer_model import TransformerSurfaceForecaster


def build_model(name: str, cfg: ModelConfig):
    key = name.lower()
    if key == "lstm":
        return LSTMSurfaceForecaster(cfg)
    if key == "gru":
        return GRUSurfaceForecaster(cfg)
    if key in {"cnn_lstm", "cnnlstm"}:
        return CNNLSTMForecaster(cfg)
    if key == "transformer":
        return TransformerSurfaceForecaster(cfg)
    if key in {"conv3d", "neural_operator"}:
        return Conv3DSurfaceForecaster(cfg)
    if key in {"autoencoder", "ae"}:
        return AutoencoderForecaster(cfg)
    raise ValueError(f"Unknown model type: {name}")
