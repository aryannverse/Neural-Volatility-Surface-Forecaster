from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_skew_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_skew = np.median(np.gradient(y_true, axis=2), axis=(1, 2))
    pred_skew = np.median(np.gradient(y_pred, axis=2), axis=(1, 2))
    return float(np.mean(np.sign(true_skew) == np.sign(pred_skew)))


def surface_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Cosine similarity on flattened surfaces.
    """
    t = y_true.reshape(y_true.shape[0], -1)
    p = y_pred.reshape(y_pred.shape[0], -1)
    denom = np.linalg.norm(t, axis=1) * np.linalg.norm(p, axis=1)
    sim = np.sum(t * p, axis=1) / np.maximum(denom, 1e-12)
    return float(np.mean(sim))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "directional_skew_accuracy": directional_skew_accuracy(y_true, y_pred),
        "surface_similarity": surface_similarity(y_true, y_pred),
    }
