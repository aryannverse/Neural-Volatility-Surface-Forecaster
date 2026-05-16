from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHED_DIR = DATA_DIR / "cached"
MODELS_DIR = ROOT / "artifacts" / "models"
METRICS_DIR = ROOT / "artifacts" / "metrics"


def ensure_dirs() -> None:
    for p in [RAW_DIR, PROCESSED_DIR, CACHED_DIR, MODELS_DIR, METRICS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
