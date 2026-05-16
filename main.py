from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.api.service import ForecastService
from src.database import ForecastRepository, get_session_factory, init_db
from src.ingestion.pipeline import IngestionSchedule, OptionsIngestionPipeline
from src.ingestion.yfinance_provider import YFinanceOptionsProvider, YFinanceProviderConfig
from src.iv_surface.builder import SurfaceBuilder
from src.utils.logger import get_logger
from src.utils.paths import RAW_DIR, ensure_dirs

LOGGER = get_logger(__name__)


def build_service() -> ForecastService:
    ensure_dirs()
    init_db()
    repo = ForecastRepository(get_session_factory())
    provider = YFinanceOptionsProvider(YFinanceProviderConfig(cache_dir=RAW_DIR))
    ingestion = OptionsIngestionPipeline(provider=provider, raw_dir=RAW_DIR)
    return ForecastService(ingestion=ingestion, surface_builder=SurfaceBuilder(), repository=repo)


def cmd_ingest(args: argparse.Namespace) -> None:
    service = build_service()
    if args.backfill_days > 0:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.backfill_days)
        schedule = IngestionSchedule(start=start, end=end, frequency=args.frequency)
        hist = service.ingestion.backfill(args.ticker, schedule)
        for ts, frame in hist.groupby(pd.to_datetime(hist["timestamp"])):
            snap = service.surface_builder.build_surface(frame)
            service.repository.save_surface(
                ticker=snap.ticker,
                timestamp=snap.timestamp,
                strike_axis=snap.strike_axis,
                expiry_axis=snap.expiry_axis,
                iv_grid=snap.iv_grid,
            )
        LOGGER.info("Backfill complete for %s rows=%d", args.ticker, len(hist))
    else:
        row = service.refresh_current_surface(args.ticker)
        LOGGER.info("Saved current surface %s at %s", row["ticker"], row["timestamp"])


def cmd_train(args: argparse.Namespace) -> None:
    service = build_service()
    res = service.train_model(
        ticker=args.ticker,
        model_name=args.model,
        lookback=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    LOGGER.info("Training complete. metrics=%s", res.metrics)


def cmd_api(_: argparse.Namespace) -> None:
    subprocess.run(["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"], check=False)


def cmd_dashboard(_: argparse.Namespace) -> None:
    subprocess.run(["streamlit", "run", "dashboard/app.py", "--server.port", "8501"], check=False)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Neural Volatility Surface Forecaster")
    sub = p.add_subparsers(dest="cmd", required=False)

    p_ing = sub.add_parser("ingest", help="Ingest options chain and build current surfaces")
    p_ing.add_argument("--ticker", default="SPY")
    p_ing.add_argument("--backfill-days", type=int, default=0)
    p_ing.add_argument("--frequency", type=str, default="1D")
    p_ing.set_defaults(func=cmd_ingest)

    p_train = sub.add_parser("train", help="Train forecasting model")
    p_train.add_argument("--ticker", default="SPY")
    p_train.add_argument(
        "--model",
        default="transformer",
        choices=["lstm", "gru", "cnn_lstm", "transformer", "autoencoder", "conv3d"],
    )
    p_train.add_argument("--lookback", type=int, default=20)
    p_train.add_argument("--horizon", type=int, default=1)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.set_defaults(func=cmd_train)

    p_api = sub.add_parser("api", help="Run FastAPI server")
    p_api.set_defaults(func=cmd_api)

    p_dash = sub.add_parser("dashboard", help="Run Streamlit dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    return p


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        build_service()
        LOGGER.info("Project initialized. Run `python main.py --help` for commands.")
