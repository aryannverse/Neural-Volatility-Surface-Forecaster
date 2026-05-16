from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text

# Ensure project root is importable when Streamlit runs from dashboard/ context.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import ForecastService
from src.database import ForecastRepository, get_session_factory, init_db
from src.features.engineer import FeatureEngineer
from src.ingestion.pipeline import OptionsIngestionPipeline
from src.ingestion.yfinance_provider import YFinanceOptionsProvider, YFinanceProviderConfig
from src.iv_surface.builder import SurfaceBuilder
from src.preprocessing.regime import RegimeTagger
from src.utils.paths import RAW_DIR
from src.visualization.plotters import (
    plot_surface_3d,
    plot_surface_heatmap,
    plot_surface_sequence_animation,
    plot_term_structure_slice,
)


@st.cache_resource
def get_service() -> ForecastService:
    init_db()
    repo = ForecastRepository(get_session_factory())
    provider = YFinanceOptionsProvider(YFinanceProviderConfig(cache_dir=RAW_DIR))
    ingestion = OptionsIngestionPipeline(provider=provider, raw_dir=RAW_DIR)
    return ForecastService(ingestion=ingestion, surface_builder=SurfaceBuilder(), repository=repo)


def page_live_options_chain(service: ForecastService, ticker: str) -> None:
    st.subheader("Live Options Chain")
    chain = service.ingestion.pull_snapshot(ticker=ticker)
    st.dataframe(chain.head(300), use_container_width=True)
    st.metric("Contracts", f"{len(chain):,}")
    st.metric("Median IV", f"{chain['iv'].median():.2%}")


def page_current_surface(service: ForecastService, ticker: str) -> None:
    st.subheader("Current IV Surface")
    data = service.refresh_current_surface(ticker=ticker)
    strike = np.asarray(data["strike_axis"], dtype=float)
    expiry = np.asarray(data["expiry_axis"], dtype=float)
    grid = np.asarray(data["iv_grid"], dtype=float)
    st.plotly_chart(plot_surface_3d(strike, expiry, grid, f"{ticker} Current Surface"), use_container_width=True)
    st.plotly_chart(plot_surface_heatmap(strike, expiry, grid, f"{ticker} Surface Heatmap"), use_container_width=True)


def page_history_playback(service: ForecastService, ticker: str) -> None:
    st.subheader("Historical Surface Playback")
    hist = service.repository.surface_history(ticker=ticker, limit=120)
    if not hist:
        st.info("No stored surfaces yet. Refresh current surface first.")
        return
    hist = sorted(hist, key=lambda x: x["timestamp"])
    strike = np.asarray(hist[0]["strike_axis"], dtype=float)
    expiry = np.asarray(hist[0]["expiry_axis"], dtype=float)
    surfaces = np.asarray([x["iv_grid"] for x in hist], dtype=float)
    st.plotly_chart(
        plot_surface_sequence_animation(strike_axis=strike, expiry_axis=expiry, surfaces=surfaces),
        use_container_width=True,
    )


def page_forecast(service: ForecastService, ticker: str) -> None:
    st.subheader("Forecasted Surface")
    row = service.repository.latest_prediction(ticker=ticker)
    latest = service.repository.latest_surface(ticker=ticker)
    if row is None or latest is None:
        st.info("Need both latest surface and forecast. Run training first.")
        return
    strike = np.asarray(latest["strike_axis"], dtype=float)
    expiry = np.asarray(latest["expiry_axis"], dtype=float)
    pred = np.asarray(row["prediction_grid"], dtype=float)
    actual = np.asarray(latest["iv_grid"], dtype=float)
    st.plotly_chart(plot_surface_3d(strike, expiry, pred, f"{ticker} Forecast ({row['model_name']})"), use_container_width=True)
    strike_idx = st.slider("Strike Slice Index", 0, len(strike) - 1, len(strike) // 2)
    st.plotly_chart(plot_term_structure_slice(expiry, actual, pred, strike_idx), use_container_width=True)


def page_performance(service: ForecastService, ticker: str) -> None:
    st.subheader("Model Performance")
    with service.repository.session_factory() as s:
        rows = s.execute(
            text(
                "SELECT model_name, run_timestamp, rmse, mae, directional_skew_accuracy, surface_similarity "
                "FROM metric_records WHERE ticker=:ticker ORDER BY run_timestamp DESC LIMIT 200"
            ),
            {"ticker": ticker},
        ).fetchall()
    if not rows:
        st.info("No metrics logged yet.")
        return
    df = pd.DataFrame(rows, columns=["model", "timestamp", "rmse", "mae", "dsa", "sim"])
    st.dataframe(df, use_container_width=True)
    st.line_chart(df.set_index("timestamp")[["rmse", "mae"]])


def page_regime_analysis(service: ForecastService, ticker: str) -> None:
    st.subheader("Regime Analysis")
    hist = service.repository.surface_history(ticker=ticker, limit=300)
    if len(hist) < 30:
        st.info("Collect more historical surfaces to estimate regimes.")
        return
    surfaces = np.asarray([x["iv_grid"] for x in sorted(hist, key=lambda x: x["timestamp"])], dtype=float)
    regimes = RegimeTagger().fit_predict_from_surfaces(surfaces)
    feats = FeatureEngineer.surface_features(surfaces)
    feats["regime"] = regimes.astype(int)
    st.dataframe(feats.tail(100), use_container_width=True)
    st.bar_chart(feats["regime"].value_counts().sort_index())


def main() -> None:
    st.set_page_config(page_title="Neural Vol Surface Forecaster", layout="wide")
    st.title("Neural Volatility Surface Forecaster")
    st.caption("Institutional-grade volatility surface research and forecasting dashboard")

    service = get_service()
    ticker = st.sidebar.text_input("Ticker", value="SPY").upper()
    horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 10, 1)
    _ = horizon  # currently horizon enters API training path, retained for UI consistency
    page = st.sidebar.selectbox(
        "Page",
        [
            "Live options chain",
            "Current IV surface",
            "Historical surface playback",
            "Forecasted surface",
            "Model performance",
            "Regime analysis",
        ],
    )

    if st.sidebar.button("Refresh Data"):
        service.refresh_current_surface(ticker)
        st.sidebar.success("Latest surface refreshed.")

    if page == "Live options chain":
        page_live_options_chain(service, ticker)
    elif page == "Current IV surface":
        page_current_surface(service, ticker)
    elif page == "Historical surface playback":
        page_history_playback(service, ticker)
    elif page == "Forecasted surface":
        page_forecast(service, ticker)
    elif page == "Model performance":
        page_performance(service, ticker)
    else:
        page_regime_analysis(service, ticker)


if __name__ == "__main__":
    main()
