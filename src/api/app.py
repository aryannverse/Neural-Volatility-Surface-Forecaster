from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi import WebSocket, WebSocketDisconnect

from src.api.schemas import ForecastResponse, SurfaceResponse, TrainRequest, TrainResponse
from src.api.service import ForecastService
from src.database import ForecastRepository, get_session_factory, init_db
from src.ingestion.pipeline import OptionsIngestionPipeline
from src.ingestion.yfinance_provider import YFinanceOptionsProvider, YFinanceProviderConfig
from src.iv_surface.builder import SurfaceBuilder
from src.utils.paths import RAW_DIR


def build_service() -> ForecastService:
    init_db()
    repo = ForecastRepository(get_session_factory())
    provider = YFinanceOptionsProvider(YFinanceProviderConfig(cache_dir=RAW_DIR))
    ingestion = OptionsIngestionPipeline(provider=provider, raw_dir=RAW_DIR)
    return ForecastService(ingestion=ingestion, surface_builder=SurfaceBuilder(), repository=repo)


def create_app() -> FastAPI:
    app = FastAPI(title="Neural Volatility Surface Forecaster API", version="0.1.0")
    service = build_service()

    @app.get("/surface/current/{ticker}", response_model=SurfaceResponse)
    async def get_surface_current(ticker: str):
        try:
            return service.refresh_current_surface(ticker.upper())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/surface/history/{ticker}", response_model=list[SurfaceResponse])
    async def get_surface_history(ticker: str, limit: int = 100):
        rows = service.repository.surface_history(ticker=ticker.upper(), limit=limit)
        if not rows:
            raise HTTPException(status_code=404, detail=f"No history found for {ticker}")
        return rows

    @app.get("/forecast/{ticker}", response_model=ForecastResponse)
    async def get_forecast(ticker: str):
        row = service.repository.latest_prediction(ticker=ticker.upper())
        if row is None:
            raise HTTPException(status_code=404, detail=f"No forecast found for {ticker}")
        return row

    @app.post("/train", response_model=TrainResponse)
    async def train_model(req: TrainRequest):
        try:
            result = service.train_model(
                ticker=req.ticker.upper(),
                model_name=req.model_name,
                lookback=req.lookback,
                horizon=req.horizon,
                epochs=req.epochs,
                batch_size=req.batch_size,
            )
            return TrainResponse(
                ticker=req.ticker.upper(),
                model_name=req.model_name,
                metrics=result.metrics,
                best_val_loss=result.best_val_loss,
                checkpoint=result.checkpoint,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.websocket("/ws/forecast/{ticker}")
    async def ws_forecast(websocket: WebSocket, ticker: str):
        await websocket.accept()
        try:
            while True:
                row = service.repository.latest_prediction(ticker=ticker.upper())
                await websocket.send_json(
                    row
                    if row is not None
                    else {"ticker": ticker.upper(), "status": "no_forecast", "message": "No forecast available yet."}
                )
                await asyncio.sleep(5.0)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
