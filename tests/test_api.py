from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api.app import create_app


class MockRepo:
    def surface_history(self, ticker: str, limit: int = 100):
        return [
            {
                "ticker": ticker,
                "timestamp": datetime.now(timezone.utc),
                "strike_axis": [0.8, 1.0, 1.2],
                "expiry_axis": [0.1, 0.5, 1.0],
                "iv_grid": [[0.2, 0.21, 0.23], [0.19, 0.2, 0.22], [0.18, 0.19, 0.21]],
            }
        ]

    def latest_prediction(self, ticker: str):
        return {
            "ticker": ticker,
            "model_name": "lstm",
            "horizon": 1,
            "created_at": datetime.now(timezone.utc),
            "prediction_grid": [[0.2, 0.21], [0.19, 0.2]],
        }


class MockService:
    def __init__(self):
        self.repository = MockRepo()

    def refresh_current_surface(self, ticker: str):
        return {
            "ticker": ticker,
            "timestamp": datetime.now(timezone.utc),
            "strike_axis": [0.8, 1.0, 1.2],
            "expiry_axis": [0.1, 0.5, 1.0],
            "iv_grid": [[0.2, 0.21, 0.23], [0.19, 0.2, 0.22], [0.18, 0.19, 0.21]],
        }

    def train_model(self, **kwargs):
        class R:
            metrics = {"rmse": 0.01, "mae": 0.005, "directional_skew_accuracy": 0.8, "surface_similarity": 0.9}
            best_val_loss = 0.02
            checkpoint = "artifacts/models/best.pt"

        _ = kwargs
        return R()


def test_api_endpoints(monkeypatch):
    monkeypatch.setattr("src.api.app.build_service", lambda: MockService())
    app = create_app()
    client = TestClient(app)

    r1 = client.get("/surface/current/SPY")
    assert r1.status_code == 200

    r2 = client.get("/surface/history/SPY")
    assert r2.status_code == 200

    r3 = client.get("/forecast/SPY")
    assert r3.status_code == 200

    r4 = client.post("/train", json={"ticker": "SPY", "model_name": "lstm"})
    assert r4.status_code == 200
