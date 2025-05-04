import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from wave.ingest import ws_app
from wave.crypto_heatmap.pipeline import PatternPipeline

client = TestClient(ws_app)

@pytest.fixture(autouse=True)
def mock_pipeline(monkeypatch):
    calls = []
    def fake_run(self, symbol, timeframe, start, end):
        calls.append((symbol, timeframe, start, end))
        return [
            {"pattern": "alpha", "score": 0.9},
            {"pattern": "beta", "score": 0.8},
        ]
    monkeypatch.setattr(PatternPipeline, "run", fake_run)
    return calls


def test_ws_integration_enriched_events(mock_pipeline):
    # Connect WebSocket
    with client.websocket_connect("/ws/ingest") as websocket:
        msg = websocket.receive_json()
        assert msg["type"] == "connection_established"
        client_id = msg.get("client_id")
        assert isinstance(client_id, str)

        # Send a valid stream event
        payload = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start": "2025-05-04T11:00:00",
            "end": "2025-05-04T12:00:00",
        }
        response = client.post("/stream", json=payload)
        assert response.status_code == 200

        # Ensure pipeline.run called with correct args
        assert len(mock_pipeline) == 1
        sym, tf, st, en = mock_pipeline[0]
        assert sym == payload["symbol"]
        assert tf == payload["timeframe"]
        assert isinstance(st, datetime)
        assert isinstance(en, datetime)

        # Receive enriched messages via WebSocket
        enriched1 = websocket.receive_json()
        enriched2 = websocket.receive_json()
        
        # Check that the enriched messages contain both pattern data and metadata
        assert enriched1["pattern"] == "alpha"
        assert enriched1["score"] == 0.9
        assert enriched1["symbol"] == payload["symbol"]
        assert enriched1["tf"] == payload["timeframe"]
        assert "ts_start" in enriched1
        assert "ts_end" in enriched1
        
        assert enriched2["pattern"] == "beta"
        assert enriched2["score"] == 0.8
        assert enriched2["symbol"] == payload["symbol"]
        assert enriched2["tf"] == payload["timeframe"]
