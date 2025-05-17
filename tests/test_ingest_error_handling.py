import logging
import pytest
from fastapi.testclient import TestClient

from wave.ingest import ws_app, manager
from wave.crypto_heatmap.pipeline import PatternPipeline

client = TestClient(ws_app)

@pytest.fixture(autouse=True)
def isolate_error(monkeypatch):
    # Simulate pipeline.run raising
    def raise_err(self, symbol, timeframe, start, end):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(PatternPipeline, "run", raise_err)

    # Capture broadcasts
    broadcasts = []
    async def fake_async_broadcast(payload):
        broadcasts.append(payload)
    monkeypatch.setattr(manager, "broadcast", fake_async_broadcast)

    return broadcasts


def test_stream_event_pipeline_exception(isolate_error, caplog):
    caplog.set_level(logging.ERROR)
    payload = {
        "ts_start": "2025-05-04T11:00:00Z",
        "tf": "1h",
        "pattern_id": "error_case_pattern",
        "score": 0.75
    }
    response = client.post("/stream", json=payload)
    assert response.status_code == 200
    # No broadcasts on error
    assert isolate_error == []
    # Exception logged
    assert any("PatternPipeline error" in record.getMessage() for record in caplog.records)
