import pytest
from fastapi.testclient import TestClient
from wave.ingest import ws_app

pytestmark = pytest.mark.integration


def test_ui_ws_ingest_echo():
    client = TestClient(ws_app)
    with client.websocket_connect("/ws/ingest") as ws:
        # Drop initial handshake message
        _ = ws.receive_json()
        payload = {"ts_start": "2025-01-01T00:00:00Z", "tf": "1m", "symbol": "btc", "pattern_id": "p1", "score": 0.5}
        ws.send_json(payload)
        resp = ws.receive_json()
        # Should include status and original fields
        assert resp["status"] == "received"
        for k, v in payload.items():
            assert resp[k] == v
