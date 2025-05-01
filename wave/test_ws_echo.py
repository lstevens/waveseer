import pytest
from fastapi.testclient import TestClient
from wave.ingest import ws_app

client = TestClient(ws_app)

def test_ws_ingest_echo():
    payload = {"ts_start": "test", "tf": "1m", "pattern_id": "abc123", "score": 0.75}
    with client.websocket_connect("/ws/ingest") as websocket:
        websocket.send_json(payload)
        data = websocket.receive_json()
        assert data == payload
