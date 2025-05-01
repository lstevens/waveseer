import pytest
from starlette.testclient import TestClient
from wave.api.app import app

@pytest.mark.asyncio
async def test_ws_streaming():
    client = TestClient(app)
    # connect websocket
    with client.websocket_connect("/ws/patterns") as websocket:
        # send a stream event
        payload = {
            "ts_start": "2025-04-29T00:00:00Z",
            "tf": "1m",
            "pattern_id": "BTCUSDT_1m_w1_c0",
            "score": 0.1,
        }
        res = client.post("/stream", json=payload)
        assert res.status_code == 200
        data = websocket.receive_json()
        assert data == payload
