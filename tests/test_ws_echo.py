from datetime import datetime
from wave.ingest import ws_app
from fastapi.testclient import TestClient


def test_ws_echo_connection_established():
    client = TestClient(ws_app)
    with client.websocket_connect("/ws/echo") as ws:
        msg = ws.receive_json()
        assert msg.get("type") == "connection_established"
        assert "client_id" in msg


def test_ws_echo_ping_pong():
    client = TestClient(ws_app)
    with client.websocket_connect("/ws/echo") as ws:
        _ = ws.receive_json()
        ws.send_json({"type": "ping"})
        msg = ws.receive_json()
        assert msg.get("type") == "pong"
        # timestamp should be valid ISO format
        datetime.fromisoformat(msg.get("timestamp"))


def test_ws_echo_echo_message():
    client = TestClient(ws_app)
    with client.websocket_connect("/ws/echo") as ws:
        _ = ws.receive_json()
        payload = {"foo": "bar", "num": 123}
        ws.send_json(payload)
        msg = ws.receive_json()
        assert msg == payload
