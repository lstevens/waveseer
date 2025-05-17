import json
import pytest
from fastapi.testclient import TestClient
from wave.ingest import ws_app

pytestmark = pytest.mark.integration

client = TestClient(ws_app)


def test_websocket_ping_pong_and_reconnect():
    # First connection
    with client.websocket_connect("/ws/ingest") as ws1:
        msg1 = ws1.receive_json()
        assert msg1.get("type") == "connection_established"
        assert "client_id" in msg1
        cid1 = msg1["client_id"]

        # Send ping, expect pong
        ws1.send_json({"type": "ping"})
        pong = ws1.receive_json()
        assert pong.get("type") == "pong"

    # After close, reconnect
    with client.websocket_connect("/ws/ingest") as ws2:
        msg2 = ws2.receive_json()
        assert msg2.get("type") == "connection_established"
        assert msg2.get("client_id") != cid1

        # Ensure ping still works
        ws2.send_json({"type": "ping"})
        pong2 = ws2.receive_json()
        assert pong2.get("type") == "pong"
