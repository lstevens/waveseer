from fastapi.testclient import TestClient

from wave.ingest import ws_app

client = TestClient(ws_app)


def test_stream_broadcast():
    payload = {"ts_start": "broadcast_test", "tf": "5m", "pattern_id": "xyz789", "score": 0.33}
    with client.websocket_connect("/ws/ingest") as websocket:
        # First get the connection established message
        conn_msg = websocket.receive_json()
        assert conn_msg["type"] == "connection_established"
        assert "client_id" in conn_msg

        # Now post the stream event
        response = client.post("/stream", json=payload)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        # Then receive the broadcast message
        data = websocket.receive_json()
        assert data == payload
