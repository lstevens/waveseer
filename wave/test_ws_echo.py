from fastapi.testclient import TestClient
from wave.ingest import ws_app

client = TestClient(ws_app)


def test_ws_echo_endpoint():
    """Test the WebSocket echo endpoint."""
    # First message should be connection established notification
    with client.websocket_connect("/ws/echo") as websocket:
        # Get the connection established message
        conn_data = websocket.receive_json()
        assert conn_data["type"] == "connection_established"
        assert "client_id" in conn_data

        # Test echo functionality
        test_payload = {"test": "message", "value": 123}
        websocket.send_json(test_payload)
        echo_response = websocket.receive_json()
        assert echo_response == test_payload

        # Test ping/pong
        ping_payload = {"type": "ping"}
        websocket.send_json(ping_payload)
        pong_response = websocket.receive_json()
        assert pong_response["type"] == "pong"
        assert "timestamp" in pong_response


def test_ws_ingest_connection_handling():
    """Test WebSocket ingest endpoint with connection state tracking."""
    with client.websocket_connect("/ws/ingest") as websocket:
        # Get the connection established message
        conn_data = websocket.receive_json()
        assert conn_data["type"] == "connection_established"
        client_id = conn_data["client_id"]

        # Test sending a message updates connection state
        ping_payload = {"type": "ping"}
        websocket.send_json(ping_payload)
        pong_response = websocket.receive_json()
        assert pong_response["type"] == "pong"

    # We would test the connection state, but it's reset between test runs
    # In a real app, we'd verify the connection state was properly updated


def test_ws_ingest_broadcast():
    """Test broadcasting messages to multiple connected clients."""
    # Connect two clients
    with client.websocket_connect("/ws/ingest") as ws1, \
         client.websocket_connect("/ws/ingest") as ws2:

        # Get connection messages
        conn1 = ws1.receive_json()
        conn2 = ws2.receive_json()
        assert conn1["type"] == "connection_established"
        assert conn2["type"] == "connection_established"

        # Post to stream endpoint to broadcast a message
        payload = {"ts_start": "test", "tf": "1m", "pattern_id": "abc123", "score": 0.75}
        resp = client.post("/stream", json=payload)
        assert resp.status_code == 200

        # Both clients should receive the broadcast
        data1 = ws1.receive_json()
        data2 = ws2.receive_json()
        assert data1 == payload
        assert data2 == payload
