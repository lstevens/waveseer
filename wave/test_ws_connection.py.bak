"""
Tests for WebSocket connection management functionality.

These tests verify:
- Connection state tracking
- Ping/pong health check
- Reconnection behavior
- Error handling
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
from wave.ingest import ws_app, ConnectionManager, WebSocketConnectionState


def test_connection_manager_add_remove():
    """Test adding and removing connections from manager."""
    manager = ConnectionManager()

    # Create mock websockets
    ws1 = MagicMock()
    ws1.client_id = "client1"
    ws2 = MagicMock()
    ws2.client_id = "client2"

    # Add connections
    manager.add_connection(ws1)
    manager.add_connection(ws2)

    assert len(manager.active_connections) == 2
    assert ws1 in manager.active_connections
    assert ws2 in manager.active_connections

    # Remove connection
    manager.remove_connection(ws1)
    assert len(manager.active_connections) == 1
    assert ws1 not in manager.active_connections
    assert ws2 in manager.active_connections


def test_connection_manager_broadcast():
    """Test broadcasting messages to all connections."""
    manager = ConnectionManager()

    # Create mock websockets
    ws1 = MagicMock()
    ws1.client_id = "client1"
    ws2 = MagicMock()
    ws2.client_id = "client2"

    # Add connections
    manager.add_connection(ws1)
    manager.add_connection(ws2)

    # Broadcast message
    message = {"test": "message"}
    manager.broadcast(message)

    # Verify both websockets received the message
    ws1.send_json.assert_called_once_with(message)
    ws2.send_json.assert_called_once_with(message)


def test_connection_manager_broadcast_with_exception():
    """Test broadcasting with client exception handling."""
    manager = ConnectionManager()

    # Create mock websockets
    ws1 = MagicMock()
    ws1.client_id = "client1"
    ws1.send_json.side_effect = WebSocketDisconnect()

    ws2 = MagicMock()
    ws2.client_id = "client2"

    # Add connections
    manager.add_connection(ws1)
    manager.add_connection(ws2)

    # Broadcast message
    message = {"test": "message"}
    manager.broadcast(message)

    # Verify ws1 was removed due to exception
    assert ws1 not in manager.active_connections
    # Verify ws2 still received the message
    ws2.send_json.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_ping_pong():
    """Test ping/pong health check mechanism."""
    client = TestClient(ws_app)

    with patch('wave.ingest.ConnectionManager.ping_client') as mock_ping:
        # Set up the mock to return True for a successful ping
        mock_ping.return_value = True

        with client.websocket_connect("/ws/ingest") as websocket:
            # Connect and get the connection state
            response = websocket.receive_json()
            assert response["type"] == "connection_established"
            assert "client_id" in response

            # Check that the ping would be sent (we're mocking it)
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response["type"] == "pong"


@pytest.mark.asyncio
async def test_connection_state_transitions():
    """Test WebSocket connection state transitions."""
    connection = WebSocketConnectionState(client_id="test123")

    # Initial state should be CONNECTED
    assert connection.state == "CONNECTED"

    # Test CONNECTED -> ACTIVE transition
    connection.mark_active()
    assert connection.state == "ACTIVE"

    # Test ACTIVE -> IDLE transition
    connection.mark_idle()
    assert connection.state == "IDLE"

    # Test reconnect
    connection.reconnect()
    assert connection.state == "CONNECTED"
    assert connection.reconnect_count == 1

    # Test disconnect
    connection.disconnect()
    assert connection.state == "DISCONNECTED"
