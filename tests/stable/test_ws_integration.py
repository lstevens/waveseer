"""
Stable test for WebSocket integration functionality.
This test verifies that the WebSocket server properly integrates with the pattern pipeline
and broadcasts enriched events. This test has been verified to pass consistently.
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import os

# Set testing mode env variable before importing modules
os.environ["TESTING"] = "true"
os.environ["PIPELINE_BYPASS_ENABLED"] = "false"
os.environ["PIPELINE_ECHO_RAW_EVENTS"] = "false"

# Now import the WS app
from wave.ingest import ws_app

client = TestClient(ws_app)

@pytest.fixture(autouse=True)
def mock_pipeline(monkeypatch):
    """Mock the PatternPipeline to avoid real pattern detection during tests."""
    # Track calls to the run method
    calls = []

    # Create a mock class and instance
    class MockPatternPipeline:
        def __init__(self):
            pass  # Skip actual initialization

        def run(self, symbol, timeframe, start, end):
            calls.append((symbol, timeframe, start, end))
            return [
                {"pattern": "alpha", "score": 0.9},
                {"pattern": "beta", "score": 0.8},
            ]

    # Direct replacement in wave.ingest module
    import wave.ingest
    monkeypatch.setattr(wave.ingest, "PatternPipeline", MockPatternPipeline)

    return calls


def test_ws_integration_enriched_events(mock_pipeline, monkeypatch):
    """
    Verify that the WebSocket server properly integrates with the pattern pipeline
    and broadcasts enriched events.
    """
    # Set environment variables to control stream_event behavior
    monkeypatch.setenv("PIPELINE_BYPASS_ENABLED", "false")
    monkeypatch.setenv("PIPELINE_ECHO_RAW_EVENTS", "false")

    # Connect WebSocket
    with client.websocket_connect("/ws/ingest") as websocket:
        msg = websocket.receive_json()  # Get the connection established message
        assert msg["type"] == "connection_established"
        client_id = msg.get("client_id")
        assert isinstance(client_id, str)

        # Send a valid stream event (PatternHit structure)
        start_time = datetime.fromisoformat("2025-05-04T11:00:00+00:00")
        end_time = start_time + timedelta(hours=1)

        # Note: Using the actual PatternHit schema fields that match the model
        payload = {
            "symbol": "BTCUSDT",          # Required by PatternHit schema
            "timeframe": "1h",          # Required by PatternHit schema
            "start": start_time.isoformat(),  # Required by PatternHit schema
            "end": end_time.isoformat()    # Required by PatternHit schema
        }

        response = client.post("/stream", json=payload)
        assert response.status_code == 200

        # Ensure pipeline.run called with correct args derived from PatternHit
        assert len(mock_pipeline) == 1
        sym_arg, tf_arg, st_arg, en_arg = mock_pipeline[0]

        expected_symbol = payload["symbol"]
        expected_tf = payload["timeframe"]
        expected_start_time = datetime.fromisoformat(payload["start"])
        expected_end_time = datetime.fromisoformat(payload["end"])

        assert sym_arg == expected_symbol
        assert tf_arg == expected_tf
        assert st_arg == expected_start_time
        assert en_arg == expected_end_time

        # Receive enriched messages via WebSocket
        try:
            # Use the TestClient's receive_json method
            enriched1 = websocket.receive_json()
            enriched2 = websocket.receive_json()
        except Exception as e:
            pytest.fail(f"Test timed out or WebSocket connection failed: {e}")
            return

        # The messages should contain the pipeline data plus the enriched fields
        assert "pattern" in enriched1
        assert "score" in enriched1
        assert enriched1["pattern"] == "alpha"
        assert enriched1["score"] == 0.9

        assert "pattern" in enriched2
        assert "score" in enriched2
        assert enriched2["pattern"] == "beta"
        assert enriched2["score"] == 0.8
