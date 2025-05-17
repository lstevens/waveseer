import pytest
from fastapi.testclient import TestClient
from wave.ingest import ws_app

pytestmark = pytest.mark.integration


def test_ui_page_loads():
    client = TestClient(ws_app)
    response = client.get("/")
    assert response.status_code == 200
    text = response.text.lower()
    # Check for main header and events table
    assert "real-time pattern events" in text
    assert '<table id="events"' in text
    # Check for script block initializing WebSocket URL
    assert 'const wsurl' in text or 'ws://' in text
