import requests
import pytest
import subprocess
import time
from playwright.sync_api import sync_playwright

# Initial and post-reconnect events
test_event1 = {
    "ts_start": "2025-05-01T08:01:00Z",
    "tf": "1m",
    "pattern_id": "reconnect-test-1",
    "score": 0.11
}
test_event2 = {
    "ts_start": "2025-05-01T08:02:00Z",
    "tf": "1m",
    "pattern_id": "reconnect-test-2",
    "score": 0.22
}

@pytest.fixture(scope="session", autouse=True)
def server():
    # Start ingest server for reconnect E2E on port 8001
    proc = subprocess.Popen(
        ["uvicorn", "wave.ingest:ws_app",
         "--host", "127.0.0.1", "--port", "8001", "--lifespan", "off"],
        cwd="/Users/imacpro/Desktop/Dev/Waveseer"
    )
    time.sleep(2)
    yield
    proc.terminate()
    proc.wait()


def test_ui_ws_reconnect():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        # Navigate to events UI and ensure WS connects
        page.goto("http://localhost:8001")
        page.wait_for_selector("#status:has-text('Connected')", timeout=5000)
        # Ensure WebSocket instance is available
        page.wait_for_function("window._ws !== undefined", timeout=5000)
        # Send first event and verify
        resp = requests.post("http://localhost:8001/stream", json=test_event1)
        assert resp.status_code == 200
        selector1 = f"td:has-text('{test_event1['pattern_id']}')"
        page.wait_for_selector(selector1, timeout=5000)

        # Simulate WebSocket drop
        page.evaluate("window._ws.close()")
        # Expect backoff countdown
        page.wait_for_selector("#status:has-text('Reconnecting in 1s')", timeout=2000)

        # Wait for reconnect (initial delay 1s)
        time.sleep(1.5)
        # After reconnect delay, send second event
        resp2 = requests.post("http://localhost:8001/stream", json=test_event2)
        assert resp2.status_code == 200
        selector2 = f"td:has-text('{test_event2['pattern_id']}')"
        page.wait_for_selector(selector2, timeout=5000)
        browser.close()
