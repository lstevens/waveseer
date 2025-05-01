import time
import requests
import pytest
import subprocess
import time
from playwright.sync_api import sync_playwright

TEST_EVENT = {
    "ts_start": "2025-05-01T08:00:00Z",
    "tf": "1m",
    "pattern_id": "e2e-test-pattern",
    "score": 0.99
}

@ pytest.fixture(scope="session", autouse=True)
def server():
    # Start ingest server (serves UI)
    proc = subprocess.Popen([
        "uvicorn", "wave.ingest:ws_app",
        "--host", "0.0.0.0", "--port", "8000"
    ], cwd="/Users/imacpro/Desktop/Dev/Waveseer")
    time.sleep(2)
    yield
    proc.terminate()
    proc.wait()

def test_ui_e2e():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8000")
        # Give WebSocket time to connect
        time.sleep(1)
        # Send test event
        resp = requests.post("http://localhost:8000/stream", json=TEST_EVENT)
        assert resp.status_code == 200
        # Wait for event to appear in UI
        selector = f"td:has-text('{TEST_EVENT['pattern_id']}')"
        page.wait_for_selector(selector, timeout=5000)
        assert TEST_EVENT['pattern_id'] in page.inner_text("#events")
        browser.close()
