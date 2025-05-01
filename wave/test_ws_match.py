import pytest
import json
from fastapi.testclient import TestClient

from wave.ingest import ws_app

client = TestClient(ws_app)

@ pytest.fixture(autouse=True)
def mock_match(monkeypatch):
    # Mock requests.post to return a fake response
    class FakeResponse:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data
        def json(self):
            return self._data
        @property
        def text(self):
            return json.dumps(self._data)

    def fake_post(url, json):
        assert "/match" in url
        return FakeResponse(200, {"result": "ok", "input": json})

    monkeypatch.setattr("wave.ingest.requests.post", fake_post)


def test_ws_match():
    payload = {"foo": "bar"}
    with client.websocket_connect("/ws/match") as ws:
        ws.send_json(payload)
        data = ws.receive_json()
        assert data == {"result": "ok", "input": payload}
