import json
import pytest
from wave.ingest import ws_app
from fastapi.testclient import TestClient


def test_ws_match_success(monkeypatch):
    # Mock HTTP match API
    def fake_post(url, json):
        class R:
            status_code = 200
            def json(self): return {"pattern_id": "pX", "score": 0.9}
        return R()
    import wave.ingest as ingest_module
    monkeypatch.setattr(ingest_module.requests, 'post', fake_post)

    client = TestClient(ws_app)
    with client.websocket_connect("/ws/match") as ws:
        ws.send_text(json.dumps({"foo": "bar"}))
        result = ws.receive_json()
        assert result["pattern_id"] == "pX"
        assert result["score"] == 0.9


def test_ws_match_error(monkeypatch):
    # Mock HTTP to return server error
    def fake_post(url, json):
        class R:
            status_code = 503
            def json(self): raise ValueError
        return R()
    import wave.ingest as ingest_module
    monkeypatch.setattr(ingest_module.requests, 'post', fake_post)

    client = TestClient(ws_app)
    with client.websocket_connect("/ws/match") as ws:
        ws.send_text(json.dumps({"bad": True}))
        result = ws.receive_json()
        assert result == {"error": 503}
