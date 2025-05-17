import json
from wave.ingest import ws_app
from fastapi.testclient import TestClient


def test_ws_ingest_echo():
    client = TestClient(ws_app)
    with client.websocket_connect("/ws/ingest") as ws:
        # Drop initial connection established message
        _ = ws.receive_json()
        # Post a dummy PatternHit
        payload = {"ts_start": "t", "tf": "1m", "pattern_id": "x", "score": 1.0}
        resp = client.post("/stream", json=payload)
        assert resp.status_code == 200
        data = ws.receive_json()
        assert data == payload


def test_ws_match_forward(monkeypatch):
    # Mock pattern API response
    def fake_post(url, json):
        class R:
            status_code = 200
            def json(self): return {"pattern_id": "p", "score": 0.42}
        return R()
    import wave.ingest as ingest_module
    monkeypatch.setattr(ingest_module.requests, 'post', fake_post)

    client = TestClient(ws_app)
    with client.websocket_connect("/ws/match") as ws:
        ws.send_text(json.dumps({"foo": "bar"}))
        # Receive forwarded result
        result = ws.receive_json()
        assert result["pattern_id"] == "p"
        assert result["score"] == 0.42
