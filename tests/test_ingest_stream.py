import pytest
from fastapi.testclient import TestClient
from wave.ingest import ws_app, manager

client = TestClient(ws_app)

@pytest.fixture(autouse=True)
def isolate_broadcast(monkeypatch):
    # Capture broadcast calls
    calls = []
    async def fake_broadcast(event):
        calls.append(event)
    monkeypatch.setattr(manager, 'broadcast', fake_broadcast)
    return calls


def test_stream_missing_field(isolate_broadcast):
    # Missing 'score' field for PatternHit
    payload = {
        "ts_start": "2025-05-04T10:00:00Z",
        "tf": "5m",
        "pattern_id": "incomplete_pattern"
        # "score": 0.5  # Missing score
    }
    response = client.post("/stream", json=payload)
    assert response.status_code == 422
    # broadcast should not be called
    assert isolate_broadcast == []


def test_stream_invalid_format(isolate_broadcast):
    # Invalid 'score' type for PatternHit
    payload = {
        "ts_start": "2025-05-04T10:00:00Z",
        "tf": "1m",
        "pattern_id": "bad_score_pattern",
        "score": "not-a-float"  # Invalid type for score
    }
    response = client.post("/stream", json=payload)
    assert response.status_code == 422
    assert isolate_broadcast == []


def test_stream_valid_payload(isolate_broadcast):
    payload = {
        "ts_start": "2025-05-04T10:00:00Z",
        "tf": "1m",
        "pattern_id": "valid_pattern_test",
        "score": 0.85
    }
    response = client.post("/stream", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    # broadcast called once with the enriched event
    assert len(isolate_broadcast) == 1

    broadcast_event = isolate_broadcast[0]
    # Check that original payload fields are in the broadcast_event
    assert broadcast_event["ts_start"] == payload["ts_start"]
    assert broadcast_event["tf"] == payload["tf"]
    assert broadcast_event["pattern_id"] == payload["pattern_id"]
    assert broadcast_event["score"] == payload["score"]
    # Check for enrichment fields
    assert "received_at" in broadcast_event
    assert "source_ip" in broadcast_event
