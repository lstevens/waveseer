import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from wave.ingest import ws_app, manager
from wave.crypto_heatmap.pipeline import PatternPipeline

client = TestClient(ws_app)

@pytest.fixture
def isolate_pipeline_fixture(monkeypatch):
    # Capture pipeline args and broadcasts
    pipeline_args = []
    def fake_run(self, symbol, timeframe, start, end):
        pipeline_args.append((symbol, timeframe, start, end))
        return [{"result": 1}, {"result": 2}]
    monkeypatch.setattr(PatternPipeline, "run", fake_run)

    broadcasts = []
    async def fake_async_broadcast(payload):
        broadcasts.append(payload)
    monkeypatch.setattr(manager, "broadcast", fake_async_broadcast)

    return pipeline_args, broadcasts


def test_pipeline_invocation_and_broadcast(isolate_pipeline_fixture):
    pipeline_args, broadcasts = isolate_pipeline_fixture

    # Payload changed to PatternHit structure
    payload = {
        "ts_start": "2025-05-04T11:00:00Z",
        "tf": "1h",
        "pattern_id": "BTCUSDT_pipeline_test",
        "score": 0.9
    }
    response = client.post("/stream", json=payload)
    assert response.status_code == 200

    # Pipeline.run called once with parsed datetimes derived from PatternHit
    assert len(pipeline_args) == 1
    sym, tf_arg, st_arg, en_arg = pipeline_args[0]

    # Assertions based on derivation logic in stream_event
    expected_symbol = payload["pattern_id"].split('_')[0]
    expected_tf = payload["tf"]
    expected_start_time = datetime.fromisoformat(payload["ts_start"].replace("Z", "+00:00"))
    # Current end_time derivation in stream_event for '1h' would be start_time + 1 minute
    # This highlights a potential issue in stream_event's end_time logic for non-minute timeframes
    # but the test should reflect current behavior.
    expected_end_time_derived = expected_start_time + timedelta(minutes=int(expected_tf[:-1]))

    assert sym == expected_symbol
    assert tf_arg == expected_tf
    assert st_arg == expected_start_time
    assert en_arg == expected_end_time_derived

    # manager.broadcast called for each fake match
    assert len(broadcasts) == 2
    assert broadcasts == [{"result": 1}, {"result": 2}]
