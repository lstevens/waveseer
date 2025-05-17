import yaml
import polars as pl
import json
import pytest
from typer.testing import CliRunner
from wave.seer import app
import yaml
import shutil


class DummyResponse:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data or {}
    def json(self):
        return self._data

@pytest.fixture(autouse=True)
def temp_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # write config
    cfg = {"symbols": ["SYM"], "timeframes": [{"tf": "1m", "windows": [2]}]}
    (tmp_path / "config.yml").write_text(yaml.dump(cfg))
    # write parquet
    df = pl.DataFrame({"datetime": ["2025-04-29T00:00:00Z", "2025-04-29T00:01:00Z"],
                       "close": [1.0, 2.0]})
    out_dir = tmp_path / "build" / "cache" / "SYM"
    out_dir.mkdir(parents=True)
    df.write_parquet(str(out_dir / "1m.parquet"))
    yield


def test_seer_cli_without_stream(monkeypatch, capsys):
    calls = []
    def fake_post(url, json=None):
        calls.append((url, json))
        return DummyResponse(200, {"pattern_id": "p1", "score": 0.5})
    monkeypatch.setattr("wave.seer.requests.post", fake_post)
    runner = CliRunner()
    result = runner.invoke(app, ["--symbol", "SYM", "--tf", "1m", "--api-url", "http://api"])
    assert result.exit_code == 0
    # one match call, no stream
    assert any("/match" in url for url, _ in calls)
    assert all("/stream" not in url for url, _ in calls)
    # output JSON printed
    output = result.stdout.strip().splitlines()[0]
    evt = json.loads(output)
    assert evt["pattern_id"] == "p1"
    assert evt["score"] == 0.5


def test_seer_cli_with_stream(monkeypatch):
    calls = []
    def fake_post(url, json=None):
        calls.append((url, json))
        if url.endswith("/match"):  # API match
            return DummyResponse(200, {"pattern_id": "p2", "score": 0.8})
        else:  # stream endpoint
            return DummyResponse(200)
    monkeypatch.setattr("wave.seer.requests.post", fake_post)
    runner = CliRunner()
    result = runner.invoke(app, ["--symbol", "SYM", "--tf", "1m",
                                 "--api-url", "http://api",
                                 "--stream-url", "http://api"])
    assert result.exit_code == 0
    # expect both match and stream calls
    match_calls = [c for c in calls if "/match" in c[0]]
    stream_calls = [c for c in calls if "/stream" in c[0]]
    assert match_calls
    assert stream_calls
    # printed event corresponds to stream payload
    printed = result.stdout.strip().splitlines()[0]
    payload = json.loads(printed)
    assert payload["pattern_id"] == "p2"
    assert payload["score"] == 0.8


def test_seer_cli_error_on_missing_cache(tmp_path, monkeypatch):
    # Simulate missing build/cache layout
    # Remove directories created by autouse fixture
    shutil.rmtree(tmp_path / "build", ignore_errors=True)
    # Write only config
    (tmp_path / "config.yml").write_text(yaml.dump({"symbols": ["SYM"], "timeframes": [{"tf": "1m", "windows": [2]}]}))
    # Run CLI
    runner = CliRunner()
    result = runner.invoke(app, ["--symbol", "SYM", "--tf", "1m", "--api-url", "http://api"])
    # Should fail due to missing cache data
    assert result.exit_code != 0
    assert "build/cache" in result.stdout.lower()
