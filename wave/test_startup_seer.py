import asyncio
import yaml
from pathlib import Path
import wave.ingest as ingest


def test_start_seer_agents(monkeypatch):
    # Fake config.yml content
    fake_cfg = {'timeframes': [{'tf': '1m'}, {'tf': '5m'}]}
    def fake_read_text(self):
        if self.name == 'config.yml':
            return yaml.dump(fake_cfg)
        return ''
    monkeypatch.setattr(ingest.Path, 'read_text', fake_read_text)

    # Fake cache directory symbols
    fake_symbols = [Path('BTC'), Path('ETH')]
    def fake_iterdir(self):
        if self.name == 'cache':
            return fake_symbols
        return []
    monkeypatch.setattr(ingest.Path, 'iterdir', fake_iterdir)

    # Capture subprocess calls
    calls = []
    class DummyPopen:
        def __init__(self, cmd, cwd=None):
            calls.append(cmd)
    monkeypatch.setattr(ingest.subprocess, 'Popen', DummyPopen)

    # Run startup hook
    asyncio.run(ingest.start_seer_agents())

    # Expected calls: one per symbol/tf
    expected = []
    for symbol in ['BTC', 'ETH']:
        for tf in ['1m', '5m']:
            expected.append([
                'python3', '-m', 'wave.seer',
                '--symbol', symbol,
                '--tf', tf,
                '--stream_url', 'http://127.0.0.1:8000/stream'
            ])
    assert calls == expected
