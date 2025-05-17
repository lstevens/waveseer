import json
import subprocess
import time
from pathlib import Path
import pytest


def test_collect_csvs(tmp_path):
    """Verify CSV discovery script produces an index JSON."""
    data_dir = Path('data')
    if not data_dir.exists():
        pytest.skip('No data directory found; place CSV files in ./data/')
    out = tmp_path / 'index.json'
    subprocess.check_call([
        'python3', 'scripts/collect_csvs.py',
        '--data-dir', str(data_dir),
        '--output', str(out)
    ])
    assert out.exists(), 'Index JSON not created'
    entries = json.loads(out.read_text())
    assert isinstance(entries, list) and entries, 'No CSV entries discovered'


def test_replay_data_smoke(tmp_path):
    """Smoke-test the replay_data script against the ingest server."""
    index = tmp_path / 'index.json'
    if not index.exists():
        pytest.skip('Index JSON not generated; run test_collect_csvs first')
    # Start ingest server on a free port
    port = 8010
    cwd = Path(__file__).parent.parent
    proc = subprocess.Popen([
        'uvicorn', 'wave.ingest:ws_app',
        '--host', '127.0.0.1', f'--port', str(port), '--lifespan', 'off'
    ], cwd=str(cwd))
    time.sleep(2)
    try:
        # Replay data at 100x speed
        subprocess.check_call([
            'python3', 'scripts/replay_data.py',
            '--index', str(index),
            '--stream_url', f'http://127.0.0.1:{port}/stream',
            '--speed', '100'
        ], cwd=str(cwd))
    finally:
        proc.terminate()
        proc.wait()
