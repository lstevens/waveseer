import asyncio
import json
import subprocess
import time
from pathlib import Path

import pytest
import websockets


def start_process(cmd, cwd):
    return subprocess.Popen(cmd, cwd=str(cwd))

@pytest.mark.asyncio
async def test_detection_on_sample():
    # Ensure sample index exists
    idx = Path('data/sample_index.json')
    if not idx.exists():
        pytest.skip('Sample index missing; run scripts/sample_data.py first')

    cwd = Path(__file__).parent.parent
    # Start ingest server
    port = 8011
    ingest_cmd = [
        'uvicorn', 'wave.ingest:ws_app',
        '--host', '127.0.0.1', '--port', str(port), '--lifespan', 'off'
    ]
    ingest_proc = start_process(ingest_cmd, cwd)
    time.sleep(2)

    # Connect to ingest WebSocket
    uri = f'ws://127.0.0.1:{port}/ws/ingest'
    async with websockets.connect(uri) as ws:
        # Start replaying sample data
        replay_cmd = [
            'python3', 'scripts/replay_data.py',
            '--index', 'data/sample_index.json',
            '--stream_url', f'http://127.0.0.1:{port}/stream',
            '--speed', '1000'
        ]
        replay_proc = start_process(replay_cmd, cwd)
        try:
            # Wait for at least one event
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
            event = json.loads(msg)
            assert 'ts_start' in event and 'open' in event, f"Unexpected event: {event}"
        finally:
            replay_proc.terminate()
            replay_proc.wait()

    ingest_proc.terminate()
    ingest_proc.wait()
