"""
Basic WebSocket test client for ingestion endpoint.
Usage:
  pip install websockets
  python wave/test_ws.py
"""
import asyncio
import json
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    exit(1)

async def main():
    uri = 'ws://localhost:8000/ws/ingest'
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        print("Connected")
        payload = {
            'ts_start': 'test',
            'tf': '1m',
            'pattern_id': 'abc123',
            'score': 0.75
        }
        msg = json.dumps(payload)
        print(f"Sending: {msg}")
        await ws.send(msg)
        response = await ws.recv()
        print(f"Received: {response}")

if __name__ == '__main__':
    asyncio.run(main())
