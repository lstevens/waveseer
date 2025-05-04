import pytest
from fastapi.testclient import TestClient
from wave.ingest import ws_app
import json

@pytest.fixture
def client():
    return TestClient(ws_app)

def test_root_and_static(client):
    # Root index.html
    r = client.get('/')
    assert r.status_code == 200
    assert 'text/html' in r.headers['content-type']
    # Static file
    r2 = client.get('/static/index.html')
    assert r2.status_code == 200
    assert 'text/html' in r2.headers['content-type']

@pytest.mark.asyncio
async def test_ws_match(monkeypatch):
    # Stub requests.post
    class DummyResp:
        status_code = 200
        text = '"ok"'
    import wave.ingest as ingest_mod
    monkeypatch.setattr(ingest_mod.requests, 'post', lambda url, json: DummyResp())
    from websockets.sync.client import connect
    uri = 'ws://localhost:8000/ws/match'
    # run server in TestClient context
    with TestClient(ws_app) as client:
        with client.websocket_connect('/ws/match') as ws:
            ws.send_text(json.dumps({'foo': 'bar'}))
            msg = ws.receive_text()
            assert msg == '"ok"'

def test_stream_broadcast(monkeypatch):
    # Connect two clients and send a stream event
    client = TestClient(ws_app)
    event = {'x': 1}
    with client.websocket_connect('/ws/ingest') as ws1, client.websocket_connect('/ws/ingest') as ws2:
        # First receive the connection established messages
        conn1 = ws1.receive_json()
        conn2 = ws2.receive_json()
        assert conn1['type'] == 'connection_established'
        assert conn2['type'] == 'connection_established'
        
        # Now post the event
        r = client.post('/stream', json=event)
        assert r.status_code == 200
        assert r.json() == {'status': 'ok'}
        
        # Both clients should receive the event
        assert ws1.receive_json() == event
        assert ws2.receive_json() == event

@pytest.mark.asyncio
async def test_ws_ingest_data_valid(client):
    from wave.ingest import ws_app
    with TestClient(ws_app) as c, c.websocket_connect('/ws/ingest-data') as ws:
        payload = {'symbol':'BTCUSD','timestamp':1618300000,'price':54321.0,'volume':1.234}
        ws.send_text(json.dumps(payload))
        resp = ws.receive_json()
        assert resp['status'] == 'received'
        for k, v in payload.items():
            assert resp[k] == v

@pytest.mark.asyncio
async def test_ws_ingest_data_invalid(client):
    from wave.ingest import ws_app
    from starlette.websockets import WebSocketDisconnect
    with TestClient(ws_app) as c, pytest.raises(WebSocketDisconnect):
        with c.websocket_connect('/ws/ingest-data') as ws:
            ws.send_text('invalid json')
            # should close immediately
            await ws.receive_text()
