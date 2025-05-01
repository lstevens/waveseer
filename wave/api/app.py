from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from typing import List
import duckdb

app = FastAPI()

# WebSocket clients storage
_clients: List[WebSocket] = []

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

class MatchRequest(BaseModel):
    tf: str
    seq: list[float]

class MatchResponse(BaseModel):
    pattern_id: str
    score: float
    dist: float

@app.post("/match", response_model=MatchResponse)
async def match(req: MatchRequest):
    """Return best pattern match for given sequence"""
    # lookup patterns table and return first pattern or default
    db = duckdb.connect("motifs.db")
    rows = db.execute("SELECT pattern_id FROM patterns LIMIT 1").fetchall()
    pattern_id = rows[0][0] if rows else ""
    # scoring stub: default
    return MatchResponse(pattern_id=pattern_id, score=0.0, dist=0.0)

@app.get("/catalog")
async def catalog():
    """Return list of available patterns"""
    # use module-level duckdb
    db = duckdb.connect("motifs.db")
    # include color
    try:
        df = db.execute("SELECT pattern_id, label, color FROM patterns").df()
    except Exception:
        # fallback if color column missing
        df = db.execute("SELECT pattern_id, label, '' AS color FROM patterns").df()
    patterns = [row._asdict() for row in df.itertuples(index=False)]
    return {"patterns": patterns}

class UpdatePatternRequest(BaseModel):
    label: str
    color: str

@app.put("/patterns/{pattern_id}")
async def update_pattern(pattern_id: str, req: UpdatePatternRequest):
    """Update label & color for a pattern"""
    # use module-level duckdb connection (supports monkeypatch)
    db = duckdb.connect("motifs.db")
    db.execute(
        "UPDATE patterns SET label = ?, color = ? WHERE pattern_id = ?",
        (req.label, req.color, pattern_id)
    )
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

class PatternHit(BaseModel):
    ts_start: str
    tf: str
    pattern_id: str
    score: float

@app.websocket("/ws/patterns")
async def patterns_ws(websocket: WebSocket):
    """WebSocket endpoint for pattern hit subscribers"""
    await websocket.accept()
    _clients.append(websocket)
    try:
        while True:
            # keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        _clients.remove(websocket)

@app.post("/stream")
async def stream_event(event: PatternHit):
    """Receive PatternHit events and broadcast to WS clients"""
    for ws in list(_clients):
        try:
            await ws.send_json(event.dict())
        except Exception:
            _clients.remove(ws)
    return {"status": "ok"}
