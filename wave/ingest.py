import typer
from pathlib import Path
import yaml
from typing import List
import json
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Body
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import polars as pl
import pandas as pd
try:
    import pandera as pa
    from pandera import Column
except ImportError:
    pa = None
from pydantic import BaseModel, ValidationError
import asyncio
import subprocess
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

"""
Real-time WebSocket server with static UI.
"""
ws_app = FastAPI()

# middleware for logging HTTP requests
@ws_app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"HTTP {request.method} {request.url}")
    response = await call_next(request)
    return response

# health-check endpoint
@ws_app.get("/health")
async def health():
    return {"status": "OK"}

ui_dir = Path(__file__).parent / 'ui'

# Serve static assets under /static
ws_app.mount("/static", StaticFiles(directory=ui_dir), name="static")

# Serve index.html at root
@ws_app.get("/")
async def root():
    return FileResponse(ui_dir / 'index.html', media_type='text/html')

"""
WebSocket endpoint for ingestion
"""
# WebSocket clients for broadcast
clients: List[WebSocket] = []

@ws_app.websocket("/ws/ingest")
async def websocket_subscribe(websocket: WebSocket):
    """Subscribe for pattern events streamed via POST /stream."""
    await websocket.accept()
    clients.append(websocket)
    try:
        # Keep connection alive for broadcast messages
        while True:
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        clients.remove(websocket)

@ws_app.post("/stream")
async def stream_event(event: dict = Body(...)):
    """Receive PatternHit events and broadcast to subscribers."""
    for ws in list(clients):
        try:
            await ws.send_json(event)
        except WebSocketDisconnect:
            clients.remove(ws)
    return {"status": "ok"}

@ws_app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics for ingestion server"""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@ws_app.websocket("/ws/match")
async def websocket_match(websocket: WebSocket):
    """Forward incoming messages to pattern API."""
    await websocket.accept()
    # load pattern API config
    cfg = yaml.safe_load((Path("config.yml")).read_text())
    host = cfg.get("pattern_api", {}).get("host", "127.0.0.1")
    port = cfg.get("pattern_api", {}).get("port", 9000)
    api_url = f"http://{host}:{port}"
    while True:
        try:
            data = await websocket.receive_text()
            payload = json.loads(data)
            res = requests.post(f"{api_url}/match", json=payload)
            if res.status_code == 200:
                try:
                    result = res.json()
                except ValueError:
                    result = json.loads(res.text)
                await websocket.send_json(result)
            else:
                await websocket.send_json({"error": res.status_code})
        except WebSocketDisconnect:
            break

@ws_app.websocket("/ws/ingest-data")
async def websocket_ingest_data(websocket: WebSocket):
    """Receive raw tick data, validate schema, and ack."""
    await websocket.accept()
    class IngestionPayload(BaseModel):
        symbol: str
        timestamp: int
        price: float
        volume: float
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = IngestionPayload.parse_raw(data)
            except ValidationError:
                await websocket.close(code=1003)
                return
            # TODO: forward payload to inference engine
            await websocket.send_json({"status": "received", **payload.dict()})
    except WebSocketDisconnect:
        pass

@ws_app.on_event("startup")
async def start_seer_agents():
    """Spawn SeerAgent CLI processes to stream PatternHit events."""
    base = Path(__file__).parent.parent
    cfg = yaml.safe_load((base / "config.yml").read_text())
    cache = base / "build" / "cache"
    # Skip if cache directory does not exist
    if not cache.exists() or not cache.is_dir():
        return
    stream_url = "http://127.0.0.1:8000/stream"
    try:
        # Launch one CLI process per symbol/tf
        for symbol_dir in cache.iterdir():
            symbol = symbol_dir.name
            for tf_cfg in cfg.get("timeframes", []):
                tf = tf_cfg.get("tf")
                cmd = ["python3", "-m", "wave.seer",
                       "--symbol", symbol,
                       "--tf", tf,
                       "--stream_url", stream_url]
                subprocess.Popen(cmd)
    except Exception as e:
        print(f"SeerAgent startup skipped: {e}")

# CLI app
app = typer.Typer()

@app.callback(invoke_without_command=True)
def ingest(
    all_: bool = typer.Option(False, "--all", "-a", help="Ingest all symbols/timeframes from config"),
    symbols: List[str] = typer.Option([], "--symbol", "-s", help="Symbols to ingest (repeatable)"),
    tfs: List[str] = typer.Option([], "--tf", "-t", help="Timeframes to ingest (repeatable)"),
):
    """CSV → Parquet ingestion"""
    cfg = yaml.safe_load((Path("config.yml")).read_text())
    cfg_symbols = cfg.get("symbols", [])
    cfg_tfs = [tf["tf"] for tf in cfg.get("timeframes", [])]
    # determine symbols
    if all_:
        symbols_to_ingest = cfg_symbols
    elif symbols:
        symbols_to_ingest = symbols
    else:
        symbols_to_ingest = [typer.prompt("Symbol", default=cfg_symbols[0] if cfg_symbols else "")]
    # determine timeframes
    if tfs:
        tfs_to_ingest = tfs
    else:
        tfs_to_ingest = cfg_tfs

    # process each symbol and timeframe
    cache_base = Path("build/cache")
    for symbol in symbols_to_ingest:
        for tf in cfg.get("timeframes", []):
            tf_name = tf.get("tf")
            if tf_name not in tfs_to_ingest:
                continue
            csv_dir = Path(symbol) / tf_name
            files = sorted(csv_dir.glob(f"{symbol}_{tf_name}_*.csv"))
            if not files:
                typer.echo(f"No CSV for {symbol}/{tf_name}, skipping")
                continue
            df_list = []
            for csv_file in files:
                df = pl.read_csv(str(csv_file))
                # parse datetime and cast numeric columns
                df = df.with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime).alias("datetime"),
                    *[pl.col(c).cast(pl.Float32) for c in ["open","high","low","close","volume"]]
                ])
                df_list.append(df)
            # combine into Polars DataFrame
            df_pl = pl.concat(df_list)
            # convert to pandas for schema validation
            df_pd = df_pl.to_pandas()
            # convert datetime to naive nanosecond datetime and drop timezone
            df_pd['datetime'] = pd.to_datetime(df_pd['datetime']).dt.tz_localize(None)
            if pa:
                schema = pa.DataFrameSchema({
                    "datetime": Column(pa.DateTime),
                    "open": Column(pa.Float32),
                    "high": Column(pa.Float32),
                    "low": Column(pa.Float32),
                    "close": Column(pa.Float32),
                    "volume": Column(pa.Float32),
                })
                df_pd = schema.validate(df_pd)
            # back to Polars
            df_all = pl.from_pandas(df_pd)
            # write parquet
            out_dir = cache_base / symbol
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{tf_name}.parquet"
            df_all.write_parquet(str(out_path))
            typer.echo(f"Wrote {out_path}")
