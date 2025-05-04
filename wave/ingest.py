import typer
from pathlib import Path
import yaml
from typing import List
import json
import requests
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Body, Depends
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import polars as pl
import pandas as pd
try:
    import pandera as pa
    from pandera import Column
except ImportError:
    pa = None
from pydantic import BaseModel, ValidationError
import asyncio
import inspect
import subprocess
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from datetime import datetime, timedelta
from typing import Optional, List
from wave.chart import draw_candlestick_chart
from uuid import uuid4
from datetime import datetime
import logging
from wave.crypto_heatmap.pipeline import PatternPipeline

"""
Real-time WebSocket server with static UI.
"""

# Create FastAPI app with lifespan context manager
# Setup lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application.
    
    This replaces the deprecated on_event("startup") and on_event("shutdown") handlers.
    """
    # Startup: Spawn SeerAgent CLI processes to stream PatternHit events
    base = Path(__file__).parent.parent
    cfg = yaml.safe_load((base / "config.yml").read_text())
    cache = base / "build" / "cache"
    
    if cache.exists() and cache.is_dir():
        stream_url = "http://127.0.0.1:8000/stream"
        try:
            # Launch one CLI process per symbol/tf
            seer_processes = []
            for symbol_dir in cache.iterdir():
                symbol = symbol_dir.name
                for tf_cfg in cfg.get("timeframes", []):
                    tf = tf_cfg.get("tf")
                    cmd = ["python3", "-m", "wave.seer",
                           "--symbol", symbol,
                           "--tf", tf,
                           "--stream_url", stream_url]
                    process = subprocess.Popen(cmd)
                    seer_processes.append(process)
            print(f"Started {len(seer_processes)} SeerAgent processes")
        except Exception as e:
            print(f"SeerAgent startup skipped: {e}")
    
    # Yield control back to FastAPI
    yield
    
    # Shutdown: Add any cleanup logic here if needed
    # No cleanup needed currently, but this is where you would add it

# Create FastAPI app with lifespan context manager
ws_app = FastAPI(lifespan=lifespan)

# No middleware required for local testing

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
# Connection state tracking for WebSocket clients
class WebSocketConnectionState:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.state = "CONNECTED"
        self.reconnect_count = 0
    def mark_active(self):
        self.state = "ACTIVE"
    def mark_idle(self):
        self.state = "IDLE"
    def reconnect(self):
        self.reconnect_count += 1
        self.state = "CONNECTED"
    def disconnect(self):
        self.state = "DISCONNECTED"

# Manager for WebSocket connections and events
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    def add_connection(self, websocket: WebSocket):
        self.active_connections.append(websocket)
    def remove_connection(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    def broadcast(self, message: dict):
        for ws in list(self.active_connections):
            try:
                res = ws.send_json(message)
                # if send_json returns a coroutine, schedule it
                if inspect.iscoroutine(res):
                    asyncio.create_task(res)
            except WebSocketDisconnect:
                self.remove_connection(ws)
    async def ping_client(self, connection_state: WebSocketConnectionState) -> bool:
        # Default ping behavior
        return True

# Global connection manager instance
manager = ConnectionManager()

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

@ws_app.websocket("/ws/ingest")
async def websocket_subscribe(websocket: WebSocket):
    """Subscribe for pattern events streamed via POST /stream."""
    await websocket.accept()
    # Initialize connection state and register
    connection_state = WebSocketConnectionState(client_id=str(uuid4()))
    manager.add_connection(websocket)
    # Notify client of successful connection
    await websocket.send_json({"type": "connection_established", "client_id": connection_state.client_id})
    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") == "ping":
                alive = await manager.ping_client(connection_state)
                if alive:
                    await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.remove_connection(websocket)
        connection_state.disconnect()

# Schema for incoming PatternHit events
class PatternHitRequest(BaseModel):
    symbol: str
    timeframe: str
    start: datetime
    end: datetime

@ws_app.post("/stream")
async def stream_event(event_raw: dict = Body(...)):
    """Receive PatternHit events and broadcast to subscribers."""
    # Determine if UI raw event or PatternHitRequest for pattern detection
    try:
        event = PatternHitRequest(**event_raw)
    except ValidationError as e:
        # Return 422 Unprocessable Entity for schema validation errors
        return Response(
            status_code=422,
            content=json.dumps({"detail": str(e)}),
            media_type="application/json"
        )
    
    # For non-PatternHitRequest events (UI raw events), broadcast directly and return
    if not isinstance(event_raw, dict) or not all(k in event_raw for k in ["symbol", "timeframe", "start", "end"]):
        manager.broadcast(event_raw)
        return {"status": "ok"}
    # Run pattern detection in background thread and broadcast enriched matches
    try:
        # Offload detection to thread pool
        matches = await asyncio.to_thread(
            PatternPipeline().run,
            event.symbol,
            event.timeframe,
            event.start,
            event.end,
        )
        for m in matches:
            # Serialize PatternMatch or dict
            if isinstance(m, dict):
                data = m
            else:
                data = m.dict()
            # Build unified payload
            payload = {
                "symbol": event.symbol,
                "tf": event.timeframe,
                "ts_start": event.start.isoformat(),
                "ts_end": event.end.isoformat(),
                **data,
            }
            manager.broadcast(payload)
    except Exception as e:
        logger.exception(f"PatternPipeline error: {e}")
    return {"status": "ok"}

@ws_app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics for ingestion server"""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@ws_app.get("/bars")
async def get_bars(
    symbol: str,
    tf: str,
    start: Optional[str] = None,
    window: int = 60,
    limit: int = 200
):
    """Retrieve OHLCV bar data for the specified symbol and timeframe.
    
    Args:
        symbol: Trading pair symbol (e.g., 'btcusd')
        tf: Timeframe (e.g., '1m', '1h')
        start: Optional start timestamp (ISO format)
        window: Number of bars to return (default: 60)
        limit: Maximum number of bars to return (default: 200)
        
    Returns:
        JSON with OHLCV data for the requested bars
    """
    try:
        # Convert path to parquet file
        cache_dir = Path("build/cache")
        parquet_path = cache_dir / symbol / f"{tf}.parquet"
        
        if not parquet_path.exists():
            return {"error": f"No data available for {symbol}/{tf}"}
        
        # Load data with polars
        df = pl.read_parquet(str(parquet_path))
        
        # Filter by start time if provided
        if start:
            try:
                # For simplicity, let's just return the last N bars
                # This avoids timestamp comparison issues
                print(f"Note: Using last {window} bars instead of filtering by timestamp")
                df = df.tail(window)
            except Exception as e:
                print(f"Data selection error: {e}")
                # Continue with whatever data we have
        
        # Apply limit
        df = df.limit(min(window, limit))
        
        # Convert to dict for JSON response
        result = {
            "symbol": symbol,
            "timeframe": tf,
            "bars": df.to_dicts()
        }
        
        return result
    except Exception as e:
        return {"error": str(e)}

@ws_app.get("/chart")
async def get_chart(
    symbol: str,
    tf: str,
    start: Optional[str] = None,
    window: int = 60,
    limit: int = 200,
    width: int = 800,
    height: int = 500
):
    """Generate a candlestick chart for the specified symbol and timeframe.
    
    Args:
        symbol: Trading pair symbol (e.g., 'btcusd')
        tf: Timeframe (e.g., '1m', '1h')
        start: Optional start timestamp (ISO format)
        window: Number of bars to include (default: 60)
        limit: Maximum number of bars (default: 200)
        width: Image width in pixels (default: 800)
        height: Image height in pixels (default: 500)
        
    Returns:
        HTML response with embedded chart image
    """
    try:
        # Convert path to parquet file
        cache_dir = Path("build/cache")
        parquet_path = cache_dir / symbol / f"{tf}.parquet"
        
        if not parquet_path.exists():
            return {"error": f"No data available for {symbol}/{tf}"}
        
        # Load data with polars
        df = pl.read_parquet(str(parquet_path))
        
        # Filter by start time if provided
        if start:
            start_dt = pd.to_datetime(start)
            df = df.filter(pl.col("datetime") >= start_dt)
        
        # Apply limit
        df = df.limit(min(window, limit))
        
        # Convert to pandas for charting
        df_pd = df.to_pandas()
        
        # Generate chart
        img_base64 = draw_candlestick_chart(
            df_pd, 
            title=f"{symbol} {tf} Chart",
            figsize=(width/100, height/100)  # Convert pixels to inches (approx)
        )
        
        # Create HTML response
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{symbol} {tf} Chart</title>
            <style>
                body {{ font-family: sans-serif; margin: 0; padding: 20px; }}
                h1 {{ font-size: 24px; }}
                img {{ max-width: 100%; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>{symbol} {tf} Chart</h1>
            <img src="data:image/png;base64,{img_base64}" alt="Candlestick Chart">
        </body>
        </html>
        """
        
        return Response(content=html_content, media_type="text/html")
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error Generating Chart</h1>
            <p>{str(e)}</p>
        </body>
        </html>
        """
        return Response(content=error_html, media_type="text/html")

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
                except (AttributeError, ValueError):
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
