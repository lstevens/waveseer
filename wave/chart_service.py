"""
Dedicated Chart Service for Waveseer pattern visualization.
Runs as independent FastAPI service for candlestick chart generation.
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
import polars as pl
import pandas as pd
import uvicorn
from typing import Optional
from datetime import datetime

from wave.chart import draw_candlestick_chart

# Initialize FastAPI app
app = FastAPI(title="Waveseer Chart Service",
              description="Generates candlestick charts for pattern visualization")


class ChartOptions(BaseModel):
    """Options for chart rendering."""
    width: int = 800
    height: int = 500
    title: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/bars", response_class=JSONResponse)
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
            return JSONResponse(content={"error": f"No data available for {symbol}/{tf}"}, status_code=404)

        # Load data with polars
        df = pl.read_parquet(str(parquet_path))

        # For simplicity, return the most recent data
        if start:
            try:
                # Return last N bars
                df = df.tail(window)
            except Exception as e:
                print(f"Data selection error: {e}")

        # Apply limit
        df = df.limit(min(window, limit))

        # Convert to serializable format - handle datetime objects explicitly
        bars_list = []
        for row in df.iter_rows(named=True):
            # Create a new dict with ISO formatted datetimes for each row
            serializable_row = {}
            for key, value in row.items():
                if isinstance(value, datetime):
                    serializable_row[key] = value.isoformat()
                else:
                    serializable_row[key] = value
            bars_list.append(serializable_row)

        # Create result dict with serializable data
        result = {
            "symbol": symbol,
            "timeframe": tf,
            "bars": bars_list
        }

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/chart", response_class=HTMLResponse)
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
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Error</title></head>
            <body>
                <h1>Error Generating Chart</h1>
                <p>No data available for {symbol}/{tf}</p>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=404)

        # Load data with polars
        df = pl.read_parquet(str(parquet_path))

        # For simplicity, return most recent data
        if start:
            try:
                df = df.tail(window)
            except Exception as e:
                print(f"Data selection error: {e}")

        # Apply limit
        df = df.limit(min(window, limit))

        # If testing with mocked data, ensure proper conversion to pandas
        # This avoids the "_NoValueType" error in testing
        try:
            # Direct conversion approach
            df_pd = df.to_pandas()

            # Verify datetime column exists and is properly formatted
            if 'datetime' in df_pd.columns and df_pd['datetime'].dtype == 'object':
                # Convert object datetime to proper pandas datetime
                df_pd['datetime'] = pd.to_datetime(df_pd['datetime'])

        except Exception as e:
            # Fallback approach - manual conversion through dictionaries
            print(f"Standard conversion failed: {e}, using fallback approach")
            records = []
            for row in df.iter_rows(named=True):
                # Create a clean record with proper datetime handling
                record = {}
                for key, value in row.items():
                    if key == 'datetime' and isinstance(value, datetime):
                        record[key] = value  # Pandas will recognize datetime objects
                    else:
                        record[key] = value
                records.append(record)
            df_pd = pd.DataFrame.from_records(records)

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
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                h1 {{ font-size: 24px; color: #333; }}
                img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .container {{ max-width: {width}px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px;
                              box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                .meta {{ color: #666; margin-bottom: 15px; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{symbol.upper()} {tf} Chart</h1>
                <div class="meta">Timeframe: {tf} • Data points: {len(df_pd)}</div>
                <img src="data:image/png;base64,{img_base64}" alt="Candlestick Chart">
            </div>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)
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
        return HTMLResponse(content=error_html, status_code=500)


def run_server(host="0.0.0.0", port=8010):
    """Run the chart service."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
