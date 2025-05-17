"""
Debug script for chart service tests to identify the root cause of test failures.
"""
import sys
import os
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import traceback

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the chart service FastAPI app
from wave.chart_service import app

# Create test client
client = TestClient(app)

def create_sample_data():
    """Create sample OHLCV data for debugging."""
    # Generate sample data with 100 rows
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": np.random.rand(100) * 100 + 20000,
        "high": np.random.rand(100) * 100 + 20100,
        "low": np.random.rand(100) * 100 + 19900,
        "close": np.random.rand(100) * 100 + 20050,
        "volume": np.random.rand(100) * 10
    })
    
    # Convert to polars DataFrame
    pl_df = pl.from_pandas(df)
    
    return {
        "pandas": df,
        "polars": pl_df,
        "dicts": pl_df.to_dicts()
    }

def debug_bars_endpoint():
    """Debug the /bars endpoint."""
    print("\n=== Debugging /bars endpoint ===")
    
    # Create sample data
    data = create_sample_data()
    
    # Print sample data info
    print(f"Polars DataFrame schema: {data['polars'].schema}")
    print(f"First row: {data['polars'][0]}")
    
    try:
        # Mock file existence and polars.read_parquet
        with patch('pathlib.Path.exists', return_value=True), \
             patch('polars.read_parquet', return_value=data['polars']):
             
            # Make request
            response = client.get("/bars?symbol=testbtc&tf=1m&window=10")
            
            # Print results
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            # Try to parse response
            try:
                if response.status_code == 500:
                    print(f"Error response: {response.text}")
                else:
                    print(f"Response JSON: {response.json()}")
            except Exception as e:
                print(f"Error parsing response: {e}")
                
    except Exception as e:
        print(f"Exception during request: {e}")
        traceback.print_exc()

def debug_chart_endpoint():
    """Debug the /chart endpoint."""
    print("\n=== Debugging /chart endpoint ===")
    
    # Create sample data
    data = create_sample_data()
    
    try:
        # Mock file existence, polars.read_parquet, and chart drawing
        with patch('pathlib.Path.exists', return_value=True), \
             patch('polars.read_parquet', return_value=data['polars']), \
             patch('wave.chart.draw_candlestick_chart', return_value="mocked_base64_data"):
             
            # Make request
            response = client.get("/chart?symbol=testbtc&tf=1m&width=800&height=500")
            
            # Print results
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 500:
                print(f"Error response: {response.text}")
            elif response.status_code == 200:
                print("Chart HTML response received successfully")
                
    except Exception as e:
        print(f"Exception during request: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Running chart service debug tests...")
    debug_bars_endpoint()
    debug_chart_endpoint()
    print("\nDebug completed.")
