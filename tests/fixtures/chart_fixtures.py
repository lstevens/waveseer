"""
Test fixtures for chart service and visualization components.
"""
import os
import tempfile
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import pytest
from datetime import datetime, timedelta

@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV dataframe for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": np.random.rand(100) * 100 + 20000,
        "high": np.random.rand(100) * 100 + 20100,
        "low": np.random.rand(100) * 100 + 19900,
        "close": np.random.rand(100) * 100 + 20050,
        "volume": np.random.rand(100) * 10
    })
    return df

@pytest.fixture
def sample_ohlcv_pl_df():
    """Create sample Polars OHLCV dataframe for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": np.random.rand(100) * 100 + 20000,
        "high": np.random.rand(100) * 100 + 20100,
        "low": np.random.rand(100) * 100 + 19900,
        "close": np.random.rand(100) * 100 + 20050,
        "volume": np.random.rand(100) * 10
    })
    return pl.from_pandas(df)

@pytest.fixture
def mock_cache_dir():
    """Create temporary cache directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create structure matching production
        cache_dir = Path(tmpdir) / "build" / "cache" / "testbtc"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample parquet file with OHLCV data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")
        df = pd.DataFrame({
            "datetime": dates,
            "open": np.random.rand(100) * 100 + 20000,
            "high": np.random.rand(100) * 100 + 20100,
            "low": np.random.rand(100) * 100 + 19900,
            "close": np.random.rand(100) * 100 + 20050,
            "volume": np.random.rand(100) * 10
        })
        
        # Save as parquet
        parquet_path = cache_dir / "1m.parquet"
        df.to_parquet(str(parquet_path))
        
        yield tmpdir

@pytest.fixture
def patch_path(monkeypatch, mock_cache_dir):
    """Patch Path to use mock cache directory."""
    from wave import chart_service
    
    original_path = Path
    
    class PatchedPath(type(Path())):
        def __new__(cls, *args, **kwargs):
            if len(args) == 1 and args[0] == "build/cache":
                return original_path(mock_cache_dir) / "build" / "cache"
            return original_path.__new__(cls, *args, **kwargs)
    
    monkeypatch.setattr(chart_service, 'Path', PatchedPath)
    
    return mock_cache_dir
