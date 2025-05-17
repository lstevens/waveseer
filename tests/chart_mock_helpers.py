"""
Helper functions for mocking filesystem paths in chart service tests.
"""
import tempfile
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional

def create_mock_parquet(
    symbol: str = "testbtc", 
    timeframe: str = "1m",
    rows: int = 100,
    base_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Create mock parquet data for testing.
    
    Args:
        symbol: Symbol name
        timeframe: Timeframe string
        rows: Number of rows to generate
        base_dir: Optional base directory, if None creates a temp dir
        
    Returns:
        Dictionary with paths and dataframe
    """
    # Create temp dir if none provided
    if base_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        base_dir = Path(temp_dir.name)
        cleanup_callback = temp_dir.cleanup
    else:
        cleanup_callback = lambda: None
        
    # Create sample data
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="1min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": np.random.rand(rows) * 100 + 20000,
        "high": np.random.rand(rows) * 100 + 20100,
        "low": np.random.rand(rows) * 100 + 19900,
        "close": np.random.rand(rows) * 100 + 20050,
        "volume": np.random.rand(rows) * 10
    })
    
    # Create directory structure
    cache_dir = base_dir / "build" / "cache" / symbol
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    parquet_path = cache_dir / f"{timeframe}.parquet"
    df.to_parquet(str(parquet_path))
    
    # Convert to polars DataFrame
    pl_df = pl.from_pandas(df)
    
    return {
        "base_dir": base_dir,
        "parquet_path": parquet_path,
        "pandas_df": df,
        "polars_df": pl_df,
        "symbol": symbol,
        "timeframe": timeframe,
        "cleanup": cleanup_callback
    }
    
def mock_path_for_chart_service(monkeypatch, mock_dir: Path) -> None:
    """Patch Path in chart_service to point to mock directory.
    
    Args:
        monkeypatch: pytest monkeypatch fixture
        mock_dir: Path to mock directory
    """
    import wave.chart_service as chart_service
    import os
    
    # Store original Path class
    original_path = Path
    
    # Create a simplified patched Path class
    class MockPath(type(Path())):
        @classmethod
        def __new__(cls, *args, **kwargs):
            # If we're looking for the cache directory
            if len(args) == 1 and isinstance(args[0], str):
                path_str = args[0]
                if path_str == "build/cache" or path_str.startswith("build/cache/"):
                    # Replace with our mock directory path
                    parts = path_str.split("/")
                    if len(parts) > 2:
                        # Keep any subdirectories after build/cache
                        subdirs = "/".join(parts[2:])
                        return original_path(mock_dir) / "build" / "cache" / subdirs
                    else:
                        return original_path(mock_dir) / "build" / "cache"
            # Otherwise use the normal Path class
            return original_path.__new__(original_path, *args, **kwargs)
    
    # Replace the Path class in the chart_service module
    monkeypatch.setattr(chart_service, 'Path', MockPath)
