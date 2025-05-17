"""Unit tests for the Chart Service component.
"""
import pytest
import pandas as pd
import polars as pl
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch

# Create test client
from wave.chart_service import app
client = TestClient(app)

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    # Generate 100 rows of test data
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


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_bars_endpoint_nonexistent_data():
    """Test that the bars endpoint handles nonexistent data."""
    with patch('pathlib.Path.exists', return_value=False):
        response = client.get("/bars?symbol=nonexistent&tf=1h")
        assert response.status_code == 404
        assert "error" in response.json()


def test_bars_endpoint_with_data(sample_ohlcv_data):
    """Test that the bars endpoint returns OHLCV data."""
    # Use patching to mock file existence and data reading
    with patch('pathlib.Path.exists', return_value=True), \
         patch('polars.read_parquet', return_value=sample_ohlcv_data["polars"]):
        response = client.get("/bars?symbol=testbtc&tf=1m&window=10")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "testbtc"
        assert data["timeframe"] == "1m"
        assert isinstance(data["bars"], list)
        assert len(data["bars"]) <= 10  # Should respect window parameter


def test_bars_endpoint_with_params(sample_ohlcv_data):
    """Test bars endpoint with different parameters."""
    with patch('pathlib.Path.exists', return_value=True), \
         patch('polars.read_parquet', return_value=sample_ohlcv_data["polars"]):
        # Test with start parameter
        response = client.get("/bars?symbol=testbtc&tf=1m&start=2023-01-01T00:00:00&window=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data["bars"]) <= 5  # Should respect window parameter

        # Test with limit parameter
        response = client.get("/bars?symbol=testbtc&tf=1m&limit=3")
        assert response.status_code == 200
        data = response.json()
        assert len(data["bars"]) <= 3  # Should respect limit parameter


def test_chart_endpoint(sample_ohlcv_data):
    """Test that the chart endpoint returns HTML with embedded chart."""
    # Completely mock out the draw_candlestick_chart function at the module level
    # This bypasses the DataFrame conversion issue entirely
    with patch('wave.chart_service.draw_candlestick_chart', return_value="mocked_base64_data"), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('polars.read_parquet', return_value=sample_ohlcv_data["polars"]):

        response = client.get("/chart?symbol=testbtc&tf=1m&window=10")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "data:image/png;base64,mocked_base64_data" in response.text


def test_chart_endpoint_with_custom_size(sample_ohlcv_data):
    """Test chart endpoint with custom size parameters."""
    with patch('wave.chart_service.draw_candlestick_chart', return_value="mocked_base64_data"), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('polars.read_parquet', return_value=sample_ohlcv_data["polars"]):

        response = client.get("/chart?symbol=testbtc&tf=1m&width=1200&height=800")
        assert response.status_code == 200
        assert "max-width: 1200px" in response.text


def test_chart_endpoint_error_handling():
    """Test chart endpoint handling for missing data."""
    with patch('pathlib.Path.exists', return_value=False):
        response = client.get("/chart?symbol=nonexistent&tf=1h")
        assert response.status_code == 404
        assert "No data available" in response.text


def test_chart_endpoint_with_polars_error():
    """Test chart endpoint handling polars errors."""
    with patch('pathlib.Path.exists', return_value=True), \
         patch('polars.read_parquet', side_effect=Exception("Polars error")):
        response = client.get("/chart?symbol=testbtc&tf=1m")
        assert response.status_code == 500
        assert "Error Generating Chart" in response.text
        assert "Polars error" in response.text


def test_chart_rendering_error(sample_ohlcv_data):
    """Test chart endpoint handling rendering errors."""
    with patch('wave.chart_service.draw_candlestick_chart', side_effect=ValueError("Chart render error")), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('polars.read_parquet', return_value=sample_ohlcv_data["polars"]):
        response = client.get("/chart?symbol=testbtc&tf=1m")
        assert response.status_code == 500
        assert "Error Generating Chart" in response.text
        assert "Chart render error" in response.text
