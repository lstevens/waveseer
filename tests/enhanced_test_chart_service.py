"""
Enhanced unit tests for the Chart Service component.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient

# Import the chart service FastAPI app
from wave.chart_service import app

# Create test client
client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_bars_endpoint_nonexistent_data():
    """Test that the bars endpoint gracefully handles nonexistent data."""
    response = client.get("/bars?symbol=nonexistent&tf=1m")
    assert response.status_code == 404
    assert "error" in response.json()


def test_bars_endpoint_with_data(patch_path):
    """Test that the bars endpoint returns OHLCV data."""
    response = client.get("/bars?symbol=testbtc&tf=1m&window=10")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "testbtc"
    assert data["timeframe"] == "1m"
    assert isinstance(data["bars"], list)
    assert len(data["bars"]) <= 10  # Should respect window parameter


def test_bars_endpoint_with_params(patch_path):
    """Test bars endpoint with different parameters."""
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


@patch('wave.chart.draw_candlestick_chart')
def test_chart_endpoint(mock_draw, patch_path):
    """Test that the chart endpoint returns HTML with embedded chart."""
    # Mock the chart drawing function to return a fixed base64 string
    mock_draw.return_value = "mocked_base64_data"

    response = client.get("/chart?symbol=testbtc&tf=1m&window=10")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert "data:image/png;base64,mocked_base64_data" in response.text

    # Verify draw_candlestick_chart was called with correct params
    mock_draw.assert_called_once()
    args, kwargs = mock_draw.call_args
    assert kwargs["title"] == "testbtc 1m Chart"
    assert kwargs["figsize"] == (8, 5)  # Default size


def test_chart_endpoint_with_custom_size(patch_path):
    """Test chart endpoint with custom size parameters."""
    with patch('wave.chart.draw_candlestick_chart', return_value="mocked_base64_data"):
        response = client.get("/chart?symbol=testbtc&tf=1m&width=1200&height=800")
        assert response.status_code == 200
        assert "max-width: 1200px" in response.text


def test_chart_endpoint_error_handling():
    """Test error handling in chart endpoint."""
    response = client.get("/chart?symbol=nonexistent&tf=1m")
    assert response.status_code == 404
    assert "Error Generating Chart" in response.text
    assert "No data available" in response.text


def test_chart_endpoint_with_polars_error(patch_path):
    """Test chart endpoint handling polars errors."""
    with patch('polars.read_parquet', side_effect=Exception("Polars error")):
        response = client.get("/chart?symbol=testbtc&tf=1m")
        assert response.status_code == 500
        assert "Error Generating Chart" in response.text
        assert "Polars error" in response.text


def test_chart_rendering_error(patch_path):
    """Test chart endpoint handling rendering errors."""
    with patch('wave.chart.draw_candlestick_chart', side_effect=ValueError("Chart render error")):
        response = client.get("/chart?symbol=testbtc&tf=1m")
        assert response.status_code == 500
        assert "Error Generating Chart" in response.text
        assert "Chart render error" in response.text
