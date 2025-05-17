"""
Chart UI components for Waveseer pattern visualization.
Provides modern frontend components to display candlestick charts.
"""
import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import requests
from typing import Dict, Optional, Any


class ChartModal:
    """Modal dialog component for displaying candlestick charts."""

    def __init__(self, chart_service_url: str = "http://localhost:8010"):
        """Initialize chart modal component.

        Args:
            chart_service_url: URL of the chart service
        """
        self.chart_service_url = chart_service_url
        self.id_counter = 0

    def _get_unique_id(self, prefix: str = "chart"):
        """Generate a unique ID for Dash components."""
        self.id_counter += 1
        return f"{prefix}-{self.id_counter}"

    def get_modal(self, id_prefix: str = "pattern") -> html.Div:
        """Create a modal dialog component.

        Args:
            id_prefix: Prefix for component IDs

        Returns:
            html.Div modal component
        """
        modal_id = f"{id_prefix}-modal"
        modal_content_id = f"{id_prefix}-modal-content"
        close_id = f"{id_prefix}-close"
        loading_id = f"{id_prefix}-loading"
        chart_id = f"{id_prefix}-chart"

        return html.Div([
            html.Div(
                id=modal_id,
                className="modal",
                style={"display": "none"},
                children=[
                    html.Div(
                        className="modal-content",
                        id=modal_content_id,
                        children=[
                            html.Span(
                                "×",
                                id=close_id,
                                className="close",
                            ),
                            html.H2(id=f"{id_prefix}-title"),
                            html.Div(
                                "Loading chart...",
                                id=loading_id,
                                style={"marginBottom": "20px"}
                            ),
                            dcc.Graph(
                                id=chart_id,
                                style={"display": "none"},
                                figure={},
                                config={"displayModeBar": True},
                            ),
                        ],
                    ),
                ],
            ),

            # JavaScript to handle modal behavior
            html.Script(
                f"""
                document.getElementById('{close_id}').onclick = function() {{
                    document.getElementById('{modal_id}').style.display = 'none';
                }};

                window.onclick = function(event) {{
                    if (event.target == document.getElementById('{modal_id}')) {{
                        document.getElementById('{modal_id}').style.display = 'none';
                    }}
                }};
                """
            ),
        ])

    def create_chart_callback(self, app: dash.Dash, id_prefix: str = "pattern"):
        """Register callbacks for the chart modal.

        Args:
            app: Dash application
            id_prefix: Prefix for component IDs
        """
        from dash.dependencies import Input, Output, State

        @app.callback(
            [
                Output(f"{id_prefix}-modal", "style"),
                Output(f"{id_prefix}-title", "children"),
                Output(f"{id_prefix}-loading", "style"),
                Output(f"{id_prefix}-chart", "style"),
                Output(f"{id_prefix}-chart", "figure"),
            ],
            [Input(f"{id_prefix}-open-btn", "n_clicks")],
            [
                State(f"{id_prefix}-symbol", "value"),
                State(f"{id_prefix}-timeframe", "value"),
                State(f"{id_prefix}-timestamp", "value"),
                State(f"{id_prefix}-pattern-id", "value"),
            ]
        )
        def update_chart(n_clicks, symbol, timeframe, timestamp, pattern_id):
            """Update chart on button click."""
            if not n_clicks:
                # Initial load, don't show modal
                return {"display": "none"}, "", {}, {"display": "none"}, {}

            # Set modal to visible and show loading
            modal_style = {"display": "block"}
            title = f"{symbol.upper()} {timeframe} Pattern: {pattern_id}"
            loading_style = {"display": "block", "marginBottom": "20px"}
            chart_style = {"display": "none"}
            empty_figure = {}

            try:
                # Get OHLCV data from chart service
                params = {
                    "symbol": symbol,
                    "tf": timeframe,
                    "start": timestamp,
                    "window": 60
                }

                response = requests.get(
                    f"{self.chart_service_url}/bars",
                    params=params,
                    timeout=5
                )

                if response.status_code != 200:
                    return modal_style, f"Error: {response.text}", {"display": "none"}, {"display": "none"}, {}

                # Parse JSON response
                data = response.json()
                if "error" in data:
                    return modal_style, f"Error: {data['error']}", {"display": "none"}, {"display": "none"}, {}

                # Create candlestick chart
                bars = data["bars"]
                df = pd.DataFrame(bars)

                # Create Plotly figure
                fig = go.Figure(data=[
                    go.Candlestick(
                        x=df["datetime"],
                        open=df["open"],
                        high=df["high"],
                        low=df["low"],
                        close=df["close"],
                        name="Price"
                    )
                ])

                # Add volume bars
                fig.add_trace(
                    go.Bar(
                        x=df["datetime"],
                        y=df["volume"],
                        name="Volume",
                        marker_color="rgba(128,128,128,0.5)",
                        opacity=0.5,
                        yaxis="y2"
                    )
                )

                # Add a marker for the pattern detection point
                if timestamp in df["datetime"].values:
                    idx = df.index[df["datetime"] == timestamp][0]
                    fig.add_trace(
                        go.Scatter(
                            x=[df["datetime"][idx]],
                            y=[df["high"][idx] * 1.01],  # Slightly above high
                            mode="markers",
                            marker=dict(
                                symbol="star",
                                size=12,
                                color="gold",
                                line=dict(width=2, color="black")
                            ),
                            name=f"Pattern: {pattern_id}",
                            hoverinfo="name"
                        )
                    )

                # Layout updates
                fig.update_layout(
                    title=title,
                    xaxis_title="Time",
                    yaxis_title="Price",
                    template="plotly_white",
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right",
                        showgrid=False
                    )
                )

                # Hide loading, show chart
                return modal_style, title, {"display": "none"}, {"display": "block"}, fig

            except Exception as e:
                # Show error
                return modal_style, f"Error loading chart: {str(e)}", {"display": "none"}, {"display": "none"}, {}


def create_pattern_row(pattern: Dict[str, Any], on_click: Optional[str] = None) -> html.Tr:
    """Create a table row for a pattern event.

    Args:
        pattern: Pattern event data
        on_click: Optional JavaScript to execute on row click

    Returns:
        html.Tr component
    """
    return html.Tr(
        className="pattern-row",
        style={"cursor": "pointer"} if on_click else {},
        onClick=on_click,
        children=[
            html.Td(pattern.get("ts_start", "")),
            html.Td(pattern.get("tf", "")),
            html.Td(pattern.get("pattern_id", "")),
            html.Td(f"{pattern.get('score', 0):.2f}"),
            html.Td(
                html.Button(
                    "View Chart",
                    className="view-chart-btn",
                    **{
                        "data-symbol": "btcusd",  # Default symbol
                        "data-tf": pattern.get("tf", "1m"),
                        "data-timestamp": pattern.get("ts_start", ""),
                        "data-pattern-id": pattern.get("pattern_id", ""),
                    }
                )
            ),
        ],
    )
