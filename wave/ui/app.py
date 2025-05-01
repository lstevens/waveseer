import dash
from dash import html, dcc, dash_table
import requests
import yaml
from pathlib import Path
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

def serve_layout():
    cfg = yaml.safe_load(Path("config.yml").read_text())
    symbols = cfg["symbols"]
    tfs = [tf["tf"] for tf in cfg["timeframes"]]
    return html.Div([
        html.H1("WaveSeer UI"),
        dcc.Dropdown(symbols, value=symbols[0], id="symbol-select"),
        dcc.Dropdown(tfs, value=tfs[0], id="tf-select"),
        dcc.Graph(id="price-chart"),
        dash_table.DataTable(id="cluster-table",
                            columns=[{"name": "cluster_id", "id": "cluster_id"},
                                     {"name": "motif_idx", "id": "motif_idx"}],
                            data=[]),
        html.H2("Pattern Labels"),
        dash_table.DataTable(
            id="pattern-table",
            columns=[
                {"name": "pattern_id", "id": "pattern_id", "editable": False},
                {"name": "label", "id": "label", "editable": True},
                {"name": "color", "id": "color", "editable": True},
            ],
            data=[],
        ),
        html.Button("Save Labels", id="save-patterns"),
        html.Div(id="save-output"),
    ])

app.layout = serve_layout

@app.callback(
    [Output("price-chart", "figure"), Output("cluster-table", "data")],
    [Input("symbol-select", "value"), Input("tf-select", "value")]
)
def update(symbol, tf):
    # load price data
    import polars as pl
    import duckdb
    import plotly.graph_objs as go
    df = pl.read_parquet(f"build/cache/{symbol}/{tf}.parquet").to_pandas()
    fig = go.Figure(data=[go.Candlestick(x=df['datetime'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
    # load clusters
    db = duckdb.connect('motifs.db')
    qdf = db.execute(f"SELECT cluster_id, motif_idx FROM clusters WHERE symbol='{symbol}' AND tf='{tf}'").df()
    return fig, qdf.to_dict('records')

@app.callback(
    Output("pattern-table", "data"),
    [Input("symbol-select", "value"), Input("tf-select", "value")]
)
def load_patterns(symbol, tf):
    # fetch patterns from API
    cfg = yaml.safe_load(Path("config.yml").read_text())
    api = cfg.get("pattern_api", {})
    url = f"http://{api.get('host','0.0.0.0')}:{api.get('port',9000)}"
    try:
        res = requests.get(f"{url}/catalog")
        return res.json().get("patterns", []) if res.status_code == 200 else []
    except Exception:
        return []

@app.callback(
    Output("save-output", "children"),
    [Input("save-patterns", "n_clicks")],
    [State("pattern-table", "data")]
)
def save_labels(n_clicks, rows):
    if not n_clicks:
        return ""
    cfg = yaml.safe_load(Path("config.yml").read_text())
    api = cfg.get("pattern_api", {})
    url = f"http://{api.get('host','0.0.0.0')}:{api.get('port',9000)}"
    for r in rows:
        pid = r.get("pattern_id")
        payload = {"label": r.get("label",""), "color": r.get("color","")}
        try:
            requests.put(f"{url}/patterns/{pid}", json=payload)
        except Exception:
            continue
    return f"Saved {len(rows)} patterns."

def run():
    app.run(host="0.0.0.0", port=8050, debug=True)
