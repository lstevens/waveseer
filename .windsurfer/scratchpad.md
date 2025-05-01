# Multi‑Time‑Frame Motif Discovery — Formal Specification

> **Project Codename:** *WaveSeer*
>
> **Delivery Target:** first internal PoC demo in **7 weeks**, prod‑ready beta in **Q3 2025**.

---

## 0  Objectives

1. **Discover & catalogue** recurrent price‑wave motifs across 1 min → 4 h BTC (and later multi‑coin) data *offline* on the local Xeon workstation.
2. Provide an **interactive visual UI** for motif auditing, labelling and search.
3. Expose a **runtime Pattern API** the *Seer* agent can call in the LLM Roundtable to flag live `PatternHit` events.
4. Maintain **testable, reproducible code** to slot straight into Windsurfer’s code‑gen pipeline.

---

## 1  Glossary (formal)

| Term                         | Definition                                                                        |
| ---------------------------- | --------------------------------------------------------------------------------- |
| **Motif**                    | A subsequence *s* of length *m* that repeats ≥ 2 times with profile distance ≤ ε. |
| **Window length *m***        | Size (in samples) of subsequence examined (e.g., 24 on 1‑h TF = 1 day).           |
| **Profile distance *d***     | Matrix‑Profile z‑normalized Euclidean distance between two subsequences.          |
| **Cluster**                  | A set of motifs grouped by DTW‑Barycentre distance ≤ δ.                           |
| **Pattern**                  | A *human‑approved* cluster given a semantic label (Leak→Flood, etc.).             |
| **PatternHit**               | JSON event `{ts_start, tf, pattern_id, score}` emitted in live trading.           |

---

## 2  Data Contract

### 2.1  Raw CSV Ingestion

```
<symbol>/<tf>/<symbol>_<tf>_<YYYY>.csv
```

Columns: `datetime, open, high, low, close, volume`.  
`datetime` MUST be ISO‑8601 & UTC.

### 2.2  Derived Time‑Frames

| Target TF | Source TF | Aggregation   |
| --------- | --------- | ------------- |
| 5 min     | 1 min     | OHLCV roll‑up |
| 15 min    | 1 min     | “             |
| 4 h       | 1 h       | OHLCV roll‑up |

Parquet caches live under `build/cache/<symbol>/<tf>.parquet` (Polars).  
All numeric columns stored as `float32` to cap RAM.

---

## 3  Configuration (YAML)

```yaml
symbols: [BTCUSDT]
timeframes:
  - tf: 1m
    windows: [30, 60, 120, 240]   # in samples
  - tf: 1h
    windows: [12, 24, 72]
dist_threshold_multiplier: 1.2     # ε = ε*mult
cluster:
  method: agglomerative
  linkage: ward
  max_dtw: 0.15                    # δ
ui:
  dash_host: 127.0.0.1
  dash_port: 8050
pattern_api:
  host: 0.0.0.0
  port: 9000
```

---

## 4  Software Architecture

```mermaid
graph TD
  subgraph Batch Pipeline
    A[CSV > Parquet Cache] --> B[STUMPY Scan per‑TF]
    B --> C[Motif Index<br>(DuckDB table)]
    C --> D[DTW Clusterer]
    D -->|cli label| E[Pattern Catalog]
  end

  subgraph Interactive UI
    G[Dash App] <-->|WS| D
  end

  subgraph Runtime
    SeerAgent -->|/match| P[Pattern API]
    PatternAPI -->|DuckDB lookup| E
  end
```

### 4.1  Module I/O Contracts

| Module       | Input                | Output                                 |
| ------------ | -------------------- | -------------------------------------- |
| `ingest.py`  | raw CSV              | `<symbol>/<tf>.parquet`                |
| `scan.py`    | parquet, YAML config | `duckdb://motifs.db::motif_idx`        |
| `cluster.py` | motif_idx           | `duckdb::clusters`, `duckdb::patterns` |
| `app.py`     | clusters, patterns   | Dash UI                                |
| `api.py`     | patterns             | REST `/match`, `/catalog`              |

---

## 5  CLI Commands (Make + Typer)

```bash
make env            # create conda env
wave ingest --all   # CSV → Parquet
wave scan 1h        # run profile @1h
wave cluster --tf 1h --window 24
wave ui             # open http://127.0.0.1:8050
wave api --reload   # Pattern API w/ hot‑reload
```

---

## 6  REST Pattern API (OpenAPI 3.1 excerpt)

```yaml
paths:
  /match:
    post:
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                tf: {type: string}
                seq: {type: array, items: {type: number}}
      responses:
        200:
          description: Match result
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Match'
components:
  schemas:
    Match:
      type: object
      properties:
        pattern_id: {type: string}
        score: {type: number}
        dist: {type: number}
```

---

## 7  UI Wireframe

- **Header bar:** symbol selector, TF drop‑down, window slider.
- **Main plot (Plotly):** price candles + coloured rectangles for motif occurrences.
- **Sidebar:**
  - Cluster table (Ag‑Grid) with sparkline preview.
  - Tag editor (text + colour).

---

## 8  Test Matrix

| Layer   | Test Type                  | Tool         |
| ------- | -------------------------- | ------------ |
| ingest  | schema validation          | pandera      |
| scan    | unit vs brute MASS         | pytest‑param |
| cluster | silhouette ≥ 0.5           | pytest       |
| API     | contract tests             | schemathesis |
| Seer    | prompt unit (OpenAI tools) | pytest‑async |

Coverage target ≥ 85 %.

---

## 9  Repository Layout

```
wave/
  ├─ wave/               # src pkg
  │   ├─ ingest.py
  │   ├─ scan.py
  │   ├─ cluster.py
  │   ├─ ui/
  │   └─ api/
  ├─ tests/
  ├─ Makefile
  ├─ env.yml            # conda
  ├─ config.yml         # default cfg
  └─ README.md
```

---

## 10  Installation & Run

```bash
conda env create -f env.yml
conda activate motif
make ingest           # first run (~10 min per year @ 1 m)
make scan-all         # loops per TF & window
make ui               # open browser
make api              # REST server
```

---

## 11  Timeline (Gantt, detail)

| Week | Tasks |
| ---- | ----- |
| 1 | MOT‑01 ingest; config parser; unit tests |
| 2 | MOT‑02 scan 1 h; store motif_idx; CLI progress bar |
| 3 | extend scan all TF; chunked MASS fallback; bench report |
| 4 | MOT‑03 Dash UI v0 (static); wire to motif_idx |
| 5 | MOT‑04 clustering; DTW barycentre impl; tagging CRUD |
| 6 | MOT‑05 Flask Pattern API; OpenAPI stub; contract tests |
| 7 | MOT‑06 Seer stub; round‑trip PatternHit simulation |

*Note:* if GPU becomes available, drop in `cupy` & switch to `stumpy.gpu_stump` (1‑line code change).

---

## 12  Stretch & Future

- Add **VALMOD** variable‑length motifs.
- Multi‑symbol correlation motifs (e.g., BTC leads ETH by N mins).
- Integrate Whisper/News embeddings to label motifs with sentiment.
- Export motifs as ONNX graph for ultra‑fast C++ matching.

---

**End of Formal Spec v1.0**
