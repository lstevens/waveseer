# API Reference

This document provides a detailed reference for the Waveseer API, covering both REST and WebSocket endpoints.

## REST API

Base URL: `http://<host>:9000`

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "ok"
}
```

### Pattern Matching

```
POST /match
```

Request body:
```json
{
  "tf": "1m",
  "seq": [1.0, 2.0, 3.0, 2.5, 2.0]
}
```

Response:
```json
{
  "pattern_id": "btcusd_1m_w20_c0",
  "score": 0.85,
  "dist": 0.15
}
```

Parameters:
- `tf`: Timeframe (e.g., "1m", "5m", "1h")
- `seq`: Price sequence to match against known patterns

### Pattern Catalog

```
GET /catalog
```

Response:
```json
{
  "patterns": [
    {
      "pattern_id": "btcusd_1m_w20_c0",
      "label": "Head and Shoulders",
      "color": "#FF5733"
    },
    {
      "pattern_id": "ethbtc_1h_w40_c2",
      "label": "Double Top",
      "color": "#33FF57"
    }
  ]
}
```

### Update Pattern

```
PUT /patterns/{pattern_id}
```

Request body:
```json
{
  "label": "Strong Reversal",
  "color": "#3357FF"
}
```

Response:
```json
{
  "status": "ok"
}
```

### Metrics

```
GET /metrics
```

Returns Prometheus metrics in plaintext format.

## Crypto Pattern Detection

### List Crypto Patterns
```http
GET /crypto/patterns?symbol={symbol}&timeframe={timeframe}&start={start}&end={end}
```
Response:
```json
[
  { "id": 1, "symbol_id": 1, "timeframe_id": 1, "pattern_type": "head_and_shoulders", "start_ts": "2025-05-03T00:00:00Z", "end_ts": "2025-05-03T01:00:00Z", "confidence": 0.87, "score": 0.92, "metadata": {}, "created_at": "2025-05-03T01:00:05Z" }
]
```

### Run Crypto Patterns
```http
POST /crypto/patterns/run
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "timeframe": "1m",
  "start": "2025-05-03T00:00:00Z",
  "end": "2025-05-03T01:00:00Z"
}
```
Response: same schema and data as List.

## WebSocket API

### Pattern Events Stream

Connect URL: `ws://<host>:8000/ws/patterns`

This WebSocket endpoint streams pattern detection events in real-time:

```json
{
  "ts_start": "2025-05-03T07:15:00Z",
  "tf": "1m",
  "pattern_id": "btcusd_1m_w20_c0",
  "score": 0.85
}
```

### Pattern Matching WebSocket

Connect URL: `ws://<host>:8000/ws/match`

Send:
```json
{
  "tf": "1m",
  "seq": [1.0, 2.0, 3.0, 2.5, 2.0]
}
```

Receive:
```json
{
  "pattern_id": "btcusd_1m_w20_c0",
  "score": 0.85,
  "dist": 0.15
}
```

### Data Ingestion WebSocket

Connect URL: `ws://<host>:8000/ws/ingest-data`

Send:
```json
{
  "symbol": "BTCUSD",
  "timestamp": 1618300000,
  "price": 54321.0,
  "volume": 1.234
}
```

Receive:
```json
{
  "status": "received",
  "symbol": "BTCUSD",
  "timestamp": 1618300000,
  "price": 54321.0,
  "volume": 1.234
}
```

## Error Handling

All API endpoints return appropriate HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Requested resource not found
- `500 Internal Server Error`: Server error

Example error response:
```json
{
  "error": "Invalid timeframe"
}
```

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

- REST endpoints: 100 requests per minute per IP
- WebSocket connections: 10 concurrent connections per IP

## Authentication

Authentication is not currently implemented but planned for future releases.

## Client Libraries

### Python

```python
import requests

def match_pattern(sequence, timeframe="1m"):
    response = requests.post(
        "http://localhost:9000/match",
        json={"tf": timeframe, "seq": sequence}
    )
    return response.json()

# Example usage
result = match_pattern([1.0, 1.2, 1.5, 1.3, 1.1])
print(f"Matched pattern: {result['pattern_id']} with score {result['score']}")
```

### JavaScript

```javascript
async function matchPattern(sequence, timeframe = "1m") {
  const response = await fetch("http://localhost:9000/match", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      tf: timeframe,
      seq: sequence,
    }),
  });
  
  return await response.json();
}

// Example usage
matchPattern([1.0, 1.2, 1.5, 1.3, 1.1])
  .then(result => console.log(`Matched pattern: ${result.pattern_id} with score ${result.score}`));
```
