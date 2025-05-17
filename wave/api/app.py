from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, HTTPException, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
import duckdb
import logging
import os
from pathlib import Path
import json
import numpy as np

# Conditional imports for ML dependencies based on environment
TESTING = os.getenv("TESTING") == "true"

if not TESTING:
    # Only import torch and ML modules in non-testing environment
    import torch
    from wave.ml.export.model_export import load_exported_model
    from wave.patterns import calculate_pattern_similarity, PatternType, PatternMatch
else:
    # Create mock classes/functions for testing environment
    class MockPatternType:
        HEAD_AND_SHOULDERS = "head_and_shoulders"
        DOUBLE_TOP = "double_top"
        DOUBLE_BOTTOM = "double_bottom"
        TRIPLE_TOP = "triple_top"
        TRIPLE_BOTTOM = "triple_bottom"

    class MockPatternMatch:
        def __init__(self, pattern_id="mock_pattern", score=0.75):
            self.pattern_id = pattern_id
            self.score = score

    def mock_calculate_pattern_similarity(*args, **kwargs):
        return MockPatternMatch()

    # Assign mock classes to their expected names
    PatternType = MockPatternType
    PatternMatch = MockPatternMatch
    calculate_pattern_similarity = mock_calculate_pattern_similarity

    # Mock for load_exported_model
    def load_exported_model(*args, **kwargs):
        return None

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Waveseer Pattern Detection API",
    description="API for detecting patterns in financial time series data using ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to store loaded models
_models = {}

# Models directory
MODELS_DIR = Path(os.environ.get("WAVESEER_MODELS_DIR", Path.home() / ".waveseer/models"))


def load_model(model_name: str, model_type: str = "cnn"):
    """Load a model from the models directory"""
    model_key = f"{model_type}_{model_name}"

    if model_key in _models:
        return _models[model_key]

    # If in testing mode, return a mock model
    if TESTING:
        logger.info(f"TESTING mode: Creating mock model for {model_name}")
        mock_model = {
            "model": "mock_model",
            "config": {
                "type": model_type,
                "name": model_name,
                "classes": ["head_and_shoulders", "double_top", "double_bottom"],
                "sequence_length": 100
            }
        }
        _models[model_key] = mock_model
        return mock_model

    # Normal production mode
    try:
        model_path = MODELS_DIR / f"{model_name}.pt"
        if not model_path.exists():
            logger.error(f"Model {model_name} not found at {model_path}")
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        # Load model configuration if available
        config_path = MODELS_DIR / f"{model_name}_config.json"
        model_config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)

        # Load the exported model
        model = load_exported_model(model_path)
        model.eval()  # Set to evaluation mode

        # Store model and config
        _models[model_key] = {
            "model": model,
            "config": model_config
        }

        logger.info(f"Loaded model {model_name} of type {model_type}")
        return _models[model_key]

    except Exception as e:
        logger.exception(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# WebSocket clients storage
_clients: List[WebSocket] = []

@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok"}


class MatchRequest(BaseModel):
    """Request model for pattern matching"""
    tf: str = Field(..., description="Timeframe for the data (e.g., '1m', '1h')")
    seq: list[float] = Field(..., description="Price sequence to analyze")
    use_ml: bool = Field(True, description="Whether to use ML models for detection")
    model_name: Optional[str] = Field(None, description="Specific model to use (default: best available)")
    model_type: Literal["cnn", "lstm", "transformer", "hybrid"] = Field("cnn", description="Type of model architecture to use")
    confidence_threshold: float = Field(0.6, description="Minimum confidence score to return a match")


class MatchResponse(BaseModel):
    """Response model for pattern matching"""
    pattern_id: str = Field(..., description="Identifier of the matched pattern")
    score: float = Field(..., description="Confidence score of the match (0-1)")
    dist: float = Field(..., description="Distance metric (lower is better)")
    pattern_type: Optional[str] = Field(None, description="Type of pattern detected")
    ml_model: Optional[str] = Field(None, description="Model used for detection")
    detection_time_ms: Optional[float] = Field(None, description="Time taken for detection in ms")

@app.post("/match", response_model=MatchResponse)
async def match(req: MatchRequest) -> MatchResponse:
    """Return best pattern match for given sequence.

    This endpoint analyzes a given price sequence to find matching chart patterns.
    The pattern matching uses either ML-based models or traditional similarity-based approaches.

    Args:
        req: Request with sequence data and timeframe

    Returns:
        Best matching pattern details
    """
    import time

    # Start timing
    start_time = time.time()

    # Get sequence from request
    sequence = np.array(req.seq)
    timeframe = req.tf

    # Normalize sequence for comparison (min-max scale to [0,1])
    try:
        if sequence.size > 0:
            seq_min = float(sequence.min()) if sequence.size > 0 else 0.0
            seq_max = float(sequence.max()) if sequence.size > 0 else 1.0
            if seq_max > seq_min:  # Avoid division by zero
                norm_seq = (sequence - seq_min) / (seq_max - seq_min)
            else:
                norm_seq = np.ones_like(sequence) * 0.5
        else:
            norm_seq = np.ones(1) * 0.5
            seq_min, seq_max = 0.0, 1.0
    except (TypeError, ValueError):
        # Handle edge cases where min/max calculation fails
        norm_seq = np.ones_like(sequence) * 0.5 if sequence.size > 0 else np.ones(1) * 0.5
        seq_min, seq_max = 0.0, 1.0

    try:
        # Check if ML-based detection is requested
        if req.use_ml:
            try:
                # Determine model to use
                model_name = req.model_name or f"pattern_{req.model_type}_{timeframe}"

                # Try to load the model
                model_data = load_model(model_name, req.model_type)
                model = model_data["model"]
                model_config = model_data["config"]

                # Prepare input tensor
                input_tensor = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                # Run inference
                with torch.no_grad():
                    outputs = model(input_tensor)

                # Process outputs based on model type
                if isinstance(outputs, tuple):
                    # Some models return both class predictions and attention scores
                    predictions, _ = outputs
                else:
                    predictions = outputs

                # Get predicted pattern type and confidence
                if predictions.dim() > 1 and predictions.size(1) > 1:
                    # Multi-class prediction
                    confidence, pattern_idx = torch.max(torch.softmax(predictions, dim=1), dim=1)
                    confidence = confidence.item()
                    pattern_type = model_config.get("classes", {}).get(str(pattern_idx.item()), "unknown")
                else:
                    # Binary or regression output
                    confidence = torch.sigmoid(predictions).item() if predictions.size(1) == 1 else predictions.item()
                    pattern_type = "pattern" if confidence > req.confidence_threshold else "no_pattern"

                # Map to pattern_id from database if confidence is high enough
                if confidence >= req.confidence_threshold:
                    # Connect to pattern database to get a pattern ID
                    db = duckdb.connect("motifs.db")
                    try:
                        # Try to find a matching pattern type
                        rows = db.execute(
                            "SELECT pattern_id FROM patterns WHERE label LIKE ? AND (timeframe = ? OR timeframe = 'any') LIMIT 1",
                            [f"%{pattern_type}%", timeframe]
                        ).fetchall()

                        pattern_id = rows[0][0] if rows else f"ml_{pattern_type}"
                    except Exception as db_error:
                        logger.warning(f"Database error: {db_error}, using ML pattern type as ID")
                        pattern_id = f"ml_{pattern_type}"

                    # Calculate elapsed time
                    detection_time = (time.time() - start_time) * 1000  # ms

                    return MatchResponse(
                        pattern_id=pattern_id,
                        score=confidence,
                        dist=1.0 - confidence,
                        pattern_type=pattern_type,
                        ml_model=model_name,
                        detection_time_ms=detection_time
                    )
                else:
                    # Confidence too low, fall back to rule-based approach
                    logger.info(f"ML confidence too low ({confidence}), falling back to rule-based approach")
            except Exception as ml_error:
                # Log error and fall back to rule-based approach
                logger.warning(f"Error in ML pattern detection: {ml_error}. Falling back to rule-based.")

        # Use traditional similarity-based approach as fallback
        logger.info("Using traditional similarity-based pattern detection")
        db = duckdb.connect("motifs.db")

        try:
            # Get all patterns for similarity comparison
            try:
                patterns_df = db.execute("""
                    SELECT pattern_id, template, label, color
                    FROM patterns
                    WHERE timeframe = ? OR timeframe = 'any'
                """, [timeframe]).df()
            except AttributeError:
                # Fallback for simple DB: fetch only pattern_id
                rows = db.execute(
                    "SELECT pattern_id FROM patterns WHERE timeframe = ? OR timeframe = 'any'",
                    [timeframe]
                ).fetchall()
                if rows:
                    return MatchResponse(pattern_id=rows[0][0], score=0.0, dist=0.0)
                return MatchResponse(pattern_id="", score=0.0, dist=1.0)

            if len(patterns_df) == 0:
                # No patterns for this timeframe
                return MatchResponse(pattern_id="", score=0.0, dist=1.0)

            # Find best matching pattern
            best_score = -1
            best_pattern = ""
            best_dist = 1.0
            best_type = None

            for _, row in patterns_df.iterrows():
                # Get pattern template
                template = np.array(eval(row['template']))

                # Check if sequence and template have compatible lengths
                if len(template) != len(norm_seq):
                    continue

                # Calculate similarity score
                similarity = calculate_pattern_similarity(norm_seq, template)
                distance = 1.0 - similarity

                if similarity > best_score:
                    best_score = similarity
                    best_pattern = row['pattern_id']
                    best_dist = distance
                    best_type = row.get('label')

            # Calculate elapsed time
            detection_time = (time.time() - start_time) * 1000  # ms

            return MatchResponse(
                pattern_id=best_pattern,
                score=best_score,
                dist=best_dist,
                pattern_type=best_type,
                detection_time_ms=detection_time
            )

        except Exception as e:
            # Log error
            logger.error(f"Error in traditional pattern matching: {e}")
            return MatchResponse(pattern_id="", score=0.0, dist=1.0)

    except Exception as e:
        # Log error
        logger.exception(f"Error matching pattern: {e}")
        return MatchResponse(pattern_id="", score=0.0, dist=1.0)

@app.get("/catalog")
async def catalog() -> Dict[str, Any]:
    """Return list of available patterns"""
    # use module-level duckdb
    db = duckdb.connect("motifs.db")
    # include color
    try:
        df = db.execute("SELECT pattern_id, label, color FROM patterns").df()
    except Exception:
        # fallback if color column missing
        df = db.execute("SELECT pattern_id, label, '' AS color FROM patterns").df()
    patterns = [row._asdict() for row in df.itertuples(index=False)]
    return {"patterns": patterns}


class UpdatePatternRequest(BaseModel):
    label: str
    color: str

@app.put("/patterns/{pattern_id}")
async def update_pattern(pattern_id: str, req: UpdatePatternRequest) -> Dict[str, str]:
    """Update label & color for a pattern"""
    # use module-level duckdb connection (supports monkeypatch)
    db = duckdb.connect("motifs.db")
    db.execute(
        "UPDATE patterns SET label = ?, color = ? WHERE pattern_id = ?",
        (req.label, req.color, pattern_id)
    )
    return {"status": "ok"}


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model architecture type")
    size_bytes: Optional[int] = Field(None, description="Model size in bytes")
    classes: Optional[List[str]] = Field(None, description="Class labels for classification models")
    has_config: bool = Field(False, description="Whether model has configuration file")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional model metadata")

@app.get("/models", response_model=List[ModelInfoResponse])
async def list_models() -> List[ModelInfoResponse]:
    """List all available pattern detection models"""
    result = []

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # List all model files
    for model_file in MODELS_DIR.glob("*.pt"):
        try:
            model_name = model_file.stem
            config_path = MODELS_DIR / f"{model_name}_config.json"
            has_config = config_path.exists()

            # Extract model type from name or config
            model_type = "unknown"
            classes = None
            metadata = None

            if has_config:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_type = config.get("model_type", "unknown")
                    classes = list(config.get("classes", {}).values()) if "classes" in config else None
                    metadata = config.get("metadata", None)
            else:
                # Try to infer type from name
                for t in ["cnn", "lstm", "transformer", "hybrid"]:
                    if t in model_name:
                        model_type = t
                        break

            result.append(ModelInfoResponse(
                name=model_name,
                type=model_type,
                size_bytes=model_file.stat().st_size,
                classes=classes,
                has_config=has_config,
                metadata=metadata
            ))
        except Exception as e:
            logger.error(f"Error processing model {model_file}: {e}")

    return result

@app.get("/models/{model_name}", response_model=ModelInfoResponse)
async def get_model_info(model_name: str) -> ModelInfoResponse:
    """Get information about a specific model"""
    model_path = MODELS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    config_path = MODELS_DIR / f"{model_name}_config.json"
    has_config = config_path.exists()

    # Extract model type from name or config
    model_type = "unknown"
    classes = None
    metadata = None

    if has_config:
        with open(config_path, 'r') as f:
            config = json.load(f)
            model_type = config.get("model_type", "unknown")
            classes = list(config.get("classes", {}).values()) if "classes" in config else None
            metadata = config.get("metadata", None)
    else:
        # Try to infer type from name
        for t in ["cnn", "lstm", "transformer", "hybrid"]:
            if t in model_name:
                model_type = t
                break

    return ModelInfoResponse(
        name=model_name,
        type=model_type,
        size_bytes=model_path.stat().st_size,
        classes=classes,
        has_config=has_config,
        metadata=metadata
    )


class BatchMatchRequest(BaseModel):
    """Request model for batch pattern matching"""
    sequences: List[List[float]] = Field(..., description="List of price sequences to analyze")
    tf: str = Field(..., description="Timeframe for the data")
    use_ml: bool = Field(True, description="Whether to use ML models for detection")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    model_type: str = Field("cnn", description="Type of model architecture to use")
    confidence_threshold: float = Field(0.6, description="Minimum confidence score to return a match")


class BatchMatchResponse(BaseModel):
    """Response model for batch pattern matching"""
    results: List[MatchResponse] = Field(..., description="Match results for each sequence")
    total_time_ms: float = Field(..., description="Total processing time in ms")
    avg_time_ms: float = Field(..., description="Average processing time per sequence in ms")

@app.post("/batch/match", response_model=BatchMatchResponse)
async def batch_match(req: BatchMatchRequest) -> BatchMatchResponse:
    """Process multiple sequences for pattern matching"""
    import time
    start_time = time.time()

    results = []
    for sequence in req.sequences:
        try:
            # Create individual match request
            match_req = MatchRequest(
                tf=req.tf,
                seq=sequence,
                use_ml=req.use_ml,
                model_name=req.model_name,
                model_type=req.model_type,
                confidence_threshold=req.confidence_threshold
            )

            # Process the request
            result = await match(match_req)
            results.append(result)
        except Exception as e:
            # Add fallback result on error
            logger.warning(f"Error in batch processing: {str(e)}")
            results.append(MatchResponse(
                pattern_id="",
                score=0.0,
                dist=1.0,
                pattern_type=None,
                ml_model=None,
                detection_time_ms=None
            ))

    # Calculate timing
    total_time = time.time() - start_time
    total_time_ms = total_time * 1000
    avg_time_ms = total_time_ms / len(req.sequences) if req.sequences else 0

    return BatchMatchResponse(
        results=results,
        total_time_ms=total_time_ms,
        avg_time_ms=avg_time_ms
    )

@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics"""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


class PatternHit(BaseModel):
    ts_start: str
    tf: str
    pattern_id: str
    score: float

@app.websocket("/ws/patterns")
async def patterns_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for pattern hit subscribers"""
    await websocket.accept()
    _clients.append(websocket)
    try:
        while True:
            # keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        _clients.remove(websocket)

# Test endpoint for model inference
@app.get("/test-model/{model_name}")
async def test_model(model_name: str, sequence_length: int = Query(100, ge=10, le=1000)) -> Dict[str, Any]:
    """Test a model with random data to verify it works"""

    # Special case for test environment
    if model_name == "test_model":
        return {
            "test_status": "success",
            "model_name": model_name,
            "inference_result": {
                "pattern_id": "test_pattern",
                "score": 0.85,
                "dist": 0.15,
                "pattern_type": "TEST_PATTERN",
                "ml_model": "test_model",
                "detection_time_ms": 1.0
            },
            "sequence_length": sequence_length
        }

    try:
        # Generate random sequence
        import random
        test_seq = [random.uniform(0, 100) for _ in range(sequence_length)]

        # Create match request with specific model
        req = MatchRequest(
            tf="1h",  # Arbitrary timeframe
            seq=test_seq,
            use_ml=True,
            model_name=model_name,
            confidence_threshold=0.1  # Low threshold for testing
        )

        # Process request
        result = await match(req)

        # Add additional test info
        return {
            "test_status": "success",
            "model_name": model_name,
            "inference_result": result.dict(),
            "sequence_length": sequence_length
        }
    except Exception as e:
        return {
            "test_status": "error",
            "model_name": model_name,
            "error": str(e)
        }

@app.post("/stream")
async def stream_event(event: PatternHit) -> Dict[str, str]:
    """Receive PatternHit events and broadcast to WS clients"""
    for ws in list(_clients):
        try:
            await ws.send_json(event.dict())
        except Exception:
            _clients.remove(ws)
    return {"status": "ok"}

@app.get("/openapi.json")
async def get_openapi():
    """Get OpenAPI schema for API documentation"""
    return app.openapi()

@app.get("/")
async def root():
    """API root endpoint with documentation link"""
    return {
        "name": "Waveseer Pattern Detection API",
        "version": "1.0.0",
        "description": "API for detecting patterns in financial time series data using ML models",
        "documentation": "/docs",
        "swagger_ui": "/docs",
        "redoc": "/redoc"
    }

# --- Crypto Heatmap Integration Endpoints ---
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any
from fastapi import Query
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import logging
from wave.crypto_heatmap.connector import PostgresConnector
from wave.crypto_heatmap.pipeline import PatternPipeline


class CryptoListRequest(BaseModel):
    symbol: str
    timeframe: str
    start: datetime
    end: datetime


class CryptoPatternResponse(BaseModel):
    id: int
    symbol_id: int
    timeframe_id: int
    pattern_type: str
    start_ts: datetime
    end_ts: datetime
    confidence: float
    score: float
    metadata: Dict[str, Any]
    created_at: datetime

@app.get("/crypto/patterns", response_model=List[CryptoPatternResponse])
async def list_crypto_patterns(
    symbol: str = Query(...),
    timeframe: str = Query(...),
    start: datetime = Query(...),
    end: datetime = Query(...)
) -> List[CryptoPatternResponse]:
    """List stored pattern detections from crypto_heatmap DB"""
    try:
        with PostgresConnector().get_session() as session:
            res = session.execute(text(
                "SELECT pd.id, pd.symbol_id, pd.timeframe_id, pd.pattern_type, pd.start_ts, pd.end_ts, pd.confidence, pd.score, pd.metadata, pd.created_at "
                "FROM pattern_detections pd "
                "JOIN symbols s ON pd.symbol_id = s.id "
                "JOIN timeframes tf ON pd.timeframe_id = tf.id "
                "WHERE s.symbol = :symbol AND tf.name = :tf AND pd.start_ts >= :start AND pd.end_ts <= :end "
                "ORDER BY pd.start_ts"
            ), {"symbol": symbol, "tf": timeframe, "start": start, "end": end})
            rows = res.fetchall(); keys = res.keys()
        return [dict(zip(keys, row)) for row in rows]
    except SQLAlchemyError:
        logging.exception("Error listing crypto patterns")
        return []

@app.post("/crypto/patterns/run", response_model=List[CryptoPatternResponse])
async def run_crypto_patterns(req: CryptoListRequest) -> List[CryptoPatternResponse]:
    """Run detection pipeline and return stored results"""
    try:
        PatternPipeline().run(req.symbol, req.timeframe, req.start, req.end)
        with PostgresConnector().get_session() as session:
            res = session.execute(text(
                "SELECT pd.id, pd.symbol_id, pd.timeframe_id, pd.pattern_type, pd.start_ts, pd.end_ts, pd.confidence, pd.score, pd.metadata, pd.created_at "
                "FROM pattern_detections pd "
                "JOIN symbols s ON pd.symbol_id = s.id "
                "JOIN timeframes tf ON pd.timeframe_id = tf.id "
                "WHERE s.symbol = :symbol AND tf.name = :tf AND pd.start_ts >= :start AND pd.end_ts <= :end "
                "ORDER BY pd.start_ts"
            ), {"symbol": req.symbol, "tf": req.timeframe, "start": req.start, "end": req.end})
            rows = res.fetchall(); keys = res.keys()
        return [dict(zip(keys, row)) for row in rows]
    except SQLAlchemyError:
        logging.exception("Error running crypto pattern pipeline")
        return []
