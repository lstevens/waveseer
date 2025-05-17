# Crypto Heatmap: Integration Plan for Windsurfer

## 1. Architecture Overview
```
├── Backend (Python 3.11, Poetry, FastAPI)
│   ├── Core Engine: crypto_heatmap/
│   │   ├── indicators - Strategy scanning, technical analysis
│   │   ├── backtesting - Simulation engine
│   │   ├── memory - pgvector embedding storage
│   │   └── db - SQLAlchemy models, migrations
│   └── AI System: crypto_heatmap_roundtable/
│       ├── agents - Technical, Sentiment, Risk, Seer
│       ├── moderator - LangGraph orchestration
│       ├── tools - Tool dispatcher registry
│       └── ml - Feature engineering, model registry
├── Frontend (React, TypeScript, Vite)
│   ├── ChatPanel - Agent interactions
│   └── HeatmapCanvas - Market visualization
└── Infrastructure
    ├── Postgres - Core tables, pgvector
    ├── Redis - Task broker, WebSockets
    └── S3/Minio - Model storage
```

## 2. Core Components to Integrate

### ML Pipeline
- **DataFetcher**: Retrieves and cleans market data
- **FeatureEngineer**: Creates technical indicators for ML
- **ModelRegistry**: Manages model versions and lifecycle
- **PriceDirectionPredictor**: Makes real-time predictions

### Multi-Agent System
- **LangGraph Moderator**: Coordinates agent workflow
- **Enhanced Tool Registry**: Structured tool access for agents
- **Memory System**: Vector-based context retrieval

### WebSocket Integration
- Real-time event streams for UI updates
- Structured message schema for agent communication

## 3. Implementation Pattern Highlights

### Model Registry Pattern
```python
# Core structure
class ModelRegistry:
    def register(self, model, symbol, metrics, metadata):
        """Register a new model in development state"""
    
    def promote(self, model_id, target_state):
        """Transition model through lifecycle states"""
    
    def get_model(self, symbol, selector="latest_production"):
        """Retrieve model by specified criteria"""
```

### Agent Integration Pattern
```python
# Agent workflow in LangGraph
graph = StateGraph()
graph.add_node("technical_analysis", TechnicalAgent())
graph.add_node("sentiment_analysis", SentimentAgent())
graph.add_node("risk_assessment", RiskAgent())
graph.add_node("ml_prediction", SeerAgent())
graph.add_node("decision", DecisionNode())

# Connect nodes
graph.add_edge("technical_analysis", "sentiment_analysis")
graph.add_edge("sentiment_analysis", "ml_prediction")
graph.add_edge("ml_prediction", "risk_assessment")
graph.add_edge("risk_assessment", "decision")
```

### Tool Registration Pattern
```python
# Enhanced tool registry
@register_tool(
    name="fetch_price_data",
    description="Get historical price data",
    category=ToolCategory.DATA_RETRIEVAL,
    parameters=[
        ToolParameter("symbol", str, "Trading symbol"),
        ToolParameter("timeframe", str, "Chart timeframe"),
        ToolParameter("bars", int, "Number of bars", default=200)
    ]
)
def fetch_price_data(symbol, timeframe, bars=200):
    """Implementation of price data fetching"""
```

## 4. Key Lessons for Integration

1. Use adapter pattern to isolate third-party dependencies
2. Implement graceful degradation when optional modules unavailable
3. Fix circular imports with proper package structure
4. Wrap imports in try/except with fallbacks
5. Use typed dictionaries for complex data structures
6. Make tests deterministic with fixed random seeds
7. Handle API version changes in pandas/numpy

## 5. Integration Path

1. **Import ML Core**: Integrate DataFetcher, FeatureEngineer, ModelRegistry
2. **Connect Agents**: Adapt LangGraph moderator for Windsurfer agents
3. **Tool Registry**: Implement enhanced registry with Windsurfer tools
4. **Memory System**: Integrate pgvector-backed memory
5. **Real-time UI**: Connect WebSocket event streams

## 6. Shared Infrastructure

- **Model Storage**: Compatible model filesystem format
- **Postgres**: Shared schema design patterns
- **Redis**: Aligned pub/sub channel structure
- **Authentication**: Consistent auth protocol

## 7. First Steps

1. Create model adapter layer between systems
2. Design shared tool interface specification
3. Establish consistent agent message format
4. Implement pattern detection interface 