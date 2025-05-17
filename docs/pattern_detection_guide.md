# Pattern Detection Guide

This document explains the pattern detection system used in Waveseer, covering both rule-based and ML-based approaches.

## Overview

Waveseer's pattern detection system identifies technical chart patterns in financial time series data. The system supports:

1. **Rule-based detection** - Algorithmic identification of classic patterns like Head & Shoulders, Double Top, etc.
2. **ML-based detection** - Machine learning models to identify patterns that may be missed by rule-based approaches.

## Rule-Based Pattern Detection

Rule-based detection is implemented in `wave/patterns.py` and follows these steps:

1. **Peak and Trough Detection** - Identify significant price peaks and troughs using adaptive thresholds
2. **Pattern Recognition** - Apply specific rules to identify known patterns:
   - Head and Shoulders
   - Double Top
   - Double Bottom
   - And more...
3. **Pattern Scoring** - Calculate confidence score for each detected pattern
4. **Pattern Annotation** - Mark patterns on charts for visualization

### Example: Head and Shoulders Pattern

The Head and Shoulders pattern consists of:
- Left shoulder (first peak)
- Head (higher second peak)
- Right shoulder (third peak at similar height to first)
- Neckline (support line connecting troughs)

Detection pseudocode:
```
1. Find peaks and troughs in price data
2. Identify sequences of 3 peaks
3. Check if middle peak (head) is higher than the other two
4. Check if the shoulders are at roughly the same height
5. Check if the troughs form a valid neckline
6. Calculate confidence score based on pattern quality
```

## ML-Based Pattern Detection

The ML pipeline consists of:

1. **Feature Engineering** (`wave/ml/feature_engineering.py`):
   - Extract time-series features (technical indicators, statistical features)
   - Normalize and prepare data for model input

2. **Model Training** (`wave/ml/train.py`):
   - Train models on known patterns with labeled data
   - Support for multiple model architectures (CNN, LSTM, Transformer)
   - Hyperparameter optimization

3. **Inference** (`wave/ml/infer.py`):
   - Apply trained models to new data
   - Ensemble predictions from multiple models
   - Integration with rule-based detection

## Multiple Timeframe Detection

Waveseer can identify patterns across different timeframes:

1. **Single Timeframe** - Detect patterns within one timeframe (e.g., 1h charts)
2. **Multi-Timeframe** - Combine detection results from multiple timeframes for higher-confidence signals

To analyze patterns across timeframes:
- Run detection on each timeframe independently
- Use the `/match` API endpoint for each timeframe
- Aggregate results in your application logic

## Adding New Patterns

To add a new pattern:

1. Create a new function in `wave/patterns.py`:
   ```python
   def your_pattern_name(df: DataFrameType, threshold: float = 0.02) -> List[PatternMatch]:
       """
       Detect Your Pattern.
       
       Args:
           df: DataFrame with OHLCV data
           threshold: Sensitivity threshold
           
       Returns:
           List of PatternMatch objects
       """
       # Pattern detection logic here
       # ...
       return matches
   ```

2. Add the pattern to the `PatternType` enum
3. Register the pattern in the `detect_patterns()` function
4. Write tests in `tests/test_patterns.py`

## Pattern Similarity Calculation

Waveseer uses distance metrics to compare patterns:

```python
def calculate_pattern_similarity(price_seq: np.ndarray, template: np.ndarray) -> float:
    """Calculate similarity between price sequence and pattern template."""
    distance = np.sqrt(np.sum((price_seq - template) ** 2)) / len(price_seq)
    similarity = 1 - min(1, distance)
    return similarity
```

## Debugging Pattern Detection

To debug pattern detection:
1. Use the `--debug` flag: `python -m wave.seer --debug`
2. Check log output for detection steps
3. Visualize intermediate results using `wave/chart.py`
4. Use test fixtures in `tests/test_patterns.py` for controlled scenarios

## Performance Considerations

- Pattern detection can be computationally expensive on large datasets
- Use appropriate window sizes and thresholds based on timeframe
- Consider running detection in parallel for multiple symbols
- Cache detection results when possible
