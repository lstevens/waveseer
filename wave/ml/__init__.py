"""
Machine Learning components for Waveseer pattern detection.

This module contains the ML pipeline for pattern detection:
1. Feature engineering - Extract features from time series data
2. Training - Train ML models on labeled pattern data
3. Inference - Use trained models to detect patterns in new data
"""

__all__ = [
    "feature_engineering",
    "train",
    "infer"
]
