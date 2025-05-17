# Pattern Detection Configuration Examples

This directory contains example configuration files for the Waveseer pattern detection system. Each file demonstrates a different model architecture with optimal settings for various pattern detection scenarios.

## Quick Start

To train a model using one of these configurations:

```bash
python -m wave.ml.train.train_patterns --config examples/configs/hybrid_config.yaml
```

## Available Configurations

### CNN (`cnn_config.yaml`)
- **Best for**: Short-term patterns, localized features
- **Strengths**: Fast training, good at detecting visual shapes
- **Sequence Length**: Medium (64 points)
- **Example Use Case**: Detecting flags, pennants, or other compact patterns

### LSTM (`lstm_config.yaml`)
- **Best for**: Temporal patterns, trend development
- **Strengths**: Remembers long-term dependencies, handles variable-length patterns
- **Sequence Length**: Long (120 points) 
- **Example Use Case**: Detecting head and shoulders, complex tops/bottoms

### Hybrid CNN-LSTM (`hybrid_config.yaml`)
- **Best for**: General-purpose pattern detection
- **Strengths**: Combines visual feature extraction with temporal analysis
- **Sequence Length**: Medium-long (100 points)
- **Example Use Case**: Balanced performance across all pattern types

### Transformer (`transformer_config.yaml`)
- **Best for**: Complex patterns with non-local relationships
- **Strengths**: Captures long-range dependencies, handles global context
- **Sequence Length**: Long (150 points)
- **Example Use Case**: Advanced pattern recognition, especially in noisy data

## Customizing Configurations

To customize a configuration for your needs:

1. Copy the most relevant example file
2. Modify parameters according to your specific requirements
3. Save with a new name
4. Train using your custom configuration

Key parameters to consider adjusting:
- `data.sequence_length`: Adjust based on your pattern timeframe
- `data.n_samples`: More samples generally improve results
- `data.pattern_types`: Select only the patterns you want to detect
- `model.hidden_size`: Larger for more complex patterns (increases model size)
- `optimizer.learning_rate`: Tune for your specific dataset

## Configuration Structure

Each configuration file contains these main sections:

- **Experiment Information**: Name, version, description, tags
- **Model Configuration**: Architecture and model-specific parameters
- **Optimizer Settings**: Learning rate, regularization, scheduling
- **Data Configuration**: Data generation, loading, and augmentation
- **Training Parameters**: Training loop, checkpointing, evaluation

For full details on all available configuration options, see the `wave.ml.train.config` module documentation.
