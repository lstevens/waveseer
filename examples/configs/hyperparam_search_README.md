# Hyperparameter Search for Pattern Detection Models

This README provides instructions for using the hyperparameter search functionality to optimize pattern detection models in Waveseer.

## Quick Start

To run a hyperparameter search:

```bash
python -m wave.ml.train.train_patterns \
  --hyperparam-search \
  --config examples/configs/cnn_config.yaml \
  --param-space examples/configs/cnn_param_space.json \
  --search-type random \
  --n-trials 20 \
  --search-metric accuracy \
  --search-mode max \
  --search-dir results/hyperparam_search_cnn
```

## Parameter Space Definition

Parameter spaces are defined in JSON files with the following structure:

```json
{
  "parameter.path": [value1, value2, value3],
  "another.parameter.path": [value1, value2]
}
```

Each key represents a path to a parameter in the configuration object using dot notation. For example:
- `model.hidden_size` refers to `config.model.hidden_size`
- `optimizer.learning_rate` refers to `config.optimizer.learning_rate`

The values are lists of possible values to try for each parameter.

## Search Types

### Random Search

Random search samples parameter combinations randomly from the parameter space. This is often more efficient than grid search for high-dimensional spaces.

```bash
--search-type random
```

### Grid Search

Grid search evaluates all possible combinations of parameters systematically. This can be exhaustive for large parameter spaces.

```bash
--search-type grid
```

## Optimizing Different Metrics

You can optimize different evaluation metrics:

```bash
# Optimize for accuracy
--search-metric accuracy --search-mode max

# Optimize for F1 score
--search-metric f1 --search-mode max

# Optimize for loss (minimize)
--search-metric loss --search-mode min
```

## Search Results

Results are saved in the specified search directory:

- `best_config.yaml`: The best configuration found
- `results_summary.csv`: Summary of all trial results
- Trial-specific folders contain individual training runs

## Example Parameter Spaces

We provide example parameter spaces for different model architectures:

- `cnn_param_space.json`: Parameter space for CNN models
- `lstm_param_space.json`: Parameter space for LSTM models
- `transformer_param_space.json`: Parameter space for Transformer models

## Tips for Effective Hyperparameter Search

1. **Start Small**: Begin with a small number of trials to validate your setup
2. **Focus on Important Parameters**: Include parameters known to have significant effects
3. **Use Logarithmic Scales**: For learning rates and regularization terms, use logarithmically spaced values
4. **Reduce Training Epochs**: Set fewer epochs for faster iterations during search
5. **Use a Small Dataset**: For initial exploration, reduce dataset size for speed

## Customizing the Search Process

For advanced customization, you can directly use the hyperparameter search API:

```python
from wave.ml.train.config import load_config
from wave.ml.train.hyperparam_search import run_hyperparam_search

# Load base configuration
base_config = load_config("examples/configs/cnn_config.yaml")

# Define parameter space
param_space = {
    "model.hidden_size": [64, 128, 256],
    "optimizer.learning_rate": [0.0001, 0.001, 0.01]
}

# Run search
best_config, results = run_hyperparam_search(
    base_config=base_config,
    param_space=param_space,
    search_type="random",
    n_trials=10,
    metric="accuracy",
    mode="max"
)
```

## Testing Your Hyperparameter Search

To validate your hyperparameter search setup without running a full search, you can:

1. Reduce the number of trials (`--n-trials 2`)
2. Use a smaller dataset (modify your config file)
3. Reduce the number of epochs (modify your config file)

This allows you to verify the search process works correctly before committing to a longer search.
