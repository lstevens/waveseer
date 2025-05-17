"""
Example script demonstrating hyperparameter search for pattern detection models.

This script shows how to use the hyperparameter search API directly
for finding optimal model configurations.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from wave.ml.train.config import load_config, create_default_config, save_config
from wave.ml.train.hyperparam_search import run_hyperparam_search


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run hyperparameter search example."""
    # Create output directory
    output_dir = Path("examples/output/hyperparam_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load example configuration
    config_path = "examples/configs/cnn_config.yaml"
    if os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        base_config = load_config(config_path)
    else:
        logger.info("Creating default configuration")
        base_config = create_default_config()
        base_config.model.model_type = "cnn"

    # Modify configuration for faster testing
    base_config.data.n_samples = 500  # Fewer samples for speed
    base_config.train.epochs = 3      # Fewer epochs for speed

    # Define a simple parameter space
    param_space = {
        "model.hidden_size": [32, 64],
        "optimizer.learning_rate": [0.0001, 0.001],
        "data.batch_size": [16, 32]
    }

    # Save the test configuration
    test_config_path = output_dir / "test_config.yaml"
    save_config(base_config, str(test_config_path))

    # Run hyperparameter search
    logger.info("Starting hyperparameter search")
    try:
        best_config, results = run_hyperparam_search(
            base_config=base_config,
            param_space=param_space,
            search_type="grid",  # Use grid search for this small example
            n_trials=8,          # Should cover all combinations (2×2×2=8)
            metric="accuracy",
            mode="max",
            search_dir=str(output_dir),
            seed=42
        )

        logger.info("Hyperparameter search completed")
        logger.info(f"Best configuration saved to {output_dir}/best_config.yaml")

        # Print top 3 configurations by performance
        logger.info("\nTop 3 configurations:")
        for i, result in enumerate(sorted(results, key=lambda x: x['metric'], reverse=True)[:3]):
            logger.info(f"Rank {i+1}:")
            logger.info(f"  Accuracy: {result['metric']:.4f}")
            logger.info(f"  hidden_size: {result['config'].model.hidden_size}")
            logger.info(f"  learning_rate: {result['config'].optimizer.learning_rate}")
            logger.info(f"  batch_size: {result['config'].data.batch_size}")

    except Exception as e:
        logger.error(f"Hyperparameter search failed: {e}")
        raise


if __name__ == "__main__":
    main()
