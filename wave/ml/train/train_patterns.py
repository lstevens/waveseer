"""
Pattern detection model training script.

This script provides command-line functionality for training pattern detection models.
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any

import torch

from wave.patterns import PatternType
from wave.ml.data.synthetic import generate_synthetic_dataset, create_pytorch_dataset
from wave.ml.data.dataset import PatternDataset, load_dataset, split_dataset
from wave.ml.train.config import (
    ExperimentConfig,
    create_default_config,
    load_config,
    save_config,
    validate_config
)
from wave.ml.train.trainer import Trainer, create_model
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


def create_datasets(config: ExperimentConfig) -> Tuple[PatternDataset, PatternDataset, PatternDataset]:
    """
    Create or load datasets for training.

    Args:
        config: Experiment configuration

    Returns:
        Train, validation, and test datasets
    """
    data_config = config.data

    # Check if paths are provided
    if data_config.train_path and data_config.val_path and data_config.test_path:
        logger.info("Loading datasets from provided paths")
        train_dataset = load_dataset(data_config.train_path)
        val_dataset = load_dataset(data_config.val_path)
        test_dataset = load_dataset(data_config.test_path)
        return train_dataset, val_dataset, test_dataset

    # Generate synthetic dataset
    if data_config.synthetic_data:
        logger.info(f"Generating synthetic dataset with {data_config.n_samples} samples")

        # Convert pattern type names to enum values
        pattern_types = []
        for pt_name in data_config.pattern_types:
            try:
                pattern_types.append(PatternType[pt_name])
            except KeyError:
                logger.warning(f"Unknown pattern type: {pt_name}")

        # Generate dataset
        dataframes, labels = generate_synthetic_dataset(
            n_samples=data_config.n_samples,
            pattern_types=pattern_types,
            length=data_config.sequence_length,
            random_seed=config.train.seed
        )

        # Convert to PyTorch dataset
        features, labels_tensor = create_pytorch_dataset(dataframes, labels)
        dataset = PatternDataset(
            features=features,
            labels=labels_tensor,
            metadata={
                "config": {
                    "n_samples": data_config.n_samples,
                    "sequence_length": data_config.sequence_length,
                    "pattern_types": [pt.name for pt in pattern_types]
                }
            }
        )

        # Split dataset
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset=dataset,
            train_ratio=data_config.train_ratio,
            val_ratio=data_config.val_ratio,
            test_ratio=data_config.test_ratio,
            stratify=data_config.stratify,
            random_seed=config.train.seed
        )

        # Save datasets if checkpoint directory is configured
        if config.train.checkpoint_dir:
            checkpoint_dir = Path(config.train.checkpoint_dir)
            train_path = checkpoint_dir / "train_dataset"
            val_path = checkpoint_dir / "val_dataset"
            test_path = checkpoint_dir / "test_dataset"

            logger.info(f"Saving datasets to {checkpoint_dir}")
            train_dataset.save(train_path)
            val_dataset.save(val_path)
            test_dataset.save(test_path)

        return train_dataset, val_dataset, test_dataset

    # If we reach here, something is wrong with the configuration
    raise ValueError("Either dataset paths or synthetic_data must be specified")


def run_training(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run model training with the given configuration.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary with training results
    """
    # Validate configuration
    if not validate_config(config):
        raise ValueError("Invalid configuration")

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config)

    logger.info(
        f"Datasets created: "
        f"Train={len(train_dataset)} samples, "
        f"Val={len(val_dataset)} samples, "
        f"Test={len(test_dataset)} samples"
    )

    # Create model
    model = create_model(config)
    logger.info(f"Created {config.model.model_type} model")

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )

    # Train model
    results = trainer.train()

    # Export final model
    if config.train.checkpoint_dir:
        checkpoint_dir = Path(config.train.checkpoint_dir)
        export_path = checkpoint_dir / "model.pth"

        logger.info(f"Exporting model to {export_path}")
        torch.save(model.state_dict(), export_path)

        # Export TorchScript model
        script_path = checkpoint_dir / "model_script.pt"
        script_model = model.to_torchscript()
        torch.jit.save(script_model, script_path)

        logger.info(f"Exported TorchScript model to {script_path}")

    return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Train pattern detection models")
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default configuration file and exit"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config.yaml",
        help="Output path for created configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "lstm", "hybrid", "transformer"],
        help="Override model type in configuration"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs in configuration"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size in configuration"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate in configuration"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed in configuration"
    )

    # Hyperparameter search arguments
    parser.add_argument(
        "--hyperparam-search",
        action="store_true",
        help="Run hyperparameter search instead of single training run"
    )
    parser.add_argument(
        "--search-type",
        type=str,
        choices=["grid", "random"],
        default="random",
        help="Type of hyperparameter search to run"
    )
    parser.add_argument(
        "--param-space",
        type=str,
        help="Path to JSON file defining parameter space for search"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of trials for hyperparameter search"
    )
    parser.add_argument(
        "--search-metric",
        type=str,
        default="accuracy",
        help="Metric to optimize during search (e.g., accuracy, f1)"
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        choices=["max", "min"],
        default="max",
        help="Whether to maximize or minimize the metric"
    )
    parser.add_argument(
        "--search-dir",
        type=str,
        default="hyperparam_search",
        help="Directory to save search results"
    )

    args = parser.parse_args()

    # Create default configuration if requested
    if args.create_config:
        config = create_default_config()
        save_config(config, args.output)
        logger.info(f"Created default configuration at {args.output}")
        return

    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()

    # Override configuration with command-line arguments
    if args.model:
        config.model.model_type = args.model
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.optimizer.learning_rate = args.lr
    if args.seed:
        config.train.seed = args.seed

    # Run hyperparameter search if requested
    if args.hyperparam_search:
        if not args.param_space:
            logger.error("Parameter space file must be provided for hyperparameter search")
            sys.exit(1)

        # Load parameter space from JSON file
        with open(args.param_space, 'r') as f:
            param_space = json.load(f)

        logger.info(f"Running {args.search_type} hyperparameter search with {args.n_trials} trials")
        logger.info(f"Optimizing {args.search_metric} to be {args.search_mode}imized")

        # Run search
        try:
            best_config, search_results = run_hyperparam_search(
                base_config=config,
                param_space=param_space,
                search_type=args.search_type,
                n_trials=args.n_trials,
                metric=args.search_metric,
                mode=args.search_mode,
                search_dir=args.search_dir,
                seed=config.train.seed
            )

            logger.info(f"Hyperparameter search completed successfully")
            logger.info(f"Best {args.search_metric}: {search_results[0]['metric']:.4f}")
            logger.info(f"Best configuration saved to {args.search_dir}/best_config.yaml")
            return
        except Exception as e:
            logger.error(f"Hyperparameter search failed: {e}")
            raise

    # Run normal training
    try:
        results = run_training(config)
        logger.info(f"Training completed successfully")
        logger.info(f"Best validation metric: {results['best_val_metric']:.4f} at epoch {results['best_epoch']+1}")

        if 'test_metrics' in results and results['test_metrics']:
            logger.info(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
            logger.info(f"Test F1 score: {results['test_metrics']['f1']:.4f}")

        logger.info(f"Results saved to {results['checkpoint_dir']}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
