"""
Hyperparameter search for pattern detection models.

This module provides utilities for performing hyperparameter optimization
to find the best configuration for training pattern detection models.
"""

import os
import sys
import logging
import random
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch

from wave.ml.data.dataset import PatternDataset
from wave.ml.train.config import ExperimentConfig, save_config
from wave.ml.train.trainer import Trainer, create_model
from wave.ml.train.train_patterns import create_datasets


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HyperparamSearch:
    """Base class for hyperparameter search algorithms."""

    def __init__(
        self,
        base_config: ExperimentConfig,
        param_space: Dict[str, List[Any]],
        search_dir: str = "hyperparam_search",
        n_trials: int = 10,
        metric: str = "accuracy",
        mode: str = "max",
        seed: int = 42
    ):
        """
        Initialize hyperparameter search.

        Args:
            base_config: Base configuration to start from
            param_space: Dictionary mapping parameter paths to lists of values to try
                         Parameter paths should be dot-separated, e.g. "model.hidden_size"
            search_dir: Directory to save search results
            n_trials: Number of trials/configurations to evaluate
            metric: Metric to optimize (must be one of the metrics computed by Trainer)
            mode: "max" to maximize metric, "min" to minimize
            seed: Random seed for reproducibility
        """
        self.base_config = base_config
        self.param_space = param_space
        self.search_dir = Path(search_dir)
        self.n_trials = n_trials
        self.metric = metric
        self.mode = mode
        self.seed = seed

        # Create search directory
        self.search_dir.mkdir(parents=True, exist_ok=True)

        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Initialize results tracking
        self.results = []
        self.best_config = None
        self.best_metric = -float('inf') if mode == "max" else float('inf')

    def run(self) -> Tuple[ExperimentConfig, Dict[str, Any]]:
        """
        Run hyperparameter search.

        Returns:
            Best configuration and its results
        """
        logger.info(f"Starting hyperparameter search with {self.n_trials} trials")
        logger.info(f"Parameter space: {self.param_space}")

        # Create datasets once to reuse across trials
        train_dataset, val_dataset, test_dataset = create_datasets(self.base_config)

        for trial in range(self.n_trials):
            logger.info(f"Starting trial {trial+1}/{self.n_trials}")

            # Generate configuration for this trial
            config = self._get_trial_config(trial)

            # Update config with trial number
            config_name = f"{config.name}_trial_{trial+1}"
            config = replace(config, name=config_name)

            # Update checkpoint directory to avoid conflicts
            trial_checkpoint_dir = str(self.search_dir / f"trial_{trial+1}")
            config.train.checkpoint_dir = trial_checkpoint_dir

            # Save the trial configuration
            os.makedirs(trial_checkpoint_dir, exist_ok=True)
            config_path = os.path.join(trial_checkpoint_dir, "config.yaml")
            save_config(config, config_path)

            # Run trial and get results
            try:
                result = self._run_trial(config, train_dataset, val_dataset, test_dataset)

                # Get the metric value to optimize
                metric_value = result['test_metrics'][self.metric]

                # Add to results
                trial_result = {
                    'trial': trial + 1,
                    'config': config,
                    'metric': metric_value,
                    'results': result
                }
                self.results.append(trial_result)

                # Update best result if needed
                if ((self.mode == "max" and metric_value > self.best_metric) or
                    (self.mode == "min" and metric_value < self.best_metric)):
                    self.best_metric = metric_value
                    self.best_config = config
                    logger.info(f"New best {self.metric}: {metric_value:.4f}")

                logger.info(f"Trial {trial+1} complete. {self.metric}: {metric_value:.4f}")

            except Exception as e:
                logger.error(f"Trial {trial+1} failed: {e}")

        # Summarize and save results
        self._summarize_results()

        if self.best_config is not None:
            logger.info(f"Best {self.metric}: {self.best_metric:.4f}")
            best_config_path = str(self.search_dir / "best_config.yaml")
            save_config(self.best_config, best_config_path)
            logger.info(f"Best configuration saved to {best_config_path}")

            return self.best_config, self.results
        else:
            logger.error("No successful trials found")
            return self.base_config, self.results

    def _get_trial_config(self, trial: int) -> ExperimentConfig:
        """
        Generate configuration for a trial.

        Args:
            trial: Trial number

        Returns:
            Configuration for this trial
        """
        raise NotImplementedError("Subclasses must implement _get_trial_config")

    def _run_trial(
        self,
        config: ExperimentConfig,
        train_dataset: PatternDataset,
        val_dataset: PatternDataset,
        test_dataset: PatternDataset
    ) -> Dict[str, Any]:
        """
        Run a single trial.

        Args:
            config: Configuration for this trial
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset

        Returns:
            Trial results
        """
        # Create model
        model = create_model(config)

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

        # Test model
        test_metrics = trainer.test()
        results['test_metrics'] = test_metrics

        return results

    def _summarize_results(self) -> None:
        """Summarize search results."""
        if not self.results:
            logger.warning("No results to summarize")
            return

        # Create DataFrame with results
        data = []
        for result in self.results:
            row = {
                'trial': result['trial'],
                self.metric: result['metric']
            }

            # Add parameter values
            for param_path in self.param_space.keys():
                value = self._get_param_value(result['config'], param_path)
                row[param_path] = value

            data.append(row)

        df = pd.DataFrame(data)

        # Sort by metric
        if self.mode == "max":
            df = df.sort_values(self.metric, ascending=False)
        else:
            df = df.sort_values(self.metric, ascending=True)

        # Save summary
        summary_path = str(self.search_dir / "results_summary.csv")
        df.to_csv(summary_path, index=False)
        logger.info(f"Results summary saved to {summary_path}")

        # Log top results
        logger.info("\nTop results:")
        logger.info(df.head(5).to_string())

    def _get_param_value(self, config: ExperimentConfig, param_path: str) -> Any:
        """
        Get parameter value from configuration using dot path.

        Args:
            config: Configuration object
            param_path: Dot-separated parameter path, e.g. "model.hidden_size"

        Returns:
            Parameter value
        """
        parts = param_path.split('.')
        value = asdict(config)

        for part in parts:
            value = value[part]

        return value

    def _set_param_value(self, config: ExperimentConfig, param_path: str, value: Any) -> ExperimentConfig:
        """
        Set parameter value in configuration using dot path.

        Args:
            config: Configuration object
            param_path: Dot-separated parameter path, e.g. "model.hidden_size"
            value: Value to set

        Returns:
            Updated configuration
        """
        # Create a copy of the config to avoid modifying the original
        config_dict = asdict(config)

        parts = param_path.split('.')
        current = config_dict

        # Navigate to the last part
        for i, part in enumerate(parts[:-1]):
            current = current[part]

        # Set the value
        current[parts[-1]] = value

        # Create a new config object
        return ExperimentConfig.from_dict(config_dict)


class GridSearch(HyperparamSearch):
    """Grid search for hyperparameter optimization."""

    def _get_trial_config(self, trial: int) -> ExperimentConfig:
        """
        Generate configuration for a trial using grid search.

        Args:
            trial: Trial number

        Returns:
            Configuration for this trial
        """
        # Get all parameter combinations
        param_keys = list(self.param_space.keys())
        param_values = list(self.param_space.values())

        # Calculate total combinations
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)

        if trial >= total_combinations:
            logger.warning(f"Trial {trial+1} exceeds total combinations ({total_combinations})")
            # Fall back to random selection for remaining trials
            return self._get_random_config()

        # Find the combination for this trial
        combination_indices = []
        remaining = trial
        for values in reversed(param_values):
            n_values = len(values)
            index = remaining % n_values
            combination_indices.insert(0, index)
            remaining //= n_values

        # Create configuration with these parameter values
        config = copy.deepcopy(self.base_config)
        for i, param_path in enumerate(param_keys):
            param_value = param_values[i][combination_indices[i]]
            config = self._set_param_value(config, param_path, param_value)

        return config

    def _get_random_config(self) -> ExperimentConfig:
        """
        Generate a random configuration from the parameter space.

        Returns:
            Random configuration
        """
        config = copy.deepcopy(self.base_config)

        for param_path, param_values in self.param_space.items():
            param_value = random.choice(param_values)
            config = self._set_param_value(config, param_path, param_value)

        return config


class RandomSearch(HyperparamSearch):
    """Random search for hyperparameter optimization."""

    def _get_trial_config(self, trial: int) -> ExperimentConfig:
        """
        Generate configuration for a trial using random search.

        Args:
            trial: Trial number

        Returns:
            Configuration for this trial
        """
        config = copy.deepcopy(self.base_config)

        for param_path, param_values in self.param_space.items():
            param_value = random.choice(param_values)
            config = self._set_param_value(config, param_path, param_value)

        return config


def run_hyperparam_search(
    base_config: ExperimentConfig,
    param_space: Dict[str, List[Any]],
    search_type: str = "random",
    n_trials: int = 10,
    metric: str = "accuracy",
    mode: str = "max",
    search_dir: str = "hyperparam_search",
    seed: int = 42
) -> Tuple[ExperimentConfig, Dict[str, Any]]:
    """
    Run hyperparameter search.

    Args:
        base_config: Base configuration to start from
        param_space: Dictionary mapping parameter paths to lists of values to try
                     Parameter paths should be dot-separated, e.g. "model.hidden_size"
        search_type: Type of search, one of "random" or "grid"
        n_trials: Number of trials/configurations to evaluate
        metric: Metric to optimize (must be one of the metrics computed by Trainer)
        mode: "max" to maximize metric, "min" to minimize
        search_dir: Directory to save search results
        seed: Random seed for reproducibility

    Returns:
        Best configuration and search results
    """
    if search_type == "random":
        search = RandomSearch(
            base_config=base_config,
            param_space=param_space,
            search_dir=search_dir,
            n_trials=n_trials,
            metric=metric,
            mode=mode,
            seed=seed
        )
    elif search_type == "grid":
        search = GridSearch(
            base_config=base_config,
            param_space=param_space,
            search_dir=search_dir,
            n_trials=n_trials,
            metric=metric,
            mode=mode,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown search type: {search_type}")

    return search.run()
