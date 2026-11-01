"""
Trainer Module

This module provides the main training logic for deep learning models,
including experiment management, training loops, and result saving.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import logging

import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils.config import Config, save_yaml, get_project_root
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed, save_seed
from src.models import get_model, build_model
from src.data.loader import load_processed_data
from src.training.callbacks import get_callbacks, BestModelTracker
from src.evaluation.metrics import calculate_all_metrics


logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for managing model training experiments.
    
    Handles the complete training workflow including:
    - Data loading
    - Model building
    - Training with callbacks
    - Evaluation
    - Result saving
    
    Attributes:
        config: Configuration object.
        model_type: Type of model ('lstm' or 'gru').
        model: Model instance.
        history: Training history.
    """
    
    def __init__(
        self,
        model_type: str = 'lstm',
        config: Optional[Config] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model_type: Type of model ('lstm' or 'gru').
            config: Configuration object.
        """
        self.model_type = model_type.lower()
        self.config = config if config is not None else Config.load(model_type=model_type)
        self.model = None
        self.history = None
        self.experiment_logger = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        callbacks: Optional[List] = None,
        experiment_name: str = "experiment",
        checkpoint_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            epochs: Number of epochs.
            batch_size: Batch size.
            callbacks: Custom callbacks (optional).
            experiment_name: Name for this experiment.
            checkpoint_dir: Directory for checkpoints.
            log_dir: Directory for logs.
            verbose: Verbosity level.
            
        Returns:
            Training history dictionary.
        """
        # Get training parameters
        training_config = self.config.base.get('training', {})
        epochs = epochs or training_config.get('epochs', 100)
        batch_size = batch_size or training_config.get('batch_size', 32)
        
        # Build model
        input_shape = X_train.shape[1:]
        self.model = get_model(self.model_type, config=self.config.model_config)
        self.model.build(input_shape)
        
        if verbose:
            self.model.summary()
        
        # Set up callbacks
        if callbacks is None:
            callbacks = get_callbacks(
                config=self.config,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                experiment_name=experiment_name
            )
        
        # Train model
        logger.info(f"Training {self.model_type.upper()} model for {epochs} epochs")
        
        self.history = self.model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Save final model
        if checkpoint_dir is not None:
            final_model_path = Path(checkpoint_dir) / "final_model.keras"
            self.model.save(final_model_path)
            logger.info(f"Saved final model to {final_model_path}")
        
        # Get best model info from tracker
        best_info = None
        for cb in callbacks:
            if isinstance(cb, BestModelTracker):
                best_info = cb.get_best_info()
                break
        
        return {
            'history': self.history,
            'best_info': best_info
        }
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scaler=None,
        target_column_idx: int = 3
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Input features.
            y: True targets.
            scaler: Scaler for inverse transform.
            target_column_idx: Index of target column.
            
        Returns:
            Dictionary of metrics.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.model.predict(X).flatten()
        
        # Inverse transform if scaler provided
        if scaler is not None:
            from src.data.preprocessing import inverse_scale
            n_features = scaler.n_features_in_
            y = inverse_scale(y, scaler, target_column_idx, n_features)
            y_pred = inverse_scale(y_pred, scaler, target_column_idx, n_features)
        
        # Calculate metrics
        metrics = calculate_all_metrics(y, y_pred)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Predictions array.
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        self.model = get_model(self.model_type)
        self.model.load(filepath)


class ExperimentRunner:
    """
    Runner for conducting complete experiments.
    
    Manages multiple training runs with different seeds,
    saves all results, and aggregates metrics.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the experiment runner.
        
        Args:
            config: Configuration object.
        """
        self.config = config if config is not None else Config.load()
        self.results = []
    
    def run_experiment(
        self,
        asset_name: str,
        group_name: str,
        model_type: str,
        run_id: int = 1
    ) -> Dict[str, Any]:
        """
        Run a single experiment.
        
        Args:
            asset_name: Name of the asset.
            group_name: Name of the feature group.
            model_type: Type of model.
            run_id: Run identifier.
            
        Returns:
            Experiment results dictionary.
        """
        # Set seed
        seed = 42  # Use single global seed
        set_seed(seed)
        
        # Set up experiment paths using new hierarchical structure
        # experiments/category/asset/group/model/
        experiment_dir = self.config.get_experiment_path(asset_name, group_name, model_type)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_dir = str(experiment_dir)  # Use experiment dir for checkpoints
        
        # Load data
        data = load_processed_data(asset_name, group_name, self.config)
        
        if not data:
            raise ValueError(f"No processed data found for {asset_name}/{group_name}")
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Create trainer
        trainer = Trainer(model_type=model_type, config=self.config)
        
        # Train
        experiment_name = f"{asset_name}_{group_name}_{model_type}_run{run_id:03d}"
        
        train_result = trainer.train(
            X_train, y_train,
            X_val, y_val,
            experiment_name=experiment_name,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(experiment_dir)
        )
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(X_test, y_test)
        
        # Get predictions
        y_pred = trainer.predict(X_test).flatten()
        
        # Save results
        results = {
            'asset': asset_name,
            'group': group_name,
            'model': model_type,
            'run_id': run_id,
            'seed': seed,
            'metrics': test_metrics,
            'history': train_result['history'],
            'best_info': train_result.get('best_info')
        }
        
        # Save to files
        self._save_experiment_results(
            results, y_test, y_pred,
            experiment_dir, seed
        )
        
        self.results.append(results)
        
        return results
    
    def _save_experiment_results(
        self,
        results: Dict[str, Any],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        experiment_dir: Path,
        seed: int
    ) -> None:
        """Save experiment results to files."""
        from src.evaluation.plots import plot_training_history
        
        # Create training history plot
        if results['history'] and hasattr(results['history'], 'history'):
            history_dict = results['history'].history
            fig = plot_training_history(
                history_dict,
                metrics=['loss', 'val_loss'],
                title=f"Training History - {results['asset']} {results['group']} {results['model']}",
                save_path=str(experiment_dir / 'training_history.png')
            )
            plt.close(fig)
    
    def run_ablation_study(
        self,
        assets: Optional[List[str]] = None,
        groups: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        n_runs: int = 5
    ) -> pd.DataFrame:
        """
        Run complete ablation study.
        
        Args:
            assets: List of assets to evaluate.
            groups: List of feature groups.
            models: List of model types.
            n_runs: Number of runs per configuration.
            
        Returns:
            DataFrame with all results.
        """
        if assets is None:
            assets = self.config.get_all_assets()
        
        if groups is None:
            groups = list(self.config.experiments['groups'].keys())
        
        if models is None:
            models = self.config.experiments.get('models', ['lstm', 'gru'])
        
        all_results = []
        total = len(assets) * len(groups) * len(models) * n_runs
        current = 0
        
        for asset in assets:
            for group in groups:
                for model_type in models:
                    for run_id in range(1, n_runs + 1):
                        current += 1
                        logger.info(
                            f"[{current}/{total}] Running: "
                            f"{asset}/{group}/{model_type}/run_{run_id}"
                        )
                        
                        try:
                            result = self.run_experiment(
                                asset_name=asset,
                                group_name=group,
                                model_type=model_type,
                                run_id=run_id
                            )
                            
                            # Flatten results for DataFrame
                            flat_result = {
                                'asset': result['asset'],
                                'group': result['group'],
                                'model': result['model'],
                                'run_id': result['run_id'],
                                'seed': result['seed'],
                                **result['metrics']
                            }
                            all_results.append(flat_result)
                            
                        except Exception as e:
                            logger.error(f"Experiment failed: {e}")
        
        return pd.DataFrame(all_results)
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of all results.
        
        Returns:
            DataFrame with mean and std of metrics.
        """
        if not self.results:
            return pd.DataFrame()
        
        # Flatten results
        flat_results = []
        for r in self.results:
            flat_results.append({
                'asset': r['asset'],
                'group': r['group'],
                'model': r['model'],
                **r['metrics']
            })
        
        df = pd.DataFrame(flat_results)
        
        # Group and aggregate
        summary = df.groupby(['asset', 'group', 'model']).agg(['mean', 'std'])
        
        return summary


def train_single_model(
    asset_name: str,
    group_name: str,
    model_type: str,
    seed: int = 42,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Convenience function to train a single model.
    
    Args:
        asset_name: Name of the asset.
        group_name: Feature group name.
        model_type: Model type ('lstm' or 'gru').
        seed: Random seed.
        epochs: Number of epochs.
        batch_size: Batch size.
        verbose: Verbosity level.
        
    Returns:
        Training results dictionary.
    """
    set_seed(seed)
    
    config = Config.load(model_type=model_type)
    runner = ExperimentRunner(config)
    
    return runner.run_experiment(
        asset_name=asset_name,
        group_name=group_name,
        model_type=model_type,
        run_id=1,
        seed=seed
    )


if __name__ == "__main__":
    # Test trainer
    from src.utils.seed import set_seed
    set_seed(42)
    
    # Create sample data
    n_samples = 1000
    window_size = 60
    n_features = 5
    
    X_train = np.random.randn(800, window_size, n_features)
    y_train = np.random.randn(800)
    X_val = np.random.randn(100, window_size, n_features)
    y_val = np.random.randn(100)
    X_test = np.random.randn(100, window_size, n_features)
    y_test = np.random.randn(100)
    
    # Test trainer
    trainer = Trainer(model_type='lstm')
    result = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=2,
        verbose=1
    )
    
    # Test evaluation
    metrics = trainer.evaluate(X_test, y_test)
    print(f"Test metrics: {metrics}")
