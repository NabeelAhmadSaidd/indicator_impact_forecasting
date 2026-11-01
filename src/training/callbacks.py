"""
Training Callbacks Module

This module provides custom Keras callbacks for training deep learning models,
including early stopping, model checkpointing, and learning rate scheduling.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import logging

import tensorflow as tf
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)

from src.utils.config import Config


logger = logging.getLogger(__name__)


class MetricsLogger(Callback):
    """
    Custom callback for logging training metrics.
    
    Logs metrics to console and optionally to a file in JSON format.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        experiment_name: str = "experiment",
        log_interval: int = 1,
        verbose: bool = True
    ):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory for log files.
            experiment_name: Name of the experiment.
            log_interval: Log every N epochs.
            verbose: Whether to print to console.
        """
        super().__init__()
        self.log_dir = Path(log_dir) if log_dir else None
        self.experiment_name = experiment_name
        self.log_interval = log_interval
        self.verbose = verbose
        self.history = []
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        """Called at the start of training."""
        self.start_time = datetime.now()
        if self.verbose:
            logger.info(f"Training started at {self.start_time}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        logs = logs or {}
        
        # Record metrics
        epoch_metrics = {'epoch': epoch + 1}
        epoch_metrics.update(logs)
        self.history.append(epoch_metrics)
        
        # Log to console
        if self.verbose and (epoch + 1) % self.log_interval == 0:
            metrics_str = " | ".join([
                f"{k}: {v:.6f}" for k, v in logs.items()
            ])
            logger.info(f"Epoch {epoch + 1:4d} | {metrics_str}")
    
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        if self.verbose:
            logger.info(f"Training completed in {duration}")
        
        # Save history to file
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            json_path = self.log_dir / f"{self.experiment_name}_history.json"
            with open(json_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # Save as CSV
            csv_path = self.log_dir / f"{self.experiment_name}_history.csv"
            pd.DataFrame(self.history).to_csv(csv_path, index=False)


class BestModelTracker(Callback):
    """
    Callback to track the best model during training.
    
    Keeps track of the best metric value and corresponding epoch.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Initialize the tracker.
        
        Args:
            monitor: Metric to monitor.
            mode: 'min' or 'max'.
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.best_epoch = 0
        self.best_weights = None
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.mode == 'min':
            if current < self.best_value:
                self.best_value = current
                self.best_epoch = epoch + 1
                self.best_weights = self.model.get_weights()
        else:
            if current > self.best_value:
                self.best_value = current
                self.best_epoch = epoch + 1
                self.best_weights = self.model.get_weights()
    
    def get_best_info(self) -> Dict[str, Any]:
        """Get information about the best model."""
        return {
            'best_epoch': self.best_epoch,
            'best_value': self.best_value,
            'monitor': self.monitor
        }


class GradientMonitor(Callback):
    """
    Callback to monitor gradient norms during training.
    
    Useful for detecting vanishing/exploding gradients.
    """
    
    def __init__(self, log_interval: int = 10):
        """
        Initialize the gradient monitor.
        
        Args:
            log_interval: Log every N batches.
        """
        super().__init__()
        self.log_interval = log_interval
        self.gradient_norms = []
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch."""
        if batch % self.log_interval == 0:
            # Get gradient norms
            weights = self.model.trainable_weights
            norms = [tf.norm(w).numpy() for w in weights]
            avg_norm = np.mean(norms)
            self.gradient_norms.append(avg_norm)


def get_callbacks(
    config: Optional[Config] = None,
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    experiment_name: str = "experiment",
    monitor: str = "val_loss",
    mode: str = "min"
) -> List[Callback]:
    """
    Get a list of callbacks based on configuration.
    
    Args:
        config: Configuration object.
        checkpoint_dir: Directory for model checkpoints.
        log_dir: Directory for logs.
        experiment_name: Name of the experiment.
        monitor: Metric to monitor.
        mode: 'min' or 'max' for metric optimization.
        
    Returns:
        List of Keras callbacks.
    """
    if config is None:
        config = Config.load()
    
    callbacks = []
    
    # Early Stopping
    early_stopping_config = config.base.get('training', {}).get('early_stopping', {})
    if early_stopping_config.get('enabled', True):
        callbacks.append(EarlyStopping(
            monitor=early_stopping_config.get('monitor', monitor),
            patience=early_stopping_config.get('patience', 15),
            min_delta=early_stopping_config.get('min_delta', 0.0001),
            mode=early_stopping_config.get('mode', mode),
            restore_best_weights=early_stopping_config.get('restore_best_weights', True),
            verbose=1
        ))
    
    # Model Checkpoint
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_config = config.base.get('checkpoints', {})
        callbacks.append(ModelCheckpoint(
            filepath=str(checkpoint_path / "best_model.keras"),
            monitor=checkpoint_config.get('monitor', monitor),
            mode=checkpoint_config.get('mode', mode),
            save_best_only=checkpoint_config.get('save_best_only', True),
            save_weights_only=checkpoint_config.get('save_weights_only', False),
            verbose=1
        ))

        # Always maintain a latest checkpoint for resuming mid-training
        callbacks.append(ModelCheckpoint(
            filepath=str(checkpoint_path / "last_model.keras"),
            monitor=checkpoint_config.get('monitor', monitor),
            mode=checkpoint_config.get('mode', mode),
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
            save_freq='epoch'
        ))
    
    # Learning Rate Scheduler
    lr_config = config.base.get('training', {}).get('lr_scheduler', {})
    if lr_config.get('enabled', True):
        callbacks.append(ReduceLROnPlateau(
            monitor=monitor,
            factor=lr_config.get('factor', 0.5),
            patience=lr_config.get('patience', 5),
            min_lr=lr_config.get('min_lr', 1e-6),
            verbose=1
        ))
    
    # CSV Logger
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        history_file = log_path / "history.csv"
        callbacks.append(CSVLogger(
            str(history_file),
            separator=',',
            append=history_file.exists()
        ))
    
    # Custom Metrics Logger
    callbacks.append(MetricsLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        verbose=True
    ))
    
    # Best Model Tracker
    callbacks.append(BestModelTracker(monitor=monitor, mode=mode))
    
    return callbacks


def get_early_stopping(
    patience: int = 15,
    min_delta: float = 0.0001,
    monitor: str = "val_loss",
    mode: str = "min",
    restore_best_weights: bool = True
) -> EarlyStopping:
    """
    Get an early stopping callback.
    
    Args:
        patience: Number of epochs with no improvement to wait.
        min_delta: Minimum change to qualify as improvement.
        monitor: Metric to monitor.
        mode: 'min' or 'max'.
        restore_best_weights: Whether to restore best weights.
        
    Returns:
        EarlyStopping callback.
    """
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        restore_best_weights=restore_best_weights,
        verbose=1
    )


def get_model_checkpoint(
    filepath: str,
    monitor: str = "val_loss",
    mode: str = "min",
    save_best_only: bool = True
) -> ModelCheckpoint:
    """
    Get a model checkpoint callback.
    
    Args:
        filepath: Path to save the model.
        monitor: Metric to monitor.
        mode: 'min' or 'max'.
        save_best_only: Only save when metric improves.
        
    Returns:
        ModelCheckpoint callback.
    """
    return ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        verbose=1
    )


def get_lr_scheduler(
    monitor: str = "val_loss",
    factor: float = 0.5,
    patience: int = 5,
    min_lr: float = 1e-6
) -> ReduceLROnPlateau:
    """
    Get a learning rate scheduler callback.
    
    Args:
        monitor: Metric to monitor.
        factor: Factor to reduce LR by.
        patience: Epochs to wait before reducing.
        min_lr: Minimum learning rate.
        
    Returns:
        ReduceLROnPlateau callback.
    """
    return ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        verbose=1
    )


if __name__ == "__main__":
    # Test callbacks
    callbacks = get_callbacks(
        checkpoint_dir="./test_checkpoints",
        log_dir="./test_logs",
        experiment_name="test_run"
    )
    
    print(f"Created {len(callbacks)} callbacks:")
    for cb in callbacks:
        print(f"  - {cb.__class__.__name__}")
