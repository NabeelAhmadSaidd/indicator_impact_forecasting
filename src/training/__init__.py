"""
Training Package Initialization

This package provides training utilities including callbacks,
training logic, and experiment management.
"""

from src.training.callbacks import (
    get_callbacks,
    get_early_stopping,
    get_model_checkpoint,
    get_lr_scheduler,
    MetricsLogger,
    BestModelTracker,
    GradientMonitor
)

from src.training.trainer import (
    Trainer,
    ExperimentRunner,
    train_single_model
)

__all__ = [
    # Callbacks
    'get_callbacks',
    'get_early_stopping',
    'get_model_checkpoint',
    'get_lr_scheduler',
    'MetricsLogger',
    'BestModelTracker',
    'GradientMonitor',
    
    # Trainer
    'Trainer',
    'ExperimentRunner',
    'train_single_model'
]
