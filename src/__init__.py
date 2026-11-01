"""
Source Package Initialization

This is the main source package for the indicator impact forecasting project.
"""

from src.utils import (
    Config,
    set_seed,
    setup_logger
)

from src.data import (
    DataLoader,
    DataPreprocessor,
    WindowGenerator,
    DatasetGenerator
)

from src.features import (
    add_all_indicators,
    add_indicators_for_group
)

from src.models import (
    get_model,
    build_model,
    LSTMForecaster,
    GRUForecaster
)

from src.training import (
    Trainer,
    ExperimentRunner
)

from src.evaluation import (
    calculate_all_metrics,
    plot_predictions
)

__version__ = '1.0.0'

__all__ = [
    # Utils
    'Config',
    'set_seed',
    'setup_logger',
    
    # Data
    'DataLoader',
    'DataPreprocessor',
    'WindowGenerator',
    'DatasetGenerator',
    
    # Features
    'add_all_indicators',
    'add_indicators_for_group',
    
    # Models
    'get_model',
    'build_model',
    'LSTMForecaster',
    'GRUForecaster',
    
    # Training
    'Trainer',
    'ExperimentRunner',
    
    # Evaluation
    'calculate_all_metrics',
    'plot_predictions'
]
