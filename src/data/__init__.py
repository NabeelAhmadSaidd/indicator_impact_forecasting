"""
Data Package Initialization

This package provides utilities for loading, preprocessing, and 
generating windowed datasets from financial time series data.
"""

from src.data.loader import (
    DataLoader,
    load_csv,
    load_parquet,
    save_parquet,
    load_asset_data,
    load_all_assets,
    load_processed_data,
    save_processed_data
)

from src.data.preprocessing import (
    DataPreprocessor,
    clean_data,
    get_scaler,
    scale_data,
    inverse_scale,
    save_scaler,
    load_scaler,
    train_val_test_split,
    preprocess_asset
)

from src.data.window_generator import (
    WindowGenerator,
    DatasetGenerator,
    create_sequences,
    create_multi_step_sequences,
    prepare_dataset
)

__all__ = [
    # Loader
    'DataLoader',
    'load_csv',
    'load_parquet',
    'save_parquet',
    'load_asset_data',
    'load_all_assets',
    'load_processed_data',
    'save_processed_data',
    
    # Preprocessing
    'DataPreprocessor',
    'clean_data',
    'get_scaler',
    'scale_data',
    'inverse_scale',
    'save_scaler',
    'load_scaler',
    'train_val_test_split',
    'preprocess_asset',
    
    # Window Generator
    'WindowGenerator',
    'DatasetGenerator',
    'create_sequences',
    'create_multi_step_sequences',
    'prepare_dataset'
]
