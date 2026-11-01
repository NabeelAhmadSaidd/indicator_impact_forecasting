"""
Window Generator Module

This module provides utilities for creating sliding window sequences
from time series data for supervised learning.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
import logging

from src.utils.config import Config
from src.utils.logger import get_preprocessing_logger


logger = get_preprocessing_logger()


def create_sequences(
    data: np.ndarray,
    window_size: int,
    horizon: int = 1,
    target_column: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences for supervised learning.
    
    Args:
        data: Input data array of shape (n_samples, n_features).
        window_size: Number of timesteps for input sequences.
        horizon: Number of timesteps to predict ahead.
        target_column: Index of the target column (default: last column).
        
    Returns:
        Tuple of (X, y) arrays.
        X shape: (n_sequences, window_size, n_features)
        y shape: (n_sequences,) or (n_sequences, horizon)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    n_sequences = n_samples - window_size - horizon + 1
    
    if n_sequences <= 0:
        raise ValueError(
            f"Not enough data: {n_samples} samples for window_size={window_size}, "
            f"horizon={horizon}. Need at least {window_size + horizon} samples."
        )
    
    X = np.zeros((n_sequences, window_size, n_features))
    y = np.zeros((n_sequences,))
    
    for i in range(n_sequences):
        X[i] = data[i:i + window_size]
        y[i] = data[i + window_size + horizon - 1, target_column]
    
    return X, y


def create_multi_step_sequences(
    data: np.ndarray,
    window_size: int,
    horizon: int,
    target_column: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for multi-step prediction.
    
    Args:
        data: Input data array.
        window_size: Number of timesteps for input sequences.
        horizon: Number of timesteps to predict.
        target_column: Index of the target column.
        
    Returns:
        Tuple of (X, y) arrays.
        y shape: (n_sequences, horizon)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    n_sequences = n_samples - window_size - horizon + 1
    
    if n_sequences <= 0:
        raise ValueError(
            f"Not enough data for the specified window_size and horizon."
        )
    
    X = np.zeros((n_sequences, window_size, n_features))
    y = np.zeros((n_sequences, horizon))
    
    for i in range(n_sequences):
        X[i] = data[i:i + window_size]
        y[i] = data[i + window_size:i + window_size + horizon, target_column]
    
    return X, y


class WindowGenerator:
    """
    Generator class for creating windowed sequences from time series data.
    
    Supports both single-step and multi-step prediction configurations.
    
    Attributes:
        window_size: Number of input timesteps.
        horizon: Prediction horizon.
        target_column: Name or index of target column.
        feature_columns: List of feature column names.
    """
    
    def __init__(
        self,
        window_size: int = 60,
        horizon: int = 1,
        target_column: Union[str, int] = 'Close',
        feature_columns: Optional[List[str]] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the window generator.
        
        Args:
            window_size: Number of input timesteps.
            horizon: Prediction horizon.
            target_column: Target column name or index.
            feature_columns: List of feature column names.
            config: Configuration object.
        """
        self.window_size = window_size
        self.horizon = horizon
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.config = config if config is not None else Config.load()
        
        # Get parameters from config if not specified
        if window_size is None:
            self.window_size = self.config.base['data']['window_size']
        if horizon is None:
            self.horizon = self.config.base['data']['horizon']
    
    def _get_target_index(self, columns: List[str]) -> int:
        """Get the index of the target column."""
        if isinstance(self.target_column, int):
            return self.target_column
        if self.target_column in columns:
            return columns.index(self.target_column)
        raise ValueError(f"Target column '{self.target_column}' not found in columns")
    
    def generate(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate windowed sequences from a DataFrame.
        
        Args:
            df: Input DataFrame.
            feature_columns: Columns to use as features.
            
        Returns:
            Tuple of (X, y) arrays.
        """
        if feature_columns is None:
            feature_columns = self.feature_columns or df.columns.tolist()
        
        data = df[feature_columns].values
        target_idx = self._get_target_index(feature_columns)
        
        if self.horizon == 1:
            X, y = create_sequences(data, self.window_size, self.horizon, target_idx)
        else:
            X, y = create_multi_step_sequences(data, self.window_size, self.horizon, target_idx)
        
        logger.info(f"Generated sequences: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def generate_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate windowed sequences for train, validation, and test splits.
        
        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            test_df: Test DataFrame.
            feature_columns: Columns to use as features.
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
        """
        X_train, y_train = self.generate(train_df, feature_columns)
        X_val, y_val = self.generate(val_df, feature_columns)
        X_test, y_test = self.generate(test_df, feature_columns)
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def generate_from_array(
        self,
        data: np.ndarray,
        target_column: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sequences from a numpy array.
        
        Args:
            data: Input array.
            target_column: Index of target column.
            
        Returns:
            Tuple of (X, y) arrays.
        """
        if self.horizon == 1:
            return create_sequences(data, self.window_size, self.horizon, target_column)
        return create_multi_step_sequences(data, self.window_size, self.horizon, target_column)


def prepare_dataset(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'Close',
    window_size: int = 60,
    horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> dict:
    """
    Complete dataset preparation pipeline.
    
    Args:
        df: Input DataFrame with all data.
        feature_columns: Columns to use as features.
        target_column: Target column for prediction.
        window_size: Number of input timesteps.
        horizon: Prediction horizon.
        train_ratio: Training data ratio.
        val_ratio: Validation data ratio.
        test_ratio: Test data ratio.
        
    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test arrays.
    """
    from src.data.preprocessing import train_val_test_split, DataPreprocessor
    
    # Split data
    train_df, val_df, test_df = train_val_test_split(
        df, train_ratio, val_ratio, test_ratio
    )
    
    # Preprocess (scale) data
    preprocessor = DataPreprocessor()
    train_scaled, val_scaled, test_scaled = preprocessor.fit_transform(
        train_df, val_df, test_df, feature_columns
    )
    
    # Generate windows
    window_gen = WindowGenerator(
        window_size=window_size,
        horizon=horizon,
        target_column=target_column,
        feature_columns=feature_columns
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = window_gen.generate_splits(
        train_scaled, val_scaled, test_scaled
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': preprocessor.scaler,
        'feature_columns': feature_columns
    }


class DatasetGenerator:
    """
    High-level dataset generator for experiments.
    
    Creates datasets for different feature groups and assets.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the dataset generator.
        
        Args:
            config: Configuration object.
        """
        self.config = config if config is not None else Config.load()
    
    def create_dataset(
        self,
        asset_name: str,
        group_name: str,
        save: bool = True
    ) -> dict:
        """
        Create a dataset for a specific asset and feature group.
        
        Args:
            asset_name: Name of the asset.
            group_name: Name of the feature group.
            save: If True, save the dataset.
            
        Returns:
            Dataset dictionary.
        """
        from src.data.loader import load_asset_data, save_processed_data
        from src.data.preprocessing import DataPreprocessor, train_val_test_split, save_scaler
        
        logger.info(f"Creating dataset for {asset_name}/{group_name}")
        
        # Get feature columns for this group
        feature_columns = self.config.get_feature_names(group_name)
        
        # Load preprocessed data with indicators
        df = load_asset_data(asset_name, self.config, use_interim=True)
        
        # Select only the columns we need
        available_cols = [col for col in feature_columns if col in df.columns]
        if len(available_cols) < len(feature_columns):
            missing = set(feature_columns) - set(available_cols)
            logger.warning(f"Missing columns: {missing}")
        
        df = df[available_cols]
        
        # Get data parameters
        data_params = self.config.base['data']
        
        # Split data
        train_df, val_df, test_df = train_val_test_split(
            df,
            data_params['train_ratio'],
            data_params['val_ratio'],
            data_params['test_ratio']
        )
        
        # Scale data
        preprocessor = DataPreprocessor(self.config)
        train_scaled, val_scaled, test_scaled = preprocessor.fit_transform(
            train_df, val_df, test_df, available_cols
        )
        
        # Generate windows
        window_gen = WindowGenerator(
            window_size=data_params['window_size'],
            horizon=data_params['horizon'],
            target_column=data_params['target_column'],
            feature_columns=available_cols,
            config=self.config
        )
        
        X_train, y_train, X_val, y_val, X_test, y_test = window_gen.generate_splits(
            train_scaled, val_scaled, test_scaled
        )
        
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        if save:
            # Save processed data
            save_processed_data(
                dataset, asset_name, group_name, self.config,
                scaler=preprocessor.scaler,
                feature_columns=available_cols
            )
        
        logger.info(f"Dataset created: X_train shape {X_train.shape}")
        
        return dataset


if __name__ == "__main__":
    # Test window generator
    np.random.seed(42)
    
    # Create sample data
    n_samples = 1000
    n_features = 5
    data = np.random.randn(n_samples, n_features)
    
    # Test sequence creation
    X, y = create_sequences(data, window_size=60, horizon=1, target_column=3)
    print(f"X shape: {X.shape}")  # (940, 60, 5)
    print(f"y shape: {y.shape}")  # (940,)
    
    # Test WindowGenerator class
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='H')
    df = pd.DataFrame(data, index=dates, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    window_gen = WindowGenerator(window_size=60, horizon=1, target_column='Close')
    X, y = window_gen.generate(df)
    print(f"WindowGenerator output: X shape {X.shape}, y shape {y.shape}")
