"""
Data Preprocessing Module

This module provides utilities for preprocessing financial data,
including cleaning, normalization, and feature scaling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import logging

from src.utils.config import Config, get_project_root
from src.utils.logger import get_preprocessing_logger


logger = get_preprocessing_logger()


def clean_data(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Clean financial data by handling missing values and outliers.
    
    Args:
        df: Input DataFrame.
        dropna: If True, drop rows with missing values.
        
    Returns:
        Cleaned DataFrame.
    """
    logger.info(f"Cleaning data with {len(df)} rows")
    
    original_len = len(df)
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Handle missing values
    if dropna:
        df = df.dropna()
    else:
        # Forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove rows with zero or negative prices (invalid data)
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            df = df[df[col] > 0]
    
    # Remove rows where High < Low (invalid data)
    if 'High' in df.columns and 'Low' in df.columns:
        df = df[df['High'] >= df['Low']]
    
    # Remove rows where Close is outside High-Low range
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        df = df[(df['Close'] >= df['Low']) & (df['Close'] <= df['High'])]
        df = df[(df['Open'] >= df['Low']) & (df['Open'] <= df['High'])]
    
    rows_removed = original_len - len(df)
    if rows_removed > 0:
        logger.info(f"Removed {rows_removed} invalid rows ({rows_removed/original_len*100:.2f}%)")
    
    return df


def get_scaler(method: str = 'minmax', **kwargs) -> Union[MinMaxScaler, StandardScaler, RobustScaler]:
    """
    Get a scaler object based on the specified method.
    
    Args:
        method: Scaling method ('minmax', 'standard', 'robust').
        **kwargs: Additional arguments for the scaler.
        
    Returns:
        Scaler object.
    """
    if method == 'minmax':
        feature_range = kwargs.get('feature_range', (0, 1))
        return MinMaxScaler(feature_range=feature_range)
    elif method == 'standard':
        return StandardScaler()
    elif method == 'robust':
        return RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def scale_data(
    df: pd.DataFrame,
    scaler: Optional[Union[MinMaxScaler, StandardScaler, RobustScaler]] = None,
    method: str = 'minmax',
    fit: bool = True,
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Union[MinMaxScaler, StandardScaler, RobustScaler]]:
    """
    Scale DataFrame columns using the specified scaler.
    
    Args:
        df: Input DataFrame.
        scaler: Pre-fitted scaler (optional).
        method: Scaling method if scaler is not provided.
        fit: If True, fit the scaler. If False, only transform.
        columns: Columns to scale. If None, scale all columns.
        
    Returns:
        Tuple of (scaled DataFrame, scaler).
    """
    if columns is None:
        columns = df.columns.tolist()
    
    if scaler is None:
        scaler = get_scaler(method)
    
    df_scaled = df.copy()
    
    if fit:
        df_scaled[columns] = scaler.fit_transform(df[columns])
    else:
        df_scaled[columns] = scaler.transform(df[columns])
    
    return df_scaled, scaler


def inverse_scale(
    data: np.ndarray,
    scaler: Union[MinMaxScaler, StandardScaler, RobustScaler],
    column_index: int = 0,
    n_features: int = 1
) -> np.ndarray:
    """
    Inverse transform scaled data.
    
    Args:
        data: Scaled data to inverse transform.
        scaler: Fitted scaler.
        column_index: Index of the column to inverse transform.
        n_features: Total number of features in the scaler.
        
    Returns:
        Inverse transformed data.
    """
    # Create dummy array with correct shape
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    dummy = np.zeros((data.shape[0], n_features))
    dummy[:, column_index] = data.flatten()
    
    # Inverse transform
    inversed = scaler.inverse_transform(dummy)
    
    return inversed[:, column_index]


def save_scaler(
    scaler: Union[MinMaxScaler, StandardScaler, RobustScaler],
    filepath: Union[str, Path]
) -> None:
    """
    Save a scaler to a file.
    
    Args:
        scaler: Scaler to save.
        filepath: Path to save the scaler.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"Saved scaler to {filepath}")


def load_scaler(filepath: Union[str, Path]) -> Union[MinMaxScaler, StandardScaler, RobustScaler]:
    """
    Load a scaler from a file.
    
    Args:
        filepath: Path to the scaler file.
        
    Returns:
        Loaded scaler.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    
    logger.info(f"Loaded scaler from {filepath}")
    return scaler


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets (time-series aware).
    
    Args:
        df: Input DataFrame (should be sorted by time).
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


class DataPreprocessor:
    """
    Data preprocessor class for financial time series.
    
    Handles data cleaning, feature engineering, scaling, and splitting.
    
    Attributes:
        config: Configuration object.
        scaler: Fitted scaler for normalization.
        feature_columns: List of feature columns.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration object.
        """
        self.config = config if config is not None else Config.load()
        self.scaler = None
        self.feature_columns = None
    
    def preprocess(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        fit_scaler: bool = True
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        
        Args:
            df: Input DataFrame.
            feature_columns: Columns to use as features.
            fit_scaler: If True, fit a new scaler.
            
        Returns:
            Preprocessed DataFrame.
        """
        # Clean data
        df = clean_data(df)
        
        # Select feature columns
        if feature_columns is None:
            feature_columns = df.columns.tolist()
        self.feature_columns = feature_columns
        
        # Scale data
        scaling_config = self.config.base.get('scaling', {})
        method = scaling_config.get('method', 'minmax')
        
        df_scaled, self.scaler = scale_data(
            df[feature_columns],
            scaler=None if fit_scaler else self.scaler,
            method=method,
            fit=fit_scaler
        )
        
        return df_scaled
    
    def fit_transform(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit scaler on training data and transform all splits.
        
        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            test_df: Test DataFrame.
            feature_columns: Columns to use as features.
            
        Returns:
            Tuple of scaled (train, val, test) DataFrames.
        """
        if feature_columns is None:
            feature_columns = train_df.columns.tolist()
        self.feature_columns = feature_columns
        
        # Clean data
        train_df = clean_data(train_df)
        val_df = clean_data(val_df)
        test_df = clean_data(test_df)
        
        # Fit scaler on training data
        scaling_config = self.config.base.get('scaling', {})
        method = scaling_config.get('method', 'minmax')
        
        train_scaled, self.scaler = scale_data(
            train_df[feature_columns],
            method=method,
            fit=True
        )
        
        # Transform validation and test data
        val_scaled, _ = scale_data(
            val_df[feature_columns],
            scaler=self.scaler,
            fit=False
        )
        
        test_scaled, _ = scale_data(
            test_df[feature_columns],
            scaler=self.scaler,
            fit=False
        )
        
        return train_scaled, val_scaled, test_scaled
    
    def inverse_transform(
        self,
        data: np.ndarray,
        column: str = 'Close'
    ) -> np.ndarray:
        """
        Inverse transform scaled data.
        
        Args:
            data: Scaled data.
            column: Column name for inverse transform.
            
        Returns:
            Original scale data.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call preprocess or fit_transform first.")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not set.")
        
        column_idx = self.feature_columns.index(column)
        return inverse_scale(data, self.scaler, column_idx, len(self.feature_columns))
    
    def save_scaler(self, asset_name: str, group_name: str = 'group0_ohlcv') -> None:
        """
        Save the fitted scaler for an asset and group.
        
        Args:
            asset_name: Name of the asset.
            group_name: Name of the feature group.
        """
        if self.scaler is None:
            raise ValueError("No scaler to save. Fit the preprocessor first.")
        
        # Save scaler in processed data directory: data/processed/category/asset/group/
        scaler_path = self.config.get_processed_path(asset_name, group_name) / "scaler.pkl"
        save_scaler(self.scaler, scaler_path)
    
    def load_scaler(self, asset_name: str, group_name: str = 'group0_ohlcv') -> None:
        """
        Load a saved scaler for an asset and group.
        
        Args:
            asset_name: Name of the asset.
            group_name: Name of the feature group.
        """
        # Load scaler from processed data directory: data/processed/category/asset/group/
        scaler_path = self.config.get_processed_path(asset_name, group_name) / "scaler.pkl"
        self.scaler = load_scaler(scaler_path)


def preprocess_asset(
    asset_name: str,
    config: Optional[Config] = None,
    save_interim: bool = True
) -> pd.DataFrame:
    """
    Preprocess a single asset's data.
    
    Args:
        asset_name: Name of the asset.
        config: Configuration object.
        save_interim: If True, save preprocessed data to interim folder.
        
    Returns:
        Preprocessed DataFrame.
    """
    from src.data.loader import load_asset_data, save_parquet
    from src.features.indicators import add_all_indicators
    
    if config is None:
        config = Config.load()
    
    logger.info(f"Preprocessing {asset_name}")
    
    # Load raw data
    df = load_asset_data(asset_name, config, use_interim=False)
    
    # Clean data
    df = clean_data(df)
    
    # Add technical indicators
    df = add_all_indicators(df)
    
    # Drop rows with NaN from indicator calculations
    df = df.dropna()
    
    logger.info(f"Preprocessed {asset_name}: {len(df)} rows, {len(df.columns)} columns")
    
    if save_interim:
        # Save to data/interim/category/asset/
        interim_path = config.get_interim_path(asset_name)
        interim_path.mkdir(parents=True, exist_ok=True)
        save_parquet(df, interim_path / f"{asset_name}.parquet")
    
    return df


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor()
    
    # Create sample data for testing
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'Open': 100 + np.random.randn(1000).cumsum(),
        'High': 101 + np.random.randn(1000).cumsum(),
        'Low': 99 + np.random.randn(1000).cumsum(),
        'Close': 100 + np.random.randn(1000).cumsum(),
        'Volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Fix High/Low
    sample_df['High'] = sample_df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    sample_df['Low'] = sample_df[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    # Test preprocessing
    processed = preprocessor.preprocess(sample_df)
    print(f"Processed shape: {processed.shape}")
    print(f"Value range: [{processed.min().min():.4f}, {processed.max().max():.4f}]")
