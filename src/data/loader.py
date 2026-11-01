"""
Data Loader Module

This module provides utilities for loading raw financial data from various
file formats and sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

from src.utils.config import Config, get_project_root
from src.utils.logger import get_preprocessing_logger


logger = get_preprocessing_logger()


def load_csv(
    filepath: Union[str, Path],
    datetime_column: str = 'timestamp',
    datetime_format: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load a CSV file and parse datetime column.
    
    Args:
        filepath: Path to the CSV file.
        datetime_column: Name of the datetime column.
        datetime_format: Format string for parsing datetime.
        **kwargs: Additional arguments passed to pd.read_csv.
        
    Returns:
        DataFrame with parsed datetime index.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    logger.info(f"Loading data from {filepath}")
    
    df = pd.read_csv(filepath, **kwargs)
    
    # Handle various datetime column names
    datetime_cols = ['timestamp', 'Timestamp', 'time', 'Time', 'date', 'Date', 'datetime', 'Datetime']
    actual_datetime_col = None
    
    for col in datetime_cols:
        if col in df.columns:
            actual_datetime_col = col
            break
    
    if actual_datetime_col is None:
        # Try to use the first column as datetime
        actual_datetime_col = df.columns[0]
        logger.warning(f"No datetime column found, using first column: {actual_datetime_col}")
    
    # Parse datetime
    try:
        if datetime_format:
            df[actual_datetime_col] = pd.to_datetime(df[actual_datetime_col], format=datetime_format)
        else:
            df[actual_datetime_col] = pd.to_datetime(df[actual_datetime_col])
    except Exception as e:
        logger.warning(f"Failed to parse datetime: {e}. Keeping original format.")
    
    # Set datetime as index
    df.set_index(actual_datetime_col, inplace=True)
    df.sort_index(inplace=True)
    
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    return df


def load_parquet(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a Parquet file.
    
    Args:
        filepath: Path to the Parquet file.
        
    Returns:
        DataFrame loaded from the Parquet file.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    logger.info(f"Loading data from {filepath}")
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded {len(df)} rows")
    
    return df


def save_parquet(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    """
    Save a DataFrame to a Parquet file.
    
    Args:
        df: DataFrame to save.
        filepath: Path to save the Parquet file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath)
    logger.info(f"Saved {len(df)} rows to {filepath}")


def load_asset_data(
    asset_name: str,
    config: Optional[Config] = None,
    use_interim: bool = False
) -> pd.DataFrame:
    """
    Load data for a specific asset.
    
    Args:
        asset_name: Name of the asset (e.g., 'BTCUSD').
        config: Configuration object. If None, loads default config.
        use_interim: If True, load from interim (preprocessed) data.
        
    Returns:
        DataFrame with asset data.
    """
    if config is None:
        config = Config.load()
    
    asset_config = config.get_asset_config(asset_name)
    
    if use_interim:
        # Load from interim (processed with indicators)
        # data/interim/category/asset/
        filepath = config.get_interim_path(asset_name) / f"{asset_name}.parquet"
        return load_parquet(filepath)
    else:
        # Load from raw data
        # data/raw/category/asset/
        filename = asset_config['filename']
        filepath = config.get_raw_path(asset_name) / filename
        return load_csv(filepath)


def load_all_assets(
    config: Optional[Config] = None,
    use_interim: bool = False,
    assets: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load data for all configured assets.
    
    Args:
        config: Configuration object.
        use_interim: If True, load from interim data.
        assets: List of specific assets to load. If None, loads all.
        
    Returns:
        Dictionary mapping asset names to DataFrames.
    """
    if config is None:
        config = Config.load()
    
    if assets is None:
        assets = config.get_all_assets()
    
    data = {}
    for asset_name in assets:
        try:
            data[asset_name] = load_asset_data(asset_name, config, use_interim)
        except FileNotFoundError as e:
            logger.warning(f"Could not load {asset_name}: {e}")
    
    return data


def load_processed_data(
    asset_name: str,
    group_name: str = 'group0_ohlcv',
    config: Optional[Config] = None
) -> Dict[str, np.ndarray]:
    """
    Load processed (windowed) data for an asset.
    
    Args:
        asset_name: Name of the asset.
        group_name: Name of the feature group.
        config: Configuration object.
        
    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test arrays.
    """
    if config is None:
        config = Config.load()
    
    # Use new hierarchical path: data/processed/category/asset/group/
    processed_dir = config.get_processed_path(asset_name, group_name)
    
    data = {}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        X_path = processed_dir / f"X_{split}.npy"
        y_path = processed_dir / f"y_{split}.npy"
        
        if X_path.exists() and y_path.exists():
            data[f'X_{split}'] = np.load(X_path)
            data[f'y_{split}'] = np.load(y_path)
            logger.info(f"Loaded {split} data: X shape {data[f'X_{split}'].shape}")
        else:
            logger.warning(f"Processed data not found for {asset_name}/{group_name}/{split}")
    
    return data


def save_processed_data(
    data: Dict[str, np.ndarray],
    asset_name: str,
    group_name: str = 'group0_ohlcv',
    config: Optional[Config] = None,
    scaler: Optional[Any] = None,
    feature_columns: Optional[List[str]] = None
) -> None:
    """
    Save processed (windowed) data for an asset.
    
    Args:
        data: Dictionary with X_train, y_train, etc. arrays.
        asset_name: Name of the asset.
        group_name: Name of the feature group.
        config: Configuration object.
        scaler: Scaler object to save.
        feature_columns: List of feature column names.
    """
    if config is None:
        config = Config.load()
    
    # Use new hierarchical path: data/processed/category/asset/group/
    processed_dir = config.get_processed_path(asset_name, group_name)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all data splits
    keys_to_save = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
    for key in keys_to_save:
        if key in data:
            filepath = processed_dir / f"{key}.npy"
            np.save(filepath, data[key])
            logger.info(f"Saved {key} with shape {data[key].shape} to {filepath}")
    
    # Save scaler if provided
    if scaler is not None:
        from src.data.preprocessing import save_scaler
        scaler_path = processed_dir / "scaler.pkl"
        save_scaler(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
    
    # Save features if provided
    if feature_columns is not None:
        features_path = processed_dir / "features.txt"
        with open(features_path, 'w') as f:
            f.write('\n'.join(feature_columns))
        logger.info(f"Saved features to {features_path}")


class DataLoader:
    """
    Data loader class for managing financial data loading operations.
    
    Attributes:
        config: Configuration object.
        cache: Dictionary for caching loaded data.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config if config is not None else Config.load()
        self.cache: Dict[str, pd.DataFrame] = {}
    
    def load(
        self,
        asset_name: str,
        use_interim: bool = False,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load data for an asset.
        
        Args:
            asset_name: Name of the asset.
            use_interim: If True, load preprocessed data.
            use_cache: If True, cache the loaded data.
            
        Returns:
            DataFrame with asset data.
        """
        cache_key = f"{asset_name}_{'interim' if use_interim else 'raw'}"
        
        if use_cache and cache_key in self.cache:
            logger.debug(f"Using cached data for {asset_name}")
            return self.cache[cache_key]
        
        df = load_asset_data(asset_name, self.config, use_interim)
        
        if use_cache:
            self.cache[cache_key] = df
        
        return df
    
    def load_processed(
        self,
        asset_name: str,
        group_name: str = 'group0_ohlcv'
    ) -> Dict[str, np.ndarray]:
        """
        Load processed data for an asset.
        
        Args:
            asset_name: Name of the asset.
            group_name: Name of the feature group.
            
        Returns:
            Dictionary with processed arrays.
        """
        return load_processed_data(asset_name, group_name, self.config)
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.cache.clear()
        logger.info("Data cache cleared")
    
    def get_asset_info(self, asset_name: str) -> Dict[str, Any]:
        """
        Get information about an asset.
        
        Args:
            asset_name: Name of the asset.
            
        Returns:
            Dictionary with asset information.
        """
        return self.config.get_asset_config(asset_name)
    
    def list_available_assets(self) -> List[str]:
        """
        List all available assets.
        
        Returns:
            List of asset names.
        """
        return self.config.get_all_assets()


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader()
    print(f"Available assets: {loader.list_available_assets()}")
    
    # Try to load an asset
    try:
        df = loader.load('BTCUSD')
        print(f"BTCUSD data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
