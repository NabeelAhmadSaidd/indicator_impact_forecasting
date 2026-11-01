"""
Configuration Utility Module

This module provides utilities for loading and managing YAML configuration files
for the indicator impact forecasting project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import copy


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Path to the project root directory.
    """
    current = Path(__file__).resolve()
    # Navigate up from src/utils/config.py to project root
    return current.parent.parent.parent


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        filepath: Path to the YAML file.
        
    Returns:
        Dictionary containing the configuration.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save a configuration dictionary to a YAML file.
    
    Args:
        config: Configuration dictionary to save.
        filepath: Path to save the YAML file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.
        
    Returns:
        Merged configuration dictionary.
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


@dataclass
class Config:
    """
    Configuration manager for the indicator impact forecasting project.
    
    Attributes:
        base: Base configuration dictionary.
        assets: Assets configuration dictionary.
        feature_groups: Feature groups configuration dictionary.
        experiments: Experiments configuration dictionary.
        model_config: Current model configuration (LSTM or GRU).
    """
    base: Dict[str, Any]
    assets: Dict[str, Any]
    feature_groups: Dict[str, Any]
    experiments: Dict[str, Any]
    model_config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def load(cls, config_dir: Optional[Union[str, Path]] = None, 
             model_type: Optional[str] = None) -> 'Config':
        """
        Load all configuration files.
        
        Args:
            config_dir: Path to the configs directory. If None, uses default location.
            model_type: Type of model ('lstm' or 'gru'). If provided, loads model config.
            
        Returns:
            Config instance with all configurations loaded.
        """
        if config_dir is None:
            config_dir = get_project_root() / 'configs'
        config_dir = Path(config_dir)
        
        base = load_yaml(config_dir / 'base.yaml')
        assets = load_yaml(config_dir / 'assets.yaml')
        feature_groups = load_yaml(config_dir / 'feature_groups.yaml')
        experiments = load_yaml(config_dir / 'experiments.yaml')
        
        model_config = None
        if model_type:
            model_config = load_yaml(config_dir / f'{model_type}.yaml')
        
        return cls(
            base=base,
            assets=assets,
            feature_groups=feature_groups,
            experiments=experiments,
            model_config=model_config
        )
    
    def get_asset_config(self, asset_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific asset.
        
        Args:
            asset_name: Name of the asset (e.g., 'bitcoin').
            
        Returns:
            Asset configuration dictionary.
            
        Raises:
            ValueError: If asset is not found in configuration.
        """
        for category in self.assets['assets']:
            if asset_name in self.assets['assets'][category]:
                config = self.assets['assets'][category][asset_name].copy()
                config['category'] = category
                return config
        raise ValueError(f"Asset '{asset_name}' not found in configuration")
    
    def get_feature_group(self, group_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific feature group.
        
        Args:
            group_name: Name of the feature group (e.g., 'group1_momentum').
            
        Returns:
            Feature group configuration dictionary.
            
        Raises:
            ValueError: If feature group is not found.
        """
        if group_name in self.feature_groups:
            return self.feature_groups[group_name]
        raise ValueError(f"Feature group '{group_name}' not found in configuration")
    
    def get_feature_names(self, group_name: str) -> list:
        """
        Get list of feature names for a specific group.
        
        Args:
            group_name: Name of the feature group.
            
        Returns:
            List of feature names.
        """
        if group_name == 'group0_ohlcv':
            return self.feature_groups['group0_ohlcv']['features']
        
        group_config = self.experiments['groups'].get(group_name, {})
        base_features = self.feature_groups['group0_ohlcv']['features'].copy()
        
        if 'add_indicators' in group_config:
            base_features.extend(group_config['add_indicators'])
        
        return base_features
    
    def get_all_assets(self) -> list:
        """
        Get list of all asset names.
        
        Returns:
            List of all asset names.
        """
        assets = []
        for category in self.assets['assets']:
            assets.extend(self.assets['assets'][category].keys())
        return assets
    
    def get_path(self, path_key: str) -> Path:
        """
        Get a path from the configuration.
        
        Args:
            path_key: Key of the path in base configuration.
            
        Returns:
            Absolute path.
        """
        relative_path = self.base['paths'].get(path_key, '')
        return get_project_root() / relative_path
    
    def get_hierarchical_path(
        self,
        path_type: str,
        category: Optional[str] = None,
        asset: Optional[str] = None,
        group: Optional[str] = None,
        model: Optional[str] = None
    ) -> Path:
        """
        Get a hierarchical path based on the new structure.
        
        Args:
            path_type: Type of path ('raw', 'processed', 'experiments', 'results', 'models')
            category: Asset category (crypto/equity/forex)
            asset: Asset name (e.g., BTCUSD)
            group: Feature group name (e.g., group5_all)
            model: Model type (lstm/gru)
            
        Returns:
            Absolute path following the hierarchy.
        """
        if 'path_patterns' in self.base:
            pattern = self.base['path_patterns'].get(path_type, '')
        else:
            # Fallback to old structure
            return self.get_path(f"{path_type}_data" if path_type in ['raw', 'processed'] else path_type)
        
        # Build path from pattern
        path_str = pattern.format(
            category=category or '',
            asset=asset or '',
            group=group or '',
            model=model or ''
        )
        return get_project_root() / path_str.rstrip('/')
    
    def get_raw_path(self, asset_name: str) -> Path:
        """Get path for raw data."""
        asset_config = self.get_asset_config(asset_name)
        return self.get_hierarchical_path('raw', category=asset_config['category'], asset=asset_name)
    
    def get_interim_path(self, asset_name: str) -> Path:
        """Get path for interim (preprocessed) data."""
        asset_config = self.get_asset_config(asset_name)
        category = asset_config['category']
        return get_project_root() / 'data' / 'interim' / category / asset_name
    
    def get_processed_path(self, asset_name: str, group_name: str) -> Path:
        """Get path for processed (windowed) datasets."""
        asset_config = self.get_asset_config(asset_name)
        return self.get_hierarchical_path('processed', 
                                         category=asset_config['category'],
                                         asset=asset_name,
                                         group=group_name)
    
    def get_experiment_path(self, asset_name: str, group_name: str, model_type: str) -> Path:
        """Get path for experiment outputs."""
        asset_config = self.get_asset_config(asset_name)
        return self.get_hierarchical_path('experiments',
                                         category=asset_config['category'],
                                         asset=asset_name,
                                         group=group_name,
                                         model=model_type)
    
    def get_results_path(self, asset_name: str) -> Path:
        """Get path for results (tables/figures)."""
        asset_config = self.get_asset_config(asset_name)
        return self.get_hierarchical_path('results',
                                         category=asset_config['category'],
                                         asset=asset_name)
    
    def get_model_path(self, asset_name: str) -> Path:
        """Get path for final trained models."""
        asset_config = self.get_asset_config(asset_name)
        return self.get_hierarchical_path('models',
                                         category=asset_config['category'],
                                         asset=asset_name)
    
    @property
    def model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get model configurations for LSTM and GRU.
        
        Returns:
            Dictionary with 'lstm' and 'gru' configurations.
        """
        config_dir = get_project_root() / 'configs'
        return {
            'lstm': load_yaml(config_dir / 'lstm.yaml'),
            'gru': load_yaml(config_dir / 'gru.yaml')
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary containing all configurations.
        """
        return {
            'base': self.base,
            'assets': self.assets,
            'feature_groups': self.feature_groups,
            'experiments': self.experiments,
            'model_config': self.model_config
        }


def get_experiment_config(group_name: str, model_type: str, 
                          asset_name: str, run_id: int) -> Dict[str, Any]:
    """
    Get complete configuration for a specific experiment run.
    
    Args:
        group_name: Name of the feature group.
        model_type: Type of model ('lstm' or 'gru').
        asset_name: Name of the asset.
        run_id: Run identifier.
        
    Returns:
        Complete experiment configuration dictionary.
    """
    config = Config.load(model_type=model_type)
    
    return {
        'group': group_name,
        'model': model_type,
        'asset': asset_name,
        'run_id': run_id,
        'seed': config.base['seeds'][run_id - 1] if run_id <= len(config.base['seeds']) else config.base['seed'],
        'features': config.get_feature_names(group_name),
        'model_params': config.model_config,
        'training_params': config.base['training'],
        'data_params': config.base['data']
    }


if __name__ == "__main__":
    # Test configuration loading
    config = Config.load(model_type='lstm')
    print("Configuration loaded successfully!")
    print(f"Available assets: {config.get_all_assets()}")
    print(f"Features for group5_all: {config.get_feature_names('group5_all')}")
