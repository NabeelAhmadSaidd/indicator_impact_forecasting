"""
Models Package Initialization

This package provides deep learning models (LSTM and GRU) for
financial time series forecasting.
"""

from src.models.lstm import (
    build_lstm_model,
    create_lstm_model,
    LSTMForecaster
)

from src.models.gru import (
    build_gru_model,
    create_gru_model,
    GRUForecaster
)


def get_model(model_type: str, input_shape=None, config=None):
    """
    Factory function to get a model by type.
    
    Args:
        model_type: Type of model ('lstm' or 'gru').
        input_shape: Shape of input data.
        config: Model configuration dictionary.
        
    Returns:
        Model forecaster instance.
    """
    if model_type.lower() == 'lstm':
        return LSTMForecaster(input_shape=input_shape, config=config)
    elif model_type.lower() == 'gru':
        return GRUForecaster(input_shape=input_shape, config=config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm' or 'gru'.")


def build_model(model_type: str, input_shape, config=None, config_path=None):
    """
    Factory function to build a model by type.
    
    Args:
        model_type: Type of model ('lstm' or 'gru').
        input_shape: Shape of input data.
        config: Model configuration dictionary.
        config_path: Path to config file.
        
    Returns:
        Compiled Keras model.
    """
    if model_type.lower() == 'lstm':
        return build_lstm_model(input_shape, config=config, config_path=config_path)
    elif model_type.lower() == 'gru':
        return build_gru_model(input_shape, config=config, config_path=config_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm' or 'gru'.")


__all__ = [
    # LSTM
    'build_lstm_model',
    'create_lstm_model',
    'LSTMForecaster',
    
    # GRU
    'build_gru_model',
    'create_gru_model',
    'GRUForecaster',
    
    # Factory functions
    'get_model',
    'build_model'
]
