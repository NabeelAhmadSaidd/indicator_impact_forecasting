"""
GRU Model Module

This module provides GRU (Gated Recurrent Unit) model implementation
using TensorFlow/Keras for financial time series forecasting.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    GRU, Dense, Dropout, Input, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

from src.utils.config import Config, load_yaml


logger = logging.getLogger(__name__)


def build_gru_model(
    input_shape: Tuple[int, int],
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None
) -> Model:
    """
    Build a GRU model based on configuration.
    
    Args:
        input_shape: Shape of input data (window_size, n_features).
        config: Model configuration dictionary.
        config_path: Path to GRU config file.
        
    Returns:
        Compiled Keras Model.
    """
    if config is None:
        if config_path is None:
            from src.utils.config import get_project_root
            config_path = get_project_root() / 'configs' / 'gru.yaml'
        config = load_yaml(config_path)
    
    architecture = config.get('architecture', {})
    training_config = config.get('training', {})
    regularization = config.get('regularization', {})
    
    # Create model
    model = Sequential(name='GRU_Model')
    
    # Add input layer explicitly
    model.add(Input(shape=input_shape))
    
    # Add GRU layers
    gru_layers = architecture.get('layers', [])
    for i, layer_config in enumerate(gru_layers):
        if layer_config.get('type') == 'gru':
            model.add(GRU(
                units=layer_config.get('units', 64),
                return_sequences=layer_config.get('return_sequences', False),
                dropout=layer_config.get('dropout', 0.0),
                recurrent_dropout=layer_config.get('recurrent_dropout', 0.0),
                kernel_regularizer=l1_l2(
                    l1=regularization.get('l1', 0.0),
                    l2=regularization.get('l2', 0.0)
                ),
                name=f'gru_{i+1}'
            ))
    
    # Add dense layers
    dense_layers = architecture.get('dense_layers', [])
    for i, layer_config in enumerate(dense_layers):
        model.add(Dense(
            units=layer_config.get('units', 32),
            activation=layer_config.get('activation', 'relu'),
            kernel_regularizer=l1_l2(
                l1=regularization.get('l1', 0.0),
                l2=regularization.get('l2', 0.0)
            ),
            name=f'dense_{i+1}'
        ))
        
        if layer_config.get('dropout', 0) > 0:
            model.add(Dropout(layer_config['dropout'], name=f'dropout_{i+1}'))
    
    # Add output layer
    output_config = architecture.get('output', {})
    model.add(Dense(
        units=output_config.get('units', 1),
        activation=output_config.get('activation', 'linear'),
        name='output'
    ))
    
    # Compile model
    optimizer_config = training_config.get('optimizer', {})
    optimizer = Adam(
        learning_rate=optimizer_config.get('learning_rate', 0.001),
        beta_1=optimizer_config.get('beta_1', 0.9),
        beta_2=optimizer_config.get('beta_2', 0.999),
        epsilon=optimizer_config.get('epsilon', 1e-7)
    )
    
    model.compile(
        optimizer=optimizer,
        loss=training_config.get('loss', 'mse'),
        metrics=training_config.get('metrics', ['mae'])
    )
    
    logger.info(f"Built GRU model with input shape {input_shape}")
    
    return model


class GRUForecaster:
    """
    GRU-based forecaster for financial time series.
    
    This class provides a high-level interface for building, training,
    and using GRU models for price forecasting.
    
    Attributes:
        model: Keras GRU model.
        config: Model configuration dictionary.
        history: Training history.
    """
    
    def __init__(
        self,
        input_shape: Optional[Tuple[int, int]] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the GRU forecaster.
        
        Args:
            input_shape: Shape of input data (window_size, n_features).
            config: Model configuration dictionary.
            config_path: Path to config file.
        """
        self.config = config
        self.config_path = config_path
        self.model = None
        self.history = None
        
        if input_shape is not None:
            self.build(input_shape)
    
    def build(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the GRU model.
        
        Args:
            input_shape: Shape of input data.
        """
        self.model = build_gru_model(
            input_shape,
            config=self.config,
            config_path=self.config_path
        )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: Optional[List] = None,
        verbose: int = 1,
        initial_epoch: int = 0
    ) -> Dict[str, List[float]]:
        """
        Train the GRU model.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            epochs: Number of training epochs.
            batch_size: Batch size.
            callbacks: List of Keras callbacks.
            verbose: Verbosity level.
            
        Returns:
            Training history dictionary.

        initial_epoch:
            Starting epoch offset, useful when resuming interrupted training.
        """
        if self.model is None:
            self.build(X_train.shape[1:])
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            initial_epoch=initial_epoch
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Predictions array.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() or fit() first.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: int = 0
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Input features.
            y: True targets.
            verbose: Verbosity level.
            
        Returns:
            Dictionary of metric names and values.
        """
        if self.model is None:
            raise ValueError("Model not built.")
        
        results = self.model.evaluate(X, y, verbose=verbose)
        
        metric_names = ['loss'] + self.model.metrics_names[1:] if hasattr(self.model, 'metrics_names') else ['loss']
        
        if not isinstance(results, list):
            results = [results]
        
        return dict(zip(metric_names, results))
    
    def save(self, filepath: str) -> None:
        """
        Save the model.
        
        Args:
            filepath: Path to save the model.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model.
        """
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is not None:
            self.model.summary()
    
    def get_weights(self) -> List[np.ndarray]:
        """Get model weights."""
        if self.model is None:
            raise ValueError("Model not built.")
        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set model weights."""
        if self.model is None:
            raise ValueError("Model not built.")
        self.model.set_weights(weights)


def create_gru_model(
    window_size: int,
    n_features: int,
    gru_units: List[int] = [128, 64],
    dense_units: List[int] = [32],
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    l2_reg: float = 0.001
) -> Model:
    """
    Create a GRU model with custom parameters.
    
    Args:
        window_size: Number of input timesteps.
        n_features: Number of input features.
        gru_units: List of units for each GRU layer.
        dense_units: List of units for each dense layer.
        dropout: Dropout rate.
        learning_rate: Learning rate for optimizer.
        l2_reg: L2 regularization factor.
        
    Returns:
        Compiled Keras Model.
    """
    model = Sequential(name='Custom_GRU')
    
    # GRU layers
    for i, units in enumerate(gru_units):
        return_sequences = i < len(gru_units) - 1
        
        if i == 0:
            model.add(GRU(
                units,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=dropout,
                kernel_regularizer=l1_l2(l2=l2_reg),
                input_shape=(window_size, n_features),
                name=f'gru_{i+1}'
            ))
        else:
            model.add(GRU(
                units,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=dropout,
                kernel_regularizer=l1_l2(l2=l2_reg),
                name=f'gru_{i+1}'
            ))
    
    # Dense layers
    for i, units in enumerate(dense_units):
        model.add(Dense(
            units,
            activation='relu',
            kernel_regularizer=l1_l2(l2=l2_reg),
            name=f'dense_{i+1}'
        ))
        model.add(Dropout(dropout, name=f'dropout_{i+1}'))
    
    # Output layer
    model.add(Dense(1, activation='linear', name='output'))
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


if __name__ == "__main__":
    # Test GRU model
    from src.utils.seed import set_seed
    set_seed(42)
    
    # Create sample data
    n_samples = 1000
    window_size = 60
    n_features = 5
    
    X = np.random.randn(n_samples, window_size, n_features)
    y = np.random.randn(n_samples)
    
    # Test building model from config
    model = build_gru_model((window_size, n_features))
    model.summary()
    
    # Test GRUForecaster class
    forecaster = GRUForecaster()
    forecaster.build((window_size, n_features))
    forecaster.summary()
    
    # Quick training test
    history = forecaster.fit(
        X[:800], y[:800],
        X_val=X[800:900], y_val=y[800:900],
        epochs=2,
        batch_size=32,
        verbose=1
    )
    
    # Test prediction
    predictions = forecaster.predict(X[900:])
    print(f"Predictions shape: {predictions.shape}")
    
    # Test evaluation
    metrics = forecaster.evaluate(X[900:], y[900:])
    print(f"Evaluation metrics: {metrics}")
