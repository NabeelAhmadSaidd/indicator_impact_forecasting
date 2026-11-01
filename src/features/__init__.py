"""
Features Package Initialization

This package provides utilities for calculating technical indicators
for financial time series analysis.
"""

from src.features.indicators import (
    # Momentum indicators
    calculate_rsi,
    calculate_stochastic,
    
    # Trend indicators
    calculate_sma,
    calculate_ema,
    calculate_adx,
    
    # Volatility indicators
    calculate_atr,
    calculate_bollinger_bands,
    calculate_bollinger_percentb,
    
    # Composite indicators
    calculate_macd,
    
    # DataFrame functions
    add_momentum_indicators,
    add_trend_indicators,
    add_volatility_indicators,
    add_macd_indicators,
    add_all_indicators,
    add_indicators_for_group,
    get_indicator_feature_names
)

__all__ = [
    # Momentum
    'calculate_rsi',
    'calculate_stochastic',
    
    # Trend
    'calculate_sma',
    'calculate_ema',
    'calculate_adx',
    
    # Volatility
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_bollinger_percentb',
    
    # Composite
    'calculate_macd',
    
    # DataFrame functions
    'add_momentum_indicators',
    'add_trend_indicators',
    'add_volatility_indicators',
    'add_macd_indicators',
    'add_all_indicators',
    'add_indicators_for_group',
    'get_indicator_feature_names'
]
