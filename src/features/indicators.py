"""
Technical Indicators Module

This module provides functions for calculating technical indicators
using TA-Lib for financial time series analysis.

All indicators are organized by category:
- Momentum: RSI, Stochastic %K/%D
- Trend: SMA, EMA, ADX
- Volatility: ATR, Bollinger Bands
- Composite: MACD family
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import logging

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Some features will be limited.")


logger = logging.getLogger(__name__)


# =============================================================================
# Momentum Indicators
# =============================================================================

def calculate_rsi(
    close: np.ndarray,
    timeperiod: int = 14
) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.
    
    Args:
        close: Array of closing prices.
        timeperiod: RSI period (default: 14).
        
    Returns:
        Array of RSI values.
    """
    if TALIB_AVAILABLE:
        return talib.RSI(close, timeperiod=timeperiod)
    else:
        # Manual RSI calculation
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[timeperiod] = np.mean(gain[:timeperiod])
        avg_loss[timeperiod] = np.mean(loss[:timeperiod])
        
        for i in range(timeperiod + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (timeperiod - 1) + gain[i-1]) / timeperiod
            avg_loss[i] = (avg_loss[i-1] * (timeperiod - 1) + loss[i-1]) / timeperiod
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:timeperiod] = np.nan
        
        return rsi


def calculate_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowk_matype: int = 0,
    slowd_period: int = 3,
    slowd_matype: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Stochastic oscillator compares a closing price to its price range
    over a given time period.
    
    Args:
        high: Array of high prices.
        low: Array of low prices.
        close: Array of closing prices.
        fastk_period: Fast %K period.
        slowk_period: Slow %K period.
        slowk_matype: Slow %K MA type.
        slowd_period: Slow %D period.
        slowd_matype: Slow %D MA type.
        
    Returns:
        Tuple of (slowk, slowd) arrays.
    """
    if TALIB_AVAILABLE:
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=slowk_matype,
            slowd_period=slowd_period,
            slowd_matype=slowd_matype
        )
        return slowk, slowd
    else:
        # Manual Stochastic calculation
        n = len(close)
        slowk = np.full(n, np.nan)
        slowd = np.full(n, np.nan)
        
        for i in range(fastk_period - 1, n):
            highest_high = np.max(high[i - fastk_period + 1:i + 1])
            lowest_low = np.min(low[i - fastk_period + 1:i + 1])
            
            if highest_high != lowest_low:
                slowk[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
            else:
                slowk[i] = 50
        
        # Apply smoothing for slowk and slowd
        for i in range(fastk_period + slowk_period - 2, n):
            slowk[i] = np.mean(slowk[i - slowk_period + 1:i + 1])
        
        for i in range(fastk_period + slowk_period + slowd_period - 3, n):
            slowd[i] = np.mean(slowk[i - slowd_period + 1:i + 1])
        
        return slowk, slowd


# =============================================================================
# Trend Indicators
# =============================================================================

def calculate_sma(
    close: np.ndarray,
    timeperiod: int = 20
) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        close: Array of closing prices.
        timeperiod: SMA period (default: 20).
        
    Returns:
        Array of SMA values.
    """
    if TALIB_AVAILABLE:
        return talib.SMA(close, timeperiod=timeperiod)
    else:
        sma = np.full_like(close, np.nan)
        for i in range(timeperiod - 1, len(close)):
            sma[i] = np.mean(close[i - timeperiod + 1:i + 1])
        return sma


def calculate_ema(
    close: np.ndarray,
    timeperiod: int = 20
) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        close: Array of closing prices.
        timeperiod: EMA period (default: 20).
        
    Returns:
        Array of EMA values.
    """
    if TALIB_AVAILABLE:
        return talib.EMA(close, timeperiod=timeperiod)
    else:
        ema = np.full_like(close, np.nan)
        multiplier = 2 / (timeperiod + 1)
        
        # Initialize with SMA
        ema[timeperiod - 1] = np.mean(close[:timeperiod])
        
        for i in range(timeperiod, len(close)):
            ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
        
        return ema


def calculate_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timeperiod: int = 14
) -> np.ndarray:
    """
    Calculate Average Directional Index (ADX).
    
    ADX measures trend strength without regard to trend direction.
    
    Args:
        high: Array of high prices.
        low: Array of low prices.
        close: Array of closing prices.
        timeperiod: ADX period (default: 14).
        
    Returns:
        Array of ADX values.
    """
    if TALIB_AVAILABLE:
        return talib.ADX(high, low, close, timeperiod=timeperiod)
    else:
        # Simplified ADX calculation
        n = len(close)
        adx = np.full(n, np.nan)
        
        # Calculate True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        tr = np.insert(tr, 0, high[0] - low[0])
        
        # Calculate +DM and -DM
        plus_dm = np.where(
            (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
            np.maximum(high[1:] - high[:-1], 0),
            0
        )
        minus_dm = np.where(
            (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
            np.maximum(low[:-1] - low[1:], 0),
            0
        )
        plus_dm = np.insert(plus_dm, 0, 0)
        minus_dm = np.insert(minus_dm, 0, 0)
        
        # Smooth with EMA
        atr = calculate_ema(tr, timeperiod)
        plus_di = 100 * calculate_ema(plus_dm, timeperiod) / np.where(atr != 0, atr, 1)
        minus_di = 100 * calculate_ema(minus_dm, timeperiod) / np.where(atr != 0, atr, 1)
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / np.where(
            (plus_di + minus_di) != 0,
            plus_di + minus_di,
            1
        )
        
        # Smooth DX to get ADX
        adx = calculate_ema(dx, timeperiod)
        
        return adx


# =============================================================================
# Volatility Indicators
# =============================================================================

def calculate_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timeperiod: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range (ATR).
    
    ATR measures market volatility.
    
    Args:
        high: Array of high prices.
        low: Array of low prices.
        close: Array of closing prices.
        timeperiod: ATR period (default: 14).
        
    Returns:
        Array of ATR values.
    """
    if TALIB_AVAILABLE:
        return talib.ATR(high, low, close, timeperiod=timeperiod)
    else:
        # Calculate True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        tr = np.insert(tr, 0, high[0] - low[0])
        
        # Calculate ATR as EMA of TR
        atr = calculate_ema(tr, timeperiod)
        
        return atr


def calculate_bollinger_bands(
    close: np.ndarray,
    timeperiod: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of a middle band (SMA) and upper/lower bands
    at standard deviation intervals.
    
    Args:
        close: Array of closing prices.
        timeperiod: Period for middle band (default: 20).
        nbdevup: Standard deviations for upper band (default: 2).
        nbdevdn: Standard deviations for lower band (default: 2).
        matype: Moving average type (default: 0 = SMA).
        
    Returns:
        Tuple of (upper, middle, lower) band arrays.
    """
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=timeperiod,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=matype
        )
        return upper, middle, lower
    else:
        middle = calculate_sma(close, timeperiod)
        
        # Calculate standard deviation
        std = np.full_like(close, np.nan)
        for i in range(timeperiod - 1, len(close)):
            std[i] = np.std(close[i - timeperiod + 1:i + 1])
        
        upper = middle + nbdevup * std
        lower = middle - nbdevdn * std
        
        return upper, middle, lower


def calculate_bollinger_percentb(
    close: np.ndarray,
    timeperiod: int = 20,
    nbdev: float = 2.0
) -> np.ndarray:
    """
    Calculate Bollinger %B.
    
    %B shows where price is relative to the bands.
    %B = (Price - Lower Band) / (Upper Band - Lower Band)
    
    Args:
        close: Array of closing prices.
        timeperiod: Period for Bollinger Bands.
        nbdev: Number of standard deviations.
        
    Returns:
        Array of %B values.
    """
    upper, middle, lower = calculate_bollinger_bands(
        close, timeperiod, nbdev, nbdev, 0
    )
    
    band_width = upper - lower
    percentb = np.where(
        band_width != 0,
        (close - lower) / band_width,
        0.5
    )
    
    return percentb


# =============================================================================
# Composite Indicators
# =============================================================================

def calculate_macd(
    close: np.ndarray,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        close: Array of closing prices.
        fastperiod: Fast EMA period (default: 12).
        slowperiod: Slow EMA period (default: 26).
        signalperiod: Signal line period (default: 9).
        
    Returns:
        Tuple of (macd, signal, histogram) arrays.
    """
    if TALIB_AVAILABLE:
        macd, signal, hist = talib.MACD(
            close,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        return macd, signal, hist
    else:
        fast_ema = calculate_ema(close, fastperiod)
        slow_ema = calculate_ema(close, slowperiod)
        
        macd = fast_ema - slow_ema
        signal = calculate_ema(macd, signalperiod)
        hist = macd - signal
        
        return macd, signal, hist


# =============================================================================
# DataFrame Functions
# =============================================================================

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data.
        
    Returns:
        DataFrame with added momentum indicators.
    """
    df = df.copy()
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    
    # RSI
    df['RSI'] = calculate_rsi(close, timeperiod=14)
    
    # Stochastic
    slowk, slowd = calculate_stochastic(high, low, close)
    df['Stochastic_K'] = slowk
    df['Stochastic_D'] = slowd
    
    logger.info("Added momentum indicators: RSI, Stochastic_K, Stochastic_D")
    
    return df


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data.
        
    Returns:
        DataFrame with added trend indicators.
    """
    df = df.copy()
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    
    # SMA
    df['SMA'] = calculate_sma(close, timeperiod=20)
    
    # EMA
    df['EMA'] = calculate_ema(close, timeperiod=20)
    
    # ADX
    df['ADX'] = calculate_adx(high, low, close, timeperiod=14)
    
    logger.info("Added trend indicators: SMA, EMA, ADX")
    
    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data.
        
    Returns:
        DataFrame with added volatility indicators.
    """
    df = df.copy()
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    
    # ATR
    df['ATR'] = calculate_atr(high, low, close, timeperiod=14)
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BB_Upper'] = upper
    df['BB_Lower'] = lower
    df['BB_PercentB'] = calculate_bollinger_percentb(close, timeperiod=20, nbdev=2)
    
    logger.info("Added volatility indicators: ATR, BB_Upper, BB_Lower, BB_PercentB")
    
    return df


def add_macd_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MACD indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data.
        
    Returns:
        DataFrame with added MACD indicators.
    """
    df = df.copy()
    
    close = df['Close'].values
    
    # MACD
    macd, signal, hist = calculate_macd(close)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Histogram'] = hist
    
    logger.info("Added MACD indicators: MACD, MACD_Signal, MACD_Histogram")
    
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data.
        
    Returns:
        DataFrame with all indicators added.
    """
    df = add_momentum_indicators(df)
    df = add_trend_indicators(df)
    df = add_volatility_indicators(df)
    df = add_macd_indicators(df)
    
    logger.info(f"Added all indicators. Total columns: {len(df.columns)}")
    
    return df


def add_indicators_for_group(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """
    Add indicators for a specific feature group.
    
    Args:
        df: DataFrame with OHLCV data.
        group_name: Name of the feature group.
        
    Returns:
        DataFrame with group-specific indicators added.
    """
    df = df.copy()
    
    if group_name == 'group0_ohlcv':
        return df
    elif group_name == 'group1_momentum':
        return add_momentum_indicators(df)
    elif group_name == 'group2_trend':
        return add_trend_indicators(df)
    elif group_name == 'group3_volatility':
        return add_volatility_indicators(df)
    elif group_name == 'group4_macd':
        return add_macd_indicators(df)
    elif group_name == 'group5_all':
        return add_all_indicators(df)
    else:
        logger.warning(f"Unknown group: {group_name}. Returning original DataFrame.")
        return df


def get_indicator_feature_names(group_name: str) -> List[str]:
    """
    Get the list of feature names for a specific group.
    
    Args:
        group_name: Name of the feature group.
        
    Returns:
        List of feature names.
    """
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    group_indicators = {
        'group0_ohlcv': [],
        'group1_momentum': ['RSI', 'Stochastic_K', 'Stochastic_D'],
        'group2_trend': ['SMA', 'EMA', 'ADX'],
        'group3_volatility': ['ATR', 'BB_Upper', 'BB_Lower', 'BB_PercentB'],
        'group4_macd': ['MACD', 'MACD_Signal', 'MACD_Histogram'],
        'group5_all': [
            'RSI', 'Stochastic_K', 'Stochastic_D',
            'SMA', 'EMA', 'ADX',
            'ATR', 'BB_Upper', 'BB_Lower', 'BB_PercentB',
            'MACD', 'MACD_Signal', 'MACD_Histogram'
        ]
    }
    
    indicators = group_indicators.get(group_name, [])
    return base_features + indicators


if __name__ == "__main__":
    # Test indicator calculations
    np.random.seed(42)
    
    # Create sample OHLCV data
    n = 500
    base_price = 100
    price = base_price + np.random.randn(n).cumsum()
    
    df = pd.DataFrame({
        'Open': price + np.random.randn(n) * 0.5,
        'High': price + np.abs(np.random.randn(n)) * 1.0,
        'Low': price - np.abs(np.random.randn(n)) * 1.0,
        'Close': price,
        'Volume': np.random.randint(1000, 10000, n)
    })
    
    # Ensure High > Low
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    # Test adding all indicators
    df_with_indicators = add_all_indicators(df)
    print(f"Columns: {list(df_with_indicators.columns)}")
    print(f"Shape: {df_with_indicators.shape}")
    print(f"\nSample data:\n{df_with_indicators.tail()}")
    
    # Check for NaN values
    nan_counts = df_with_indicators.isna().sum()
    print(f"\nNaN counts:\n{nan_counts[nan_counts > 0]}")
