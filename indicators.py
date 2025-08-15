import numpy as np
import pandas as pd

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method='bfill')

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = close.rolling(window=period).mean()
    sd = close.rolling(window=period).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return upper, ma, lower

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period).mean()
    return atr_val

def rolling_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20):
    typical = (high + low + close) / 3.0
    pv = typical * volume
    vwap = pv.rolling(window=window).sum() / volume.rolling(window=window).sum()
    return vwap

def volume_profile(close: pd.Series, volume: pd.Series, bins: int = 24, lookback: int = 90):
    # Simple price-volume histogram for support/resistance estimation
    close_lb = close[-lookback:]
    vol_lb = volume[-lookback:]
    if len(close_lb) < 5:
        return []
    hist, bin_edges = np.histogram(close_lb, bins=bins, weights=vol_lb)
    # Get top 5 levels by volume
    idxs = np.argsort(hist)[-5:][::-1]
    levels = []
    for i in idxs:
        price_level = (bin_edges[i] + bin_edges[i+1]) / 2.0
        levels.append((float(price_level), float(hist[i])))
    return levels
