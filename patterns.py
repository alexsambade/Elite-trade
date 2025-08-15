import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import List, Dict

def _peak_troughs(series: pd.Series, distance: int = 5, prominence: float = None):
    peaks, _ = find_peaks(series.values, distance=distance, prominence=prominence)
    troughs, _ = find_peaks(-series.values, distance=distance, prominence=prominence)
    return peaks, troughs

def detect_double_top_bottom(close: pd.Series, tolerance: float = 0.02) -> List[Dict]:
    # Double top: two recent peaks within tolerance; Double bottom: two troughs within tolerance
    patterns = []
    peaks, troughs = _peak_troughs(close, distance=max(len(close)//30, 3))
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        lvl1, lvl2 = close.iloc[p1], close.iloc[p2]
        if abs(lvl1 - lvl2) / close.iloc[-1] <= tolerance:
            patterns.append({"pattern": "Double Top", "level": float((lvl1+lvl2)/2), "confidence": 0.6})
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        lvl1, lvl2 = close.iloc[t1], close.iloc[t2]
        if abs(lvl1 - lvl2) / close.iloc[-1] <= tolerance:
            patterns.append({"pattern": "Double Bottom", "level": float((lvl1+lvl2)/2), "confidence": 0.6})
    return patterns

def detect_head_shoulders(close: pd.Series) -> List[Dict]:
    # Heuristic H&S: three peaks with middle highest; Inverse H&S: three troughs with middle lowest
    patterns = []
    peaks, troughs = _peak_troughs(close, distance=max(len(close)//30, 3))
    if len(peaks) >= 3:
        p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
        if close.iloc[p2] > close.iloc[p1] and close.iloc[p2] > close.iloc[p3] and abs(close.iloc[p1] - close.iloc[p3]) / close.iloc[p2] < 0.05:
            neckline = (close.iloc[p1] + close.iloc[p3]) / 2
            patterns.append({"pattern": "Head & Shoulders", "level": float(neckline), "confidence": 0.55})
    if len(troughs) >= 3:
        t1, t2, t3 = troughs[-3], troughs[-2], troughs[-1]
        if close.iloc[t2] < close.iloc[t1] and close.iloc[t2] < close.iloc[t3] and abs(close.iloc[t1] - close.iloc[t3]) / close.iloc[t2] < 0.05:
            neckline = (close.iloc[t1] + close.iloc[t3]) / 2
            patterns.append({"pattern": "Inverse Head & Shoulders", "level": float(neckline), "confidence": 0.55})
    return patterns

def detect_trend_channels(high: pd.Series, low: pd.Series):
    # Simple linear regression on highs and lows to approximate channels/wedges
    import numpy as np
    x = np.arange(len(high))
    if len(x) < 20:
        return []
    up_coef = np.polyfit(x, high.values, 1)  # slope, intercept
    lo_coef = np.polyfit(x, low.values, 1)
    up_slope, lo_slope = up_coef[0], lo_coef[0]
    patterns = []
    # Channel if slopes are similar
    if abs(up_slope - lo_slope) / max(abs(up_slope), 1e-6) < 0.25:
        patterns.append({"pattern": "Rising Channel" if up_slope > 0 else "Falling Channel", "confidence": 0.5})
    # Wedge if slopes converge (upper down, lower up) or diverge
    if up_slope < 0 and lo_slope > 0:
        patterns.append({"pattern": "Falling Wedge (bullish)", "confidence": 0.5})
    if up_slope > 0 and lo_slope < 0:
        patterns.append({"pattern": "Rising Wedge (bearish)", "confidence": 0.5})
    return patterns

def detect_triangles(high: pd.Series, low: pd.Series):
    # Approximate triangles by comparing slopes on highs vs lows
    import numpy as np
    x = np.arange(len(high))
    if len(x) < 20:
        return []
    up_coef = np.polyfit(x, high.values, 1)
    lo_coef = np.polyfit(x, low.values, 1)
    up_slope, lo_slope = up_coef[0], lo_coef[0]
    patterns = []
    if abs(up_slope) < 1e-6 and lo_slope > 0:
        patterns.append({"pattern": "Ascending Triangle", "confidence": 0.5})
    if abs(lo_slope) < 1e-6 and up_slope < 0:
        patterns.append({"pattern": "Descending Triangle", "confidence": 0.5})
    if up_slope < 0 and lo_slope > 0:
        patterns.append({"pattern": "Symmetrical Triangle", "confidence": 0.45})
    return patterns

def detect_all(high: pd.Series, low: pd.Series, close: pd.Series):
    results = []
    results += detect_double_top_bottom(close)
    results += detect_head_shoulders(close)
    results += detect_trend_channels(high, low)
    results += detect_triangles(high, low)
    # Cap to a few most recent/likely
    return results[:4]
