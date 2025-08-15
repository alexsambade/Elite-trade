import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import ema, macd, rsi, stochastic, bollinger_bands, atr, rolling_vwap, volume_profile
from patterns import detect_all
from news import fetch_yf_news, sentiment_from_headlines, classify_sentiment
from fundamentals import get_sector, next_earnings_date

RISK_SETTINGS = {
    "Low":   {"confirm_buffer_atr": 1.0, "stop_mult": 1.8, "target_mults": [1.2, 2.0], "min_score": 0.65},
    "Medium":{"confirm_buffer_atr": 0.6, "stop_mult": 1.4, "target_mults": [1.5, 2.5], "min_score": 0.6},
    "High":  {"confirm_buffer_atr": 0.3, "stop_mult": 1.1, "target_mults": [1.8, 3.0], "min_score": 0.55},
}

def fetch_history(ticker: str, period: str, interval: str):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adjclose", "Volume": "volume"})
    # normalize lower-case
    for c in list(df.columns):
        if c.lower() != c:
            df.rename(columns={c: c.lower()}, inplace=True)
    return df

def technical_snapshot(df: pd.DataFrame):
    out = {}
    out["ema9"] = ema(df["close"], 9)
    out["ema21"] = ema(df["close"], 21)
    out["ema50"] = ema(df["close"], 50)
    out["ema200"] = ema(df["close"], 200)
    macd_line, sig, hist = macd(df["close"])
    out["macd_line"], out["macd_signal"], out["macd_hist"] = macd_line, sig, hist
    out["rsi14"] = rsi(df["close"], 14)
    k, d = stochastic(df["high"], df["low"], df["close"], 14, 3)
    out["stoch_k"], out["stoch_d"] = k, d
    bb_u, bb_m, bb_l = bollinger_bands(df["close"], 20, 2.0)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bb_u, bb_m, bb_l
    out["atr14"] = atr(df["high"], df["low"], df["close"], 14)
    out["vwap20"] = rolling_vwap(df["high"], df["low"], df["close"], df["volume"], 20)
    out["vol_profile"] = volume_profile(df["close"], df["volume"], bins=24, lookback=min(90, len(df)))
    out["patterns"] = detect_all(df["high"], df["low"], df["close"])
    return out

def score_technicals(df: pd.DataFrame, snap: Dict) -> Tuple[float, str, Dict]:
    # Basic momentum/trend score
    latest = df.iloc[-1]
    price = latest["close"]
    score = 0.0
    notes = []

    # EMA alignment
    ema_ok_long = price > snap["ema9"].iloc[-1] > snap["ema21"].iloc[-1] > snap["ema50"].iloc[-1]
    ema_ok_short = price < snap["ema9"].iloc[-1] < snap["ema21"].iloc[-1] < snap["ema50"].iloc[-1]

    if ema_ok_long:
        score += 0.15; notes.append("EMA stack bullish")
    if ema_ok_short:
        score -= 0.15; notes.append("EMA stack bearish")

    # MACD
    if snap["macd_line"].iloc[-1] > snap["macd_signal"].iloc[-1] and snap["macd_hist"].iloc[-1] > 0:
        score += 0.15; notes.append("MACD bull cross")
    elif snap["macd_line"].iloc[-1] < snap["macd_signal"].iloc[-1] and snap["macd_hist"].iloc[-1] < 0:
        score -= 0.15; notes.append("MACD bear cross")

    # RSI
    r = snap["rsi14"].iloc[-1]
    if 50 < r < 70:
        score += 0.1; notes.append("RSI bullish")
    elif 30 < r < 50:
        score -= 0.1; notes.append("RSI bearish")
    elif r >= 70:
        notes.append("RSI overbought")
    elif r <= 30:
        notes.append("RSI oversold")

    # Stochastic
    k, d = snap["stoch_k"].iloc[-1], snap["stoch_d"].iloc[-1]
    if k > d and k < 80:
        score += 0.05; notes.append("Stoch up")
    elif k < d and k > 20:
        score -= 0.05; notes.append("Stoch down")

    # Bollinger position
    if price > snap["bb_mid"].iloc[-1]:
        score += 0.05; notes.append("Above BB mid")
    else:
        score -= 0.05; notes.append("Below BB mid")

    # Pattern influence
    for p in snap["patterns"]:
        if "Wedge (bullish)" in p["pattern"] or "Inverse Head" in p["pattern"] or "Double Bottom" in p["pattern"] or "Ascending Triangle" in p["pattern"]:
            score += 0.08
            notes.append(f"Pattern bullish: {p['pattern']}")
        if "Rising Wedge" in p["pattern"] or "Head & Shoulders" in p["pattern"] or "Double Top" in p["pattern"] or "Descending Triangle" in p["pattern"]:
            score -= 0.08
            notes.append(f"Pattern bearish: {p['pattern']}")

    # Normalize to 0..1 bullishness, negative means bearishness
    # We'll return a direction bias and magnitude
    bias = "bullish" if score >= 0 else "bearish"
    magnitude = min(abs(score), 1.0)
    return magnitude, bias, {"notes": notes}

def build_entry_exit(df: pd.DataFrame, snap: Dict, bias: str, risk: str):
    latest = df.iloc[-1]
    price = latest["close"]
    atrv = snap["atr14"].iloc[-1]
    params = RISK_SETTINGS.get(risk, RISK_SETTINGS["Medium"])

    # Determine breakout/breakdown levels based on recent highs/lows and patterns
    recent = df[-20:]
    breakout = recent["high"].max()
    breakdown = recent["low"].min()

    if bias == "bullish":
        entry_low = breakout + params["confirm_buffer_atr"] * atrv * 0.2
        entry_high = entry_low + 0.3 * atrv
        stop = entry_low - params["stop_mult"] * atrv
        targets = [entry_low + m * atrv for m in params["target_mults"]]
        rr = [(t - entry_low) / (entry_low - stop) for t in targets]
        return (entry_low, entry_high), stop, targets, rr
    else:
        entry_high = breakdown - params["confirm_buffer_atr"] * atrv * 0.2
        entry_low = entry_high - 0.3 * atrv
        stop = entry_high + params["stop_mult"] * atrv
        targets = [entry_high - m * atrv for m in params["target_mults"]]
        rr = [(entry_high - t) / (stop - entry_high) for t in targets]
        return (entry_low, entry_high), stop, targets, rr

def fundamental_news_bias(ticker: str):
    news = fetch_yf_news(ticker)
    headlines = [n["headline"] for n in news if n.get("headline")]
    sent = sentiment_from_headlines(headlines)
    label = classify_sentiment(sent["compound"])
    # Earnings proximity penalty
    edate = next_earnings_date(ticker)
    days_to_earn = None
    if edate:
        days_to_earn = (edate - dt.date.today()).days
    penalty = 0.0
    if days_to_earn is not None and days_to_earn <= 3:
        penalty = 0.15  # De-risk around earnings
    return {"sentiment": label, "compound": sent["compound"], "news_count": len(headlines), "earnings_in_days": days_to_earn, "penalty": penalty, "raw_news": news[:8]}

def combine_scores(tech_mag: float, tech_bias: str, fn: dict):
    base = tech_mag
    # Apply sentiment alignment
    if fn["sentiment"] == "positive" and tech_bias == "bullish":
        base += 0.15
    elif fn["sentiment"] == "negative" and tech_bias == "bearish":
        base += 0.15
    else:
        base -= 0.1  # disagreement penalty

    base -= fn.get("penalty", 0.0)
    # Clamp 0..1
    return max(0.0, min(1.0, base))

def generate_signal(ticker: str, risk: str = "Medium"):
    # Pull multi-timeframe data
    d1 = fetch_history(ticker, period="1y", interval="1d")
    w1 = fetch_history(ticker, period="5y", interval="1wk")
    m1 = fetch_history(ticker, period="10y", interval="1mo")

    if d1 is None or d1.empty:
        return {"status": "error", "message": f"No data for {ticker}"}

    snap_d = technical_snapshot(d1)
    snap_w = technical_snapshot(w1) if not w1.empty else None
    snap_m = technical_snapshot(m1) if not m1.empty else None

    # Aggregate technical bias (weighted: 1D 0.5, 1W 0.3, 1M 0.2)
    md_d, b_d, n_d = score_technicals(d1, snap_d)
    md_w, b_w, n_w = (0, "neutral", {"notes": []})
    md_m, b_m, n_m = (0, "neutral", {"notes": []})
    if snap_w is not None:
        md_w, b_w, n_w = score_technicals(w1, snap_w)
    if snap_m is not None:
        md_m, b_m, n_m = score_technicals(m1, snap_m)

    # Direction voting
    votes = {"bullish": 0.0, "bearish": 0.0}
    weights = {"d": 0.5, "w": 0.3, "m": 0.2}
    votes[b_d] += md_d * weights["d"]
    votes[b_w] += md_w * weights["w"]
    votes[b_m] += md_m * weights["m"]
    tech_bias = "bullish" if votes["bullish"] >= votes["bearish"] else "bearish"
    tech_mag = votes[tech_bias]

    fn = fundamental_news_bias(ticker)
    combined = combine_scores(tech_mag, tech_bias, fn)
    params = RISK_SETTINGS.get(risk, RISK_SETTINGS["Medium"])

    if combined < params["min_score"]:
        return {
            "status": "ok",
            "signal": "HOLD",
            "reason": "No high-probability trade available - alignment insufficient",
            "score": combined,
            "technical_bias": tech_bias,
            "technical_notes": list(set(n_d["notes"] + n_w["notes"] + n_m["notes"]))[:8],
            "news": fn,
        }

    # Build entries/exits using daily frame
    entry_range, stop, targets, rr = build_entry_exit(d1, snap_d, tech_bias, risk)
    direction = "LONG" if tech_bias == "bullish" else "SHORT"
    # Time sensitivity: if last close within 0.5*ATR of entry range, mark urgent
    atrv = snap_d["atr14"].iloc[-1]
    px = d1.iloc[-1]["close"]
    urgent = (entry_range[0] - 0.5 * atrv) <= px <= (entry_range[1] + 0.5 * atrv)

    return {
        "status": "ok",
        "signal": direction,
        "entry": [float(round(entry_range[0], 2)), float(round(entry_range[1], 2))],
        "stop": float(round(stop, 2)),
        "targets": [float(round(t, 2)) for t in targets],
        "rr": [float(round(x, 2)) for x in rr],
        "score": float(round(combined, 2)),
        "technical_bias": tech_bias,
        "technical_notes": list(set(n_d["notes"] + n_w["notes"] + n_m["notes"]))[:8],
        "patterns": snap_d["patterns"],
        "vwap20": float(round(snap_d["vwap20"].iloc[-1], 2)) if not np.isnan(snap_d["vwap20"].iloc[-1]) else None,
        "bbands": {
            "upper": float(round(snap_d["bb_upper"].iloc[-1], 2)),
            "mid": float(round(snap_d["bb_mid"].iloc[-1], 2)),
            "lower": float(round(snap_d["bb_lower"].iloc[-1], 2)),
        },
        "atr": float(round(atrv, 2)),
        "news": fn,
        "sector": get_sector(ticker),
        "urgent": urgent,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds")
    }
