import os
import datetime as dt
from typing import List, Dict, Optional

import yfinance as yf

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except Exception:
    nltk = None
    SentimentIntensityAnalyzer = None

def _ensure_vader():
    global nltk, SentimentIntensityAnalyzer
    if nltk is None or SentimentIntensityAnalyzer is None:
        return None
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

def fetch_yf_news(ticker: str, limit: int = 15):
    try:
        t = yf.Ticker(ticker)
        # Some yfinance versions expose .news or .get_news()
        items = []
        news_list = []
        if hasattr(t, "news"):
            news_list = t.news or []
        elif hasattr(t, "get_news"):
            news_list = t.get_news() or []
        for it in news_list[:limit]:
            headline = it.get("title") or it.get("title", "")
            link = it.get("link") or it.get("url") or ""
            pub_ts = it.get("providerPublishTime") or it.get("published", None)
            if isinstance(pub_ts, (int, float)):
                pub_dt = dt.datetime.utcfromtimestamp(pub_ts)
            else:
                try:
                    pub_dt = dt.datetime.fromisoformat(pub_ts.replace('Z',''))
                except Exception:
                    pub_dt = None
            items.append({"headline": headline, "url": link, "published": pub_dt})
        return items
    except Exception:
        return []

def sentiment_from_headlines(headlines: List[str]) -> Dict[str, float]:
    sia = _ensure_vader()
    if sia is None or not headlines:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
    scores = [sia.polarity_scores(h) for h in headlines]
    if not scores:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
    # Average
    avg = {k: sum(s[k] for s in scores) / len(scores) for k in scores[0].keys()}
    return avg

def classify_sentiment(compound: float):
    if compound >= 0.2:
        return "positive"
    if compound <= -0.2:
        return "negative"
    return "neutral"
