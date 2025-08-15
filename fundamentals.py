import datetime as dt
from typing import Dict, Optional
import yfinance as yf

def get_sector(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).get_info()
        return info.get("sector", "Unknown") or "Unknown"
    except Exception:
        return "Unknown"

def next_earnings_date(ticker: str):
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if df is None or df.empty:
            return None
        # Get next upcoming
        today = dt.date.today()
        df = df.reset_index()
        for _, row in df.iterrows():
            d = row.get("Earnings Date")
            if hasattr(d, "date"):
                edate = d.date()
            else:
                try:
                    edate = dt.datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
                except Exception:
                    edate = None
            if edate and edate >= today:
                return edate
        return None
    except Exception:
        return None
