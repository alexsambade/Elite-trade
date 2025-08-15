import streamlit as st
import pandas as pd
import datetime as dt

from analysis import generate_signal
from utils import describe_risk

st.set_page_config(page_title="Elite Swing-Trade Assistant", layout="wide")

CUSTOM_CSS = """
<style>
.big-num {font-size: 1.2rem; font-weight: 600;}
.kpill {padding: 6px 10px; border-radius: 8px; display: inline-block; margin-right: 6px; background: rgba(0,0,0,0.05);}
.urgent {background: #ffe5e5; border-left: 6px solid #ff3b30; padding: 10px; border-radius: 8px;}
.good {color: #0a7f4b; font-weight: 600;}
.bad {color: #b00020; font-weight: 600;}
table.dataframe td, table.dataframe th {font-size: 0.95rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("📈 Elite US Stock Swing-Trading Assistant")
st.caption("For educational purposes only. Not investment advice. Always do your own research.")

col1, col2 = st.columns([1,1])
with col1:
    ticker = st.text_input("Ticker (US)", value="AAPL").upper().strip()
with col2:
    risk = st.selectbox("Risk Profile", ["Low", "Medium", "High"], index=1, help=describe_risk("Medium"))

if st.button("🔍 Generate Signal", type="primary"):
    with st.spinner("Analyzing..."):
        out = generate_signal(ticker, risk)

    if out.get("status") != "ok":
        st.error(out.get("message", "Unknown error"))
    else:
        if out["signal"] == "HOLD":
            st.info("**No high-probability trade available.**\n\nAlignment insufficient based on current technicals, fundamentals, and sentiment.")
        else:
            if out.get("urgent"):
                st.markdown(f'<div class="urgent">⏱️ <b>Time-sensitive:</b> Price is near the entry band for {ticker}.</div>', unsafe_allow_html=True)
            kcols = st.columns(5)
            kcols[0].metric("Signal", out["signal"])
            kcols[1].metric("Score", out["score"])
            kcols[2].metric("ATR(14)", out["atr"])
            kcols[3].metric("VWAP(20)", out.get("vwap20", None))
            kcols[4].metric("Sector", out.get("sector", "—"))

            df = pd.DataFrame([{
                "Ticker": ticker,
                "Direction": out["signal"],
                "Entry Range": f'{out["entry"][0]} – {out["entry"][1]}',
                "Stop-Loss": out["stop"],
                "Targets": ", ".join(map(str, out["targets"])),
                "Best R/R": max(out["rr"]) if out.get("rr") else None,
                "Score": out["score"],
                "When": out["timestamp"]
            }])
            st.dataframe(df, use_container_width=True)

        with st.expander("🔎 Why this signal? (key factors)"):
            st.write(", ".join(out.get("technical_notes", [])) or "—")
            st.write("Patterns detected: " + ", ".join(p["pattern"] for p in out.get("patterns", [])) if out.get("patterns") else "Patterns: —")
            if out.get("bbands"):
                st.write(f"Bollinger Bands (20,2): U {out['bbands']['upper']} | M {out['bbands']['mid']} | L {out['bbands']['lower']}")

        with st.expander("📰 News & Sentiment"):
            news = out["news"]
            st.write(f"Sentiment: **{news['sentiment']}** (compound {round(news['compound'],2)})")
            if news.get("earnings_in_days") is not None:
                st.write(f"Earnings in **{news['earnings_in_days']}** day(s)")
            if news.get("raw_news"):
                for n in news["raw_news"]:
                    st.write(f"- [{n['headline']}]({n['url']})  {'(' + str(n['published']) + ')' if n.get('published') else ''}")
            else:
                st.write("No recent Yahoo Finance headlines available.")

st.caption("Built with Streamlit • Indicators: EMA, MACD, RSI, Stoch, Bollinger, VWAP, ATR • Patterns: H&S, Double Tops/Bottoms, Triangles, Channels, Wedges")
