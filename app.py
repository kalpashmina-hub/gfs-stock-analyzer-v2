from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="GFS Stock Analyzer V2", page_icon="📈", layout="wide")
st.title("📈 GFS Stock Analyzer — Version 2")
st.caption("Grandfather: Monthly  |  Father: Weekly  |  Son: Daily  •  EOD decision-support system")

REQUIRED = ["Open", "High", "Low", "Close", "Volume"]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yahoo(symbol: str, period: str):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval="1d", auto_adjust=True, actions=False)
    try:
        info = ticker.get_info()
    except Exception:
        info = {}
    return hist, info


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip().title() for c in d.columns]
    # Accept common CSV date column names or a date index.
    date_col = next((c for c in d.columns if c.lower() in ("date", "datetime", "timestamp")), None)
    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.set_index(date_col)
    d.index = pd.to_datetime(d.index, errors="coerce")
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    d = d[~d.index.isna()].sort_index()
    d = d.loc[~d.index.duplicated(keep="last")]
    # Map adjusted/alternate close header if needed.
    if "Adj Close" in d.columns and "Close" not in d.columns:
        d = d.rename(columns={"Adj Close": "Close"})
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    for c in REQUIRED:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    d = d[d["Close"] > 0]
    return d[REQUIRED]


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d.Close
    for n in (10, 20, 50):
        d[f"EMA{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
    for n in (50, 200):
        d[f"SMA{n}"] = c.rolling(n, min_periods=n).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))
    d.loc[(loss == 0) & (gain > 0), "RSI"] = 100
    d.loc[(gain == 0) & (loss > 0), "RSI"] = 0
    d["RSI_MA9"] = d.RSI.rolling(9, min_periods=9).mean()
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = e12 - e26
    d["MACD_SIGNAL"] = d.MACD.ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d.MACD - d.MACD_SIGNAL
    d["VOL_MA20"] = d.Volume.rolling(20, min_periods=10).mean()
    return d


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample daily OHLCV data. Labels represent period-end dates."""
    return df.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Open", "High", "Low", "Close"])


def completed_resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Return only periods whose calendar end date has passed in India.

    This prevents the developing current week/month from influencing the
    confirmed Father/Grandfather scores. The latest period remains visible
    in the daily data and is deliberately not promoted to a confirmed signal.
    """
    bars = resample_ohlcv(df, rule)
    if bars.empty:
        return bars
    today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    return bars.loc[bars.index.normalize() <= today]


def evaluate(df: pd.DataFrame, label: str, weight: int, is_daily=False):
    if len(df) < 60:
        return None
    r = df.iloc[-1]
    prev = df.iloc[-2]
    checks = {
        "Close above EMA20": bool(pd.notna(r.EMA20) and r.Close > r.EMA20),
        "EMA20 above EMA50": bool(pd.notna(r.EMA50) and r.EMA20 > r.EMA50),
        "RSI above 50": bool(pd.notna(r.RSI) and r.RSI > 50),
        "RSI above its 9-period average": bool(pd.notna(r.RSI_MA9) and r.RSI > r.RSI_MA9),
        "MACD histogram positive": bool(pd.notna(r.MACD_HIST) and r.MACD_HIST > 0),
        "MACD histogram improving": bool(pd.notna(prev.MACD_HIST) and pd.notna(r.MACD_HIST) and r.MACD_HIST > prev.MACD_HIST),
    }
    if is_daily:
        checks["Close above SMA200"] = bool(pd.notna(r.SMA200) and r.Close > r.SMA200)
    passed = sum(checks.values())
    pct = passed / len(checks) * 100
    state = "Bullish" if pct >= 75 else "Improving" if pct >= 50 else "Weak"
    return {"Timeframe":label, "Date":pd.Timestamp(r.name).strftime("%d %b %Y"), "Close":r.Close,
            "RSI":r.RSI, "RSI MA9":r.RSI_MA9, "EMA20":r.EMA20, "EMA50":r.EMA50,
            "SMA200":r.get("SMA200", np.nan), "MACD histogram":r.MACD_HIST,
            "Score":round(pct), "Weight":weight, "Weighted points":pct*weight/100,
            "State":state, "Checks":checks}


def fmt(v, digits=2):
    return "—" if pd.isna(v) else f"{v:,.{digits}f}"


def pivots(df: pd.DataFrame, window=3):
    d = df.tail(180).copy()
    lows, highs = [], []
    for i in range(window, len(d)-window):
        if d.Low.iloc[i] == d.Low.iloc[i-window:i+window+1].min(): lows.append(float(d.Low.iloc[i]))
        if d.High.iloc[i] == d.High.iloc[i-window:i+window+1].max(): highs.append(float(d.High.iloc[i]))
    price = float(d.Close.iloc[-1])
    supports = sorted([x for x in lows if x < price], reverse=True)[:3]
    resistances = sorted([x for x in highs if x > price])[:3]
    return supports, resistances


def fib_levels(df: pd.DataFrame):
    d = df.tail(120)
    lo_idx, hi_idx = d.Low.idxmin(), d.High.idxmax()
    lo, hi = float(d.loc[lo_idx, "Low"]), float(d.loc[hi_idx, "High"])
    # Use the most recent swing direction based on which extreme occurred later.
    if lo_idx < hi_idx:
        return "Up-swing retracement", {"23.6%":hi-(hi-lo)*.236, "38.2%":hi-(hi-lo)*.382, "50.0%":hi-(hi-lo)*.5, "61.8%":hi-(hi-lo)*.618, "78.6%":hi-(hi-lo)*.786}
    return "Down-swing retracement", {"23.6%":lo+(hi-lo)*.236, "38.2%":lo+(hi-lo)*.382, "50.0%":lo+(hi-lo)*.5, "61.8%":lo+(hi-lo)*.618, "78.6%":lo+(hi-lo)*.786}


def make_chart(df, title):
    d = df.tail(220)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.05, row_heights=[.75,.25])
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Price"), row=1,col=1)
    for col, name in [("EMA10","EMA 10"),("EMA20","EMA 20"),("EMA50","EMA 50"),("SMA200","SMA 200")]:
        if col in d and d[col].notna().any(): fig.add_trace(go.Scatter(x=d.index,y=d[col],name=name,mode="lines"),row=1,col=1)
    fig.add_trace(go.Bar(x=d.index,y=d.Volume,name="Volume"),row=2,col=1)
    fig.update_layout(title=title, height=650, xaxis_rangeslider_visible=False, legend_orientation="h", margin=dict(l=15,r=15,t=55,b=15))
    return fig

with st.sidebar:
    st.header("Analysis settings")
    symbol = st.text_input("NSE symbol", "STLTECH").strip().upper()
    source = st.radio("EOD data source", ["Yahoo Finance (free; unofficial)", "Upload OHLCV CSV"], index=0)
    period = st.selectbox("History", ["5y", "10y", "max"], index=1)
    uploaded = st.file_uploader("Upload daily OHLCV CSV", type=["csv"]) if source.startswith("Upload") else None
    st.caption("CSV needs Date, Open, High, Low, Close, Volume columns. Daily bars work best.")
    run = st.button("Run GFS analysis", type="primary", use_container_width=True)
    st.divider()
    st.caption("EOD analysis only. No live prices, trade execution or guaranteed signals.")

if run:
    try:
        with st.spinner("Loading EOD history and calculating indicators…"):
            info = {}
            if source.startswith("Yahoo"):
                yf_symbol = symbol if "." in symbol else symbol + ".NS"
                raw, info = fetch_yahoo(yf_symbol, period)
                if raw.empty and not symbol.endswith(".BO"):
                    st.warning("NSE symbol returned no data. You can try the BSE suffix .BO or upload a CSV.")
                    st.stop()
                raw = clean_ohlcv(raw)
                source_label = f"Yahoo Finance / yfinance ({yf_symbol}) — unofficial EOD feed"
            else:
                if uploaded is None:
                    st.error("Please upload a daily OHLCV CSV first.")
                    st.stop()
                raw = clean_ohlcv(pd.read_csv(uploaded))
                source_label = f"User-uploaded CSV: {uploaded.name}"
        if len(raw) < 220:
            st.error(f"Only {len(raw)} daily rows are available. Please provide at least 220 trading sessions for the daily SMA200 and meaningful multi-timeframe analysis.")
            st.stop()
        daily = indicators(raw)
        weekly = indicators(completed_resample_ohlcv(raw, "W-FRI"))
        monthly = indicators(completed_resample_ohlcv(raw, "ME"))
        results = [evaluate(monthly, "Grandfather / Monthly", 35), evaluate(weekly, "Father / Weekly", 35), evaluate(daily, "Son / Daily", 30, True)]
        if any(x is None for x in results):
            st.error("Insufficient history after resampling. Upload a longer price history.")
            st.stop()
        g, f, s = results
        total = round(sum(x["Weighted points"] for x in results))
        st.subheader(info.get("longName", symbol) + (f" ({yf_symbol})" if source.startswith("Yahoo") else ""))
        st.caption(f"Source: {source_label}  |  Last available daily bar: {s['Date']}  |  {len(raw):,} daily sessions")
        st.warning("Data quality: verify the latest bar date and corporate-action adjustments. Yahoo Finance is an unofficial source; CSV data quality depends on the uploaded file.")

        cols = st.columns(4)
        for col, item in zip(cols, [g, f, s, {"Score":total, "State":"Composite GFS", "Timeframe":"Overall"}]):
            with col:
                st.metric(item["Timeframe"], f"{item['Score']}%" if item["Timeframe"] != "Overall" else f"{total}/100", item.get("State", ""))
        st.subheader("GFS decision framework")
        if g["State"] == "Bullish" and f["State"] == "Bullish" and s["State"] == "Bullish":
            verdict = "STRONG BUY CANDIDATE — only after entry, valuation and risk checks"
            st.success(verdict)
        elif g["State"] == "Bullish" and f["State"] in ("Bullish", "Improving") and s["State"] != "Weak":
            verdict = "BUY ON CONFIRMATION — require price-action and volume confirmation"
            st.success(verdict)
        elif g["State"] != "Weak" and (f["State"] == "Improving" or s["State"] == "Improving"):
            verdict = "WATCHLIST — wait for clearer weekly/daily alignment"
            st.warning(verdict)
        elif all(x["State"] == "Weak" for x in results):
            verdict = "AVOID FRESH ENTRY — all three timeframes are weak"
            st.error(verdict)
        else:
            verdict = "WAIT / WATCHLIST — mixed timeframe signals"
            st.warning(verdict)
        st.caption("The 0–100 score is a transparent checklist score, not a probability of profit. It is not a personalised recommendation. Confirm fundamentals, liquidity, news, position sizing and reward/risk before acting.")

        st.subheader("Timeframe indicator dashboard")
        dashboard = pd.DataFrame([{k:v for k,v in x.items() if k not in ("Checks",)} for x in results])
        st.dataframe(dashboard, use_container_width=True, hide_index=True)
        with st.expander("Detailed GFS checklist", expanded=True):
            rows=[]
            for item in results:
                for check, passed in item["Checks"].items(): rows.append({"Timeframe":item["Timeframe"], "Condition":check, "Result":"Pass" if passed else "Not met"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Daily chart")
        st.plotly_chart(make_chart(daily, f"{symbol} — Daily candles and moving averages"), use_container_width=True)
        st.subheader("Momentum dashboard")
        mom = daily.tail(250)
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=mom.index, y=mom["RSI"], name="RSI 14", mode="lines"))
        rsi_fig.add_trace(go.Scatter(x=mom.index, y=mom["RSI_MA9"], name="RSI MA 9", mode="lines"))
        for level in (30, 50, 70):
            rsi_fig.add_hline(y=level, line_dash="dot", annotation_text=str(level))
        rsi_fig.update_yaxes(range=[0, 100], title="RSI")
        rsi_fig.update_layout(title="RSI 14 with 9-period average", height=320, margin=dict(l=15, r=15, t=45, b=15), legend_orientation="h")
        st.plotly_chart(rsi_fig, use_container_width=True)

        macd_fig = go.Figure()
        macd_fig.add_trace(go.Bar(x=mom.index, y=mom["MACD_HIST"], name="Histogram"))
        macd_fig.add_trace(go.Scatter(x=mom.index, y=mom["MACD"], name="MACD", mode="lines"))
        macd_fig.add_trace(go.Scatter(x=mom.index, y=mom["MACD_SIGNAL"], name="Signal", mode="lines"))
        macd_fig.update_layout(title="MACD (12, 26, 9)", height=320, margin=dict(l=15, r=15, t=45, b=15), legend_orientation="h")
        st.plotly_chart(macd_fig, use_container_width=True)
        st.caption("RSI 40–60 is an assessment zone, not a rigid filter. RSI above 60 can reflect strong momentum; below 30 is not an automatic buy signal.")

        st.subheader("Support, resistance and Fibonacci")
        supports, resistances = pivots(daily)
        fib_name, fib = fib_levels(daily)
        a,b,c = st.columns(3)
        with a:
            st.markdown("**Nearby support pivots**")
            st.write([fmt(x) for x in supports] or "No confirmed pivot support found")
        with b:
            st.markdown("**Nearby resistance pivots**")
            st.write([fmt(x) for x in resistances] or "No confirmed pivot resistance found")
        with c:
            st.markdown(f"**{fib_name} — 120-session range**")
            st.write({k:fmt(v) for k,v in fib.items()})
        st.caption("Pivot levels are mechanical local highs/lows and Fibonacci uses the 120-session extreme range. Validate swing anchors and levels visually; they are not guaranteed turning points.")

        st.subheader("Fundamental snapshot")
        if info:
            fields = [
                ("Market cap", "marketCap", "number"), ("Trailing P/E", "trailingPE", "number"),
                ("Forward P/E", "forwardPE", "number"), ("Price / book", "priceToBook", "number"),
                ("Return on equity", "returnOnEquity", "percent"), ("Debt / equity", "debtToEquity", "number"),
                ("Profit margin", "profitMargins", "percent"), ("Revenue growth", "revenueGrowth", "percent"),
                ("Earnings growth", "earningsGrowth", "percent"), ("52-week high", "fiftyTwoWeekHigh", "number"),
                ("52-week low", "fiftyTwoWeekLow", "number")]
            fund_rows=[]
            for label,key,kind in fields:
                value=info.get(key)
                if value is not None:
                    if kind=="percent": value=f"{value*100:.2f}%"
                    elif kind=="number": value=f"{value:,.2f}" if isinstance(value,(int,float)) else str(value)
                    fund_rows.append({"Metric":label,"Provider value":value})
            if fund_rows:
                st.dataframe(pd.DataFrame(fund_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No fundamental fields returned by the provider.")
        else:
            st.info("Fundamentals are not included for CSV uploads. Use company annual reports, quarterly results and exchange filings for verification.")
        st.caption("Fundamental data is provider-reported, may be missing or stale, and is not a substitute for reading audited statements, quarterly results, shareholding and exchange announcements.")

        st.subheader("Export")
        csv = daily.to_csv(index_label="Date").encode("utf-8")
        st.download_button("Download daily indicators CSV", csv, file_name=f"{symbol}_GFS_V2_daily.csv", mime="text/csv")
        summary_csv = pd.DataFrame([{k:v for k,v in x.items() if k!="Checks"} for x in results]).to_csv(index=False).encode("utf-8")
        st.download_button("Download GFS summary CSV", summary_csv, file_name=f"{symbol}_GFS_V2_summary.csv", mime="text/csv")
    except Exception as exc:
        st.error(f"Analysis could not be completed: {exc}")
        st.info("Try another symbol, upload a clean OHLCV CSV, or retry later. The app does not silently replace missing data with estimates.")
else:
    st.markdown("### What Version 2 includes")
    st.markdown("- Monthly / weekly / daily GFS scoring with 35 / 35 / 30 weighting\n- EMA 10/20/50, SMA 50/200, RSI 14 + RSI MA 9, MACD and volume average\n- Candlestick chart, momentum panels, pivot support/resistance and Fibonacci retracement estimates\n- Fundamental snapshot when the selected provider supplies it\n- CSV export and a user-uploaded EOD CSV fallback")
    st.info("Start with an NSE symbol such as INFY, TCS, STLTECH, COCHINSHIP or BDL. The Yahoo feed is free but unofficial; for higher reliability, use a licensed EOD provider or upload its daily OHLCV export.")
