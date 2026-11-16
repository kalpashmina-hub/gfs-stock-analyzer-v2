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
    # Group related signals so four momentum checks cannot outweigh the
    # underlying price-trend structure. The daily SMA200 check belongs to trend.
    trend_checks = ["Close above EMA20", "EMA20 above EMA50"]
    if is_daily:
        trend_checks.append("Close above SMA200")
    momentum_checks = ["RSI above 50", "MACD histogram positive"]
    confirmation_checks = ["RSI above its 9-period average", "MACD histogram improving"]

    def group_pct(names):
        return sum(checks[name] for name in names) / len(names) * 100

    trend_pct = group_pct(trend_checks)
    momentum_pct = group_pct(momentum_checks)
    confirmation_pct = group_pct(confirmation_checks)
    pct = trend_pct * 0.40 + momentum_pct * 0.35 + confirmation_pct * 0.25
    state = "Bullish" if pct >= 75 else "Improving" if pct >= 50 else "Weak"
    check_groups = {name: "Trend" for name in trend_checks}
    check_groups.update({name: "Momentum" for name in momentum_checks})
    check_groups.update({name: "Confirmation" for name in confirmation_checks})
    return {"Timeframe":label, "Date":pd.Timestamp(r.name).strftime("%d %b %Y"), "Close":r.Close,
            "RSI":r.RSI, "RSI MA9":r.RSI_MA9, "EMA20":r.EMA20, "EMA50":r.EMA50,
            "SMA200":r.get("SMA200", np.nan),
            "Trend score":round(trend_pct), "Momentum score":round(momentum_pct),
            "Confirmation score":round(confirmation_pct),
            "Checks passed":sum(checks.values()), "Checks total":len(checks),
            "MACD histogram":r.MACD_HIST, "Score":round(pct), "Weight":weight,
            "Weighted points":pct*weight/100, "State":state, "Checks":checks,
            "Check groups":check_groups}


def fmt(v, digits=2):
    return "—" if pd.isna(v) else f"{v:,.{digits}f}"

def volume_context(df: pd.DataFrame):
    """Describe latest completed-bar volume versus its 20-period average."""
    if df.empty or "Volume" not in df or "VOL_MA20" not in df:
        return {"Volume": "Unavailable", "Relative volume": "—", "Price direction": "—"}
    row = df.iloc[-1]
    avg = row.get("VOL_MA20", np.nan)
    vol = row.get("Volume", np.nan)
    if pd.isna(avg) or pd.isna(vol) or avg <= 0:
        relative = np.nan
        label = "Insufficient volume history"
    else:
        relative = float(vol / avg)
        label = "Elevated" if relative >= 1.5 else "Quiet" if relative < 0.8 else "Near average"
    direction = "Up bar" if row.Close > row.Open else "Down bar" if row.Close < row.Open else "Flat bar"
    return {"Volume": label, "Relative volume": "—" if pd.isna(relative) else f"{relative:.2f}x",
            "Price direction": direction}


def format_market_cap_inr(value):
    """Format Yahoo Finance market capitalisation in Indian numbering units."""
    if value is None or pd.isna(value):
        return "—"
    amount = float(value)
    crore = amount / 10_000_000
    lakh_crore = crore / 100_000
    if lakh_crore >= 1:
        return f"₹{lakh_crore:,.2f} lakh crore"
    return f"₹{crore:,.2f} crore"


def confirmed_swings(df: pd.DataFrame, window=5, lookback=250):
    """Find confirmed fractal swing points; each point is confirmed only after
    'window' bars have formed on its right side.
    """
    d = df.tail(lookback).copy()
    points = []
    if len(d) < window * 2 + 5:
        return points
    for i in range(window, len(d) - window):
        low_window = d.Low.iloc[i-window:i+window+1]
        high_window = d.High.iloc[i-window:i+window+1]
        if d.Low.iloc[i] == low_window.min():
            points.append({"date": d.index[i], "price": float(d.Low.iloc[i]), "kind": "Low"})
        if d.High.iloc[i] == high_window.max():
            points.append({"date": d.index[i], "price": float(d.High.iloc[i]), "kind": "High"})
    points.sort(key=lambda x: x["date"])

    # Consecutive highs or lows are consolidated into the more extreme point.
    alternating = []
    for point in points:
        if alternating and alternating[-1]["kind"] == point["kind"]:
            previous = alternating[-1]
            more_extreme = (point["price"] < previous["price"] if point["kind"] == "Low"
                            else point["price"] > previous["price"])
            if more_extreme:
                alternating[-1] = point
        else:
            alternating.append(point)
    return alternating


def pivots(df: pd.DataFrame, window=5, lookback=250, reference_price=None):
    """Return confirmed swing levels relative to a shared reference price."""
    d = df.tail(lookback)
    swings = confirmed_swings(d, window=window, lookback=lookback)
    price = float(reference_price) if reference_price is not None else float(d.Close.iloc[-1])
    supports = sorted(
        [p for p in swings if p["kind"] == "Low" and p["price"] < price],
        key=lambda p: p["price"], reverse=True
    )
    resistances = sorted(
        [p for p in swings if p["kind"] == "High" and p["price"] > price],
        key=lambda p: p["price"]
    )
    return supports, resistances


def split_nearby_levels(points, reference_price, max_distance_pct=10):
    """Separate levels within the actionable distance band from farther structural levels."""
    nearby, major = [], []
    for point in points:
        distance_pct = abs(float(point["price"]) / float(reference_price) - 1) * 100
        (nearby if distance_pct <= max_distance_pct else major).append(point)
    return nearby[:3], major[:3]


def fib_levels(df: pd.DataFrame, current_price=None):
    """Select a confirmed, material daily swing and assess its current validity.

    A candidate must alternate direction, span at least five sessions and move
    at least the greater of 3% or two ATR(14). Old anchors remain visible but
    are explicitly marked stale rather than presented as an active setup.
    """
    d = df.dropna(subset=["High", "Low", "Close"]).tail(250).copy()
    swings = confirmed_swings(d, window=5, lookback=250)
    if len(swings) < 2:
        return None, "Insufficient confirmed swing history", "Unavailable"

    # ATR(14) adapts the minimum impulse requirement to the stock's volatility.
    previous_close = d.Close.shift(1)
    true_range = pd.concat([
        d.High - d.Low,
        (d.High - previous_close).abs(),
        (d.Low - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean().iloc[-1]
    latest_close = float(d.Close.iloc[-1])
    if pd.isna(atr) or latest_close <= 0:
        return None, "Insufficient data to validate swing size (ATR unavailable)", "Unavailable"

    pair = None
    for i in range(len(swings) - 1, 0, -1):
        first, second = swings[i-1], swings[i]
        if first["kind"] == second["kind"]:
            continue
        try:
            first_pos = d.index.get_loc(pd.Timestamp(first["date"]))
            second_pos = d.index.get_loc(pd.Timestamp(second["date"]))
        except KeyError:
            continue
        span = second_pos - first_pos
        if span < 5:
            continue
        move_abs = abs(float(second["price"]) - float(first["price"]))
        move_pct = move_abs / float(first["price"]) * 100
        minimum_move = max(float(first["price"]) * 0.03, 2.0 * float(atr))
        if move_abs >= minimum_move:
            pair = (first, second, move_pct, span, len(d) - 1 - second_pos)
            break
    if pair is None:
        return None, "No confirmed swing meets the minimum size and duration checks (move must be at least the greater of 3% or 2× ATR, across at least 5 sessions)", "Unavailable"

    first, second, move_pct, span, age_sessions = pair
    low = min(float(first["price"]), float(second["price"]))
    high = max(float(first["price"]), float(second["price"]))
    first_date = pd.Timestamp(first["date"])
    second_date = pd.Timestamp(second["date"])
    price = float(current_price) if current_price is not None else latest_close

    if first["kind"] == "Low" and second["kind"] == "High":
        direction = "Up-swing"
        label = (f"Up-swing retracement ({first_date.strftime('%d %b %Y')} low to "
                 f"{second_date.strftime('%d %b %Y')} high; {move_pct:.1f}% move; "
                 f"{span} sessions; anchor age {age_sessions} sessions)")
        levels = {k: high - (high-low)*v for k, v in
                  [("23.6%",.236),("38.2%",.382),("50.0%",.5),("61.8%",.618),("78.6%",.786)]}
        if price < low:
            status = f"Invalidated: latest close ₹{price:,.2f} is below the swing low ₹{low:,.2f}. Do not treat these retracements as active support."
        elif price > high:
            status = f"Upswing extended: latest close ₹{price:,.2f} is above the swing high ₹{high:,.2f}. These are historical reference levels, not an active pullback setup."
        elif age_sessions > 90:
            status = f"Stale reference: the swing high is {age_sessions} sessions old. Keep these levels for historical context only; wait for a fresh confirmed swing."
        elif price < levels["78.6%"]:
            status = f"Deep retracement: latest close has moved beyond the 78.6% level but remains above the swing low ₹{low:,.2f}. Treat the impulse as severely retraced; do not assume the Fibonacci levels are reliable active support."
        else:
            status = f"Valid anchor range: latest close is within ₹{low:,.2f}–₹{high:,.2f}. Treat Fibonacci prices as potential reaction zones, not confirmed support."
    else:
        direction = "Down-swing"
        label = (f"Down-swing retracement ({first_date.strftime('%d %b %Y')} high to "
                 f"{second_date.strftime('%d %b %Y')} low; {move_pct:.1f}% move; "
                 f"{span} sessions; anchor age {age_sessions} sessions)")
        levels = {k: low + (high-low)*v for k, v in
                  [("23.6%",.236),("38.2%",.382),("50.0%",.5),("61.8%",.618),("78.6%",.786)]}
        if price > high:
            status = f"Invalidated: latest close ₹{price:,.2f} is above the swing high ₹{high:,.2f}. Do not treat these retracements as active resistance."
        elif price < low:
            status = f"Downswing extended: latest close ₹{price:,.2f} is below the swing low ₹{low:,.2f}. These are historical reference levels, not an active pullback setup."
        elif age_sessions > 90:
            status = f"Stale reference: the swing low is {age_sessions} sessions old. Keep these levels for historical context only; wait for a fresh confirmed swing."
        elif price > levels["78.6%"]:
            status = f"Deep retracement: latest close has moved beyond the 78.6% level but remains below the swing high ₹{high:,.2f}. Treat the impulse as severely retraced; do not assume the Fibonacci levels are reliable active resistance."
        else:
            status = f"Valid anchor range: latest close is within ₹{low:,.2f}–₹{high:,.2f}. Treat Fibonacci prices as potential reaction zones, not confirmed resistance."
    return levels, label, status


def make_chart(df, title):
    # Keep the full selected history available; use Plotly range controls to
    # navigate long histories without forcing thousands of candles into view.
    d = df.copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.05, row_heights=[.75,.25])
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Price"), row=1,col=1)
    for col, name in [("EMA10","EMA 10"),("EMA20","EMA 20"),("EMA50","EMA 50"),("SMA200","SMA 200")]:
        if col in d and d[col].notna().any(): fig.add_trace(go.Scatter(x=d.index,y=d[col],name=name,mode="lines"),row=1,col=1)
    fig.add_trace(go.Bar(x=d.index,y=d.Volume,name="Volume"),row=2,col=1)
    range_buttons = [
        dict(count=6, label="6m", step="month", stepmode="backward"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(count=3, label="3y", step="year", stepmode="backward"),
        dict(count=5, label="5y", step="year", stepmode="backward"),
        dict(step="all", label="All"),
    ]
    fig.update_layout(
        title=title, height=650,
        xaxis=dict(rangeselector=dict(buttons=range_buttons), rangeslider=dict(visible=True)),
        legend_orientation="h", margin=dict(l=15,r=15,t=55,b=15)
    )
    # Start with a readable one-year view; the range controls expose all
    # candles from the selected 5y/10y/max history.
    if len(d.index):
        fig.update_xaxes(range=[d.index[-1] - pd.DateOffset(years=1), d.index[-1]], row=1, col=1)
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
                # State is a category, not a numeric delta. Show it separately
                # so Streamlit does not imply that even a "Weak" state is positive.
                st.metric(item["Timeframe"], f"{item['Score']}%" if item["Timeframe"] != "Overall" else f"{total}/100")
                st.caption(item.get("State", ""))
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

        # Momentum extremes are entry-timing cautions, not trend-score penalties.
        overbought = []
        for item in results:
            rsi_value = item.get("RSI")
            if pd.notna(rsi_value) and rsi_value >= 70:
                overbought.append(f"{item['Timeframe'].split('/')[-1].strip()} RSI is {rsi_value:.1f}")
        if overbought:
            st.info("Momentum caution — " + "; ".join(overbought) +
                    ". Strong momentum can persist, so this does not reduce the GFS trend score; avoid chasing and assess pullbacks, price action and risk/reward before entry.")
        st.caption("Each timeframe score weights Trend 40%, Momentum 35% and Confirmation 25%. Related RSI/MACD checks are grouped so momentum signals do not dominate the underlying price trend. Monthly/weekly/daily composite weights remain 35/35/30. The 0–100 score is a checklist, not a probability of profit or a personalised recommendation; confirm fundamentals, liquidity, news, position sizing and reward/risk before acting.")

        st.subheader("GFS score breakdown")
        score_rows = []
        for x in results:
            score_rows.append({"Timeframe":x["Timeframe"], "Checks passed":f'{x["Checks passed"]}/{x["Checks total"]}', "Trend":f'{x["Trend score"]}%', "Momentum":f'{x["Momentum score"]}%', "Confirmation":f'{x["Confirmation score"]}%', "Score":f'{x["Score"]}/100', "Weight":f'{x["Weight"]}%', "Contribution (pts)":round(x["Weighted points"], 1)})
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)
        st.caption("Checks passed is a simple count; timeframe scores weight Trend 40%, Momentum 35% and Confirmation 25%. Overall weights are Monthly / Weekly / Daily 35% / 35% / 30%. Identical scores can occur when pass/fail patterns match; the diagnostic panel below shows signal strength without changing the checklist score.")

        st.markdown("**Signal-strength diagnostics (not additional score points)**")
        diagnostic_rows = []
        for frame_name, frame_data, result in [
            ("Grandfather / Monthly", monthly, g),
            ("Father / Weekly", weekly, f),
            ("Son / Daily", daily, s),
        ]:
            current = frame_data.iloc[-1]
            previous = frame_data.iloc[-2]
            rsi_value = current.get("RSI", np.nan)
            if pd.isna(rsi_value):
                rsi_zone = "Unavailable"
            elif rsi_value < 30:
                rsi_zone = "Oversold (<30)"
            elif rsi_value < 40:
                rsi_zone = "Weak (30–40)"
            elif rsi_value < 50:
                rsi_zone = "Below 50 (40–50)"
            elif rsi_value < 55:
                rsi_zone = "Early positive (50–55)"
            elif rsi_value < 60:
                rsi_zone = "Positive (55–60)"
            elif rsi_value < 70:
                rsi_zone = "Strong (60–70)"
            else:
                rsi_zone = "Extended (≥70)"
            ema20 = current.get("EMA20", np.nan)
            ema50 = current.get("EMA50", np.nan)
            close_value = current.get("Close", np.nan)
            distance_ema20 = (close_value / ema20 - 1) * 100 if pd.notna(ema20) and ema20 else np.nan
            distance_ema50 = (close_value / ema50 - 1) * 100 if pd.notna(ema50) and ema50 else np.nan
            hist_now = current.get("MACD_HIST", np.nan)
            hist_prev = previous.get("MACD_HIST", np.nan)
            if pd.isna(hist_now) or pd.isna(hist_prev):
                macd_read = "Unavailable"
            elif hist_now > 0 and hist_now > hist_prev:
                macd_read = "Positive and improving"
            elif hist_now > 0:
                macd_read = "Positive but easing"
            elif hist_now < 0 and hist_now > hist_prev:
                macd_read = "Negative but improving"
            elif hist_now < 0:
                macd_read = "Negative and weakening"
            else:
                macd_read = "At zero"
            diagnostic_rows.append({
                "Timeframe": frame_name,
                "RSI (14)": fmt(rsi_value),
                "RSI zone": rsi_zone,
                "Close vs EMA20": f"{distance_ema20:+.1f}%" if pd.notna(distance_ema20) else "—",
                "Close vs EMA50": f"{distance_ema50:+.1f}%" if pd.notna(distance_ema50) else "—",
                "MACD hist (% of close)": f"{(hist_now / close_value) * 100:+.3f}%" if pd.notna(hist_now) and pd.notna(close_value) and close_value else "—",
                "MACD reading": macd_read,
            })
        st.dataframe(pd.DataFrame(diagnostic_rows), use_container_width=True, hide_index=True)
        st.caption("RSI bands and moving-average distances add context to the binary checklist. A large positive distance can mean price is extended rather than offering a good entry. MACD readings compare the latest completed bar with the previous completed bar. These diagnostics do not alter the GFS score.")

        st.markdown("**Volume confirmation (separate from GFS score)**")
        volume_rows = []
        for frame, data in [("Grandfather / Monthly", monthly), ("Father / Weekly", weekly), ("Son / Daily", daily)]:
            volume_rows.append({"Timeframe": frame, **volume_context(data)})
        st.dataframe(pd.DataFrame(volume_rows), use_container_width=True, hide_index=True)
        st.caption("Relative volume compares the latest completed bar with its 20-period average. Elevated volume is context, not automatically bullish: interpret it with the bar direction, price structure and the next session. Volume is displayed separately and does not change the GFS score.")
        with st.expander("Detailed GFS checklist", expanded=True):
            rows=[]
            for item in results:
                for check, passed in item["Checks"].items():
                    rows.append({"Timeframe":item["Timeframe"], "Category":item["Check groups"][check],
                                 "Condition":check, "Result":"Pass" if passed else "Not met"})
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

        st.subheader("Multi-timeframe support and resistance")
        latest_daily_close = float(daily.Close.iloc[-1])
        st.caption(f"Cross-timeframe comparison reference: latest daily close ₹{latest_daily_close:,.2f}. Monthly and weekly levels provide structural context; daily levels are more relevant to immediate trade planning.")
        level_specs = [
            ("Grandfather / Monthly — major historical levels", monthly, 2, 120),
            ("Father / Weekly — intermediate levels", weekly, 3, 260),
            ("Son / Daily — immediate levels", daily, 5, 250),
        ]
        nearest_level_rows = []
        for level_title, level_df, swing_window, swing_lookback in level_specs:
            st.markdown(f"**{level_title}**")
            timeframe_close = float(level_df.Close.iloc[-1])
            supports, resistances = pivots(level_df, window=swing_window, lookback=swing_lookback, reference_price=latest_daily_close)
            nearby_supports, major_supports = split_nearby_levels(supports, latest_daily_close)
            nearby_resistances, major_resistances = split_nearby_levels(resistances, latest_daily_close)
            nearest_support = supports[0] if supports else None
            nearest_resistance = resistances[0] if resistances else None
            nearest_level_rows.append({
                "Timeframe": level_title.split(" — ")[0],
                "Nearest support (₹)": fmt(nearest_support["price"]) if nearest_support else "Not found",
                "Support distance": f"{(nearest_support['price'] / latest_daily_close - 1) * 100:+.1f}%" if nearest_support else "—",
                "Nearest resistance (₹)": fmt(nearest_resistance["price"]) if nearest_resistance else "Not found",
                "Resistance distance": f"{(nearest_resistance['price'] / latest_daily_close - 1) * 100:+.1f}%" if nearest_resistance else "—",
            })
            def level_rows(points, prefix):
                rows = []
                for i, point in enumerate(points):
                    level_price = float(point["price"])
                    distance_from_daily = (level_price / latest_daily_close - 1) * 100
                    rows.append({
                        "Level": f"{prefix} {i+1}",
                        "Swing date": pd.Timestamp(point["date"]).strftime("%d %b %Y"),
                        "Price (₹)": fmt(level_price),
                        "Distance from daily close": f"{distance_from_daily:+.1f}%"
                    })
                return rows
            st.caption(f"Reference: latest daily close ₹{latest_daily_close:,.2f} (timeframe close: ₹{timeframe_close:,.2f}). Nearby band: within 10% of the daily close.")
            st.caption("Nearby confirmed support — potential trading reference")
            support_table = pd.DataFrame(level_rows(nearby_supports, "Support"))
            if support_table.empty:
                st.info("No confirmed support within 10% below the latest daily close.")
            else:
                st.dataframe(support_table, use_container_width=True, hide_index=True)
            st.caption("Nearby confirmed resistance — potential trading reference")
            resistance_table = pd.DataFrame(level_rows(nearby_resistances, "Resistance"))
            if resistance_table.empty:
                st.info("No confirmed resistance within 10% above the latest daily close.")
            else:
                st.dataframe(resistance_table, use_container_width=True, hide_index=True)
            st.caption("Major historical support — beyond the 10% band")
            major_support_table = pd.DataFrame(level_rows(major_supports, "Major support"))
            if major_support_table.empty:
                st.info("No additional confirmed major support in the selected history.")
            else:
                st.dataframe(major_support_table, use_container_width=True, hide_index=True)
            st.caption("Major historical resistance — beyond the 10% band")
            major_resistance_table = pd.DataFrame(level_rows(major_resistances, "Major resistance"))
            if major_resistance_table.empty:
                st.info("No additional confirmed major resistance in the selected history.")
            else:
                st.dataframe(major_resistance_table, use_container_width=True, hide_index=True)

        st.markdown("**Nearest support / resistance — quick comparison**")
        st.dataframe(pd.DataFrame(nearest_level_rows), use_container_width=True, hide_index=True)
        st.caption("This compact comparison highlights the closest confirmed swing below and above the latest daily close in each timeframe. It is a navigation aid; consult the detailed tables for swing dates and historical context.")

        st.markdown("**Daily Fibonacci retracement**")
        fib, fib_name, fib_status = fib_levels(daily, current_price=latest_daily_close)
        if fib:
            fib_table = pd.DataFrame(
                [{"Retracement": level, "Price (₹)": fmt(value), "Distance from daily close": f"{(value / latest_daily_close - 1) * 100:+.1f}%"} for level, value in fib.items()]
            )
            st.caption(fib_name)
            if fib_status.startswith("Invalidated:"):
                st.warning(fib_status)
            elif (fib_status.startswith("Upswing extended") or fib_status.startswith("Downswing extended")
                  or fib_status.startswith("Stale reference")):
                st.info(fib_status)
            else:
                st.success(fib_status)
            st.dataframe(fib_table, use_container_width=True, hide_index=True)
        else:
            st.info(fib_name)
        st.caption("Monthly swings use two completed bars on each side (up to 120 months of history), weekly swings use three (up to 260 weeks), and daily swings use five (up to 250 sessions). Support and resistance are compared with the latest daily close across all timeframes. Levels within 10% are separated from farther major historical levels; the 10% band is a display convention, not a trading rule. Only confirmed pivots are shown; recent bars may not yet qualify. Yahoo Finance history is downloaded with auto-adjustment enabled; for uploaded CSVs, verify split/dividend adjustments yourself. Fibonacci remains based on the latest confirmed daily swing pair. Levels are estimates, not guaranteed turning points.")

        st.subheader("Fundamental snapshot")
        if info:
            fields = [
                ("Trailing P/E", "trailingPE", "ratio"), ("Forward P/E", "forwardPE", "ratio"),
                ("Price / book", "priceToBook", "ratio"),
                ("Return on equity", "returnOnEquity", "percent"),
                ("Debt / equity (provider-reported)", "debtToEquity", "provider_ratio"),
                ("Profit margin", "profitMargins", "percent"), ("Revenue growth", "revenueGrowth", "percent"),
                ("Earnings growth", "earningsGrowth", "percent"), ("52-week high (₹)", "fiftyTwoWeekHigh", "rupees"),
                ("52-week low (₹)", "fiftyTwoWeekLow", "rupees")]
            fund_rows=[]
            market_cap = info.get("marketCap")
            if market_cap is not None:
                fund_rows.append({"Metric":"Market cap", "Provider value":format_market_cap_inr(market_cap)})
            for label,key,kind in fields:
                value=info.get(key)
                if value is not None:
                    note = ""
                    if kind=="percent":
                        numeric_value = pd.to_numeric(value, errors="coerce")
                        value = f"{numeric_value*100:.2f}%" if pd.notna(numeric_value) else "Unavailable / invalid"
                        if key in ("revenueGrowth", "earningsGrowth") and pd.notna(numeric_value) and abs(numeric_value) >= 1:
                            note = "Unusually large growth figure — verify reporting period, base effect and company filing"
                    elif kind=="ratio": value=f"{value:,.2f}x"
                    elif kind=="provider_ratio": value=f"{value:,.2f} (provider unit; verify definition)"
                    elif kind=="rupees": value=f"₹{value:,.2f}"
                    fund_rows.append({"Metric":label,"Provider value":value,"Data-quality note":note})
            if fund_rows:
                st.dataframe(pd.DataFrame(fund_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No fundamental fields returned by the provider.")
        else:
            st.info("Fundamentals are not included for CSV uploads. Use company annual reports, quarterly results and exchange filings for verification.")
        st.caption("Fundamental data is provider-reported and may be missing or stale. Growth figures of 100% or more in either direction are flagged for manual verification because reporting periods, low-base effects and exceptional items can distort comparisons. Debt/equity is shown in the provider's raw unit because Yahoo field conventions can vary; verify it against company filings. This snapshot is not a substitute for audited statements, quarterly results, shareholding and exchange announcements.")

        st.subheader("Export")
        csv = daily.to_csv(index_label="Date").encode("utf-8")
        st.download_button("Download daily indicators CSV", csv, file_name=f"{symbol}_GFS_V2_daily.csv", mime="text/csv")
        summary_csv = pd.DataFrame([{k:v for k,v in x.items() if k not in ("Checks", "Check groups")} for x in results]).to_csv(index=False).encode("utf-8")
        st.download_button("Download GFS summary CSV", summary_csv, file_name=f"{symbol}_GFS_V2_summary.csv", mime="text/csv")
    except Exception as exc:
        st.error(f"Analysis could not be completed: {exc}")
        st.info("Try another symbol, upload a clean OHLCV CSV, or retry later. The app does not silently replace missing data with estimates.")
else:
    st.markdown("### What Version 2 includes")
    st.markdown("- Monthly / weekly / daily GFS scoring with 35 / 35 / 30 weighting\n- EMA 10/20/50, SMA 50/200, RSI 14 + RSI MA 9, MACD and volume average\n- Candlestick chart, momentum panels, pivot support/resistance and Fibonacci retracement estimates\n- Fundamental snapshot when the selected provider supplies it\n- CSV export and a user-uploaded EOD CSV fallback")
    st.info("Start with an NSE symbol such as INFY, TCS, STLTECH, COCHINSHIP or BDL. The Yahoo feed is free but unofficial; for higher reliability, use a licensed EOD provider or upload its daily OHLCV export.")
