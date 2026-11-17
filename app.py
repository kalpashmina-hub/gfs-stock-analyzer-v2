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

    # Reject impossible OHLC bars instead of silently calculating indicators
    # from corrupted prices. Missing volume is allowed, but negative volume is not.
    invalid_ohlc = (
        (d["High"] < d[["Open", "Close", "Low"]].max(axis=1))
        | (d["Low"] > d[["Open", "Close", "High"]].min(axis=1))
        | (d[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    )
    if invalid_ohlc.any():
        examples = ", ".join(pd.Timestamp(x).strftime("%d %b %Y") for x in d.index[invalid_ohlc][:3])
        raise ValueError(
            f"{int(invalid_ohlc.sum())} OHLC row(s) contain impossible prices "
            f"(for example: {examples}). Correct the source data before analysis."
        )
    invalid_volume = d["Volume"].notna() & (d["Volume"] < 0)
    if invalid_volume.any():
        raise ValueError(
            f"{int(invalid_volume.sum())} row(s) contain negative volume. "
            "Correct the source data before analysis."
        )
    return d[REQUIRED]


def wilder_average(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing seeded with the simple average of the first period."""
    values = series.astype(float).to_numpy()
    output = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < period:
        return pd.Series(output, index=series.index)
    first = int(valid[0])
    seed_end = first + period
    if seed_end > len(values) or not np.isfinite(values[first:seed_end]).all():
        return pd.Series(output, index=series.index)
    output[seed_end - 1] = float(np.mean(values[first:seed_end]))
    for i in range(seed_end, len(values)):
        if np.isfinite(values[i]) and np.isfinite(output[i - 1]):
            output[i] = ((period - 1) * output[i - 1] + values[i]) / period
    return pd.Series(output, index=series.index)


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d.Close
    for n in (10, 20, 50):
        d[f"EMA{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
    for n in (50, 200):
        d[f"SMA{n}"] = c.rolling(n, min_periods=n).mean()
    delta = c.diff()
    # Standard Wilder RSI: SMA seed from the first 14 price changes,
    # followed by recursive Wilder smoothing.
    gain = wilder_average(delta.clip(lower=0), 14)
    loss = wilder_average(-delta.clip(upper=0), 14)
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))
    d.loc[(loss == 0) & (gain > 0), "RSI"] = 100
    d.loc[(gain == 0) & (loss > 0), "RSI"] = 0
    d.loc[(gain == 0) & (loss == 0), "RSI"] = 50
    d["RSI_MA9"] = d.RSI.rolling(9, min_periods=9).mean()
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = e12 - e26
    d["MACD_SIGNAL"] = d.MACD.ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d.MACD - d.MACD_SIGNAL
    # Compare each completed bar with the preceding 20 bars, excluding itself.
    d["VOL_MA20"] = d.Volume.shift(1).rolling(20, min_periods=10).mean()
    return d

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample daily OHLCV data. Labels represent period-end dates."""
    return df.resample(rule).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": lambda values: values.sum(min_count=1),
        }
    ).dropna(subset=["Open", "High", "Low", "Close"])


def completed_daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude today's still-forming daily bar until a conservative EOD buffer passes.

    NSE cash-market close is 15:30 IST. We wait until 16:15 IST before
    treating a same-day Yahoo/CSV bar as a completed EOD observation.
    Historical bars and the developing current-month raw series are unchanged.
    """
    if df.empty:
        return df.copy()
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    today = now.tz_localize(None).normalize()
    latest_date = pd.Timestamp(df.index.max()).normalize()
    if latest_date == today and (now.hour, now.minute) < (16, 15):
        return df.loc[df.index.normalize() < today].copy()
    return df.copy()


def completed_resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Return only fully completed weekly/monthly periods in India time.

    A period whose calendar end date is today is not considered complete
    until after the NSE session has closed and today's daily bar is present.
    Periods ending before today are complete, including holiday-shortened
    weeks/months. This avoids promoting a still-forming Friday/month-end bar
    into the confirmed Father/Grandfather scores.
    """
    bars = resample_ohlcv(df, rule)
    if bars.empty:
        return bars

    now = pd.Timestamp.now(tz="Asia/Kolkata")
    today = now.tz_localize(None).normalize()
    latest_daily_date = pd.Timestamp(df.index.max()).normalize()
    market_close_buffer_passed = (now.hour, now.minute) >= (16, 15)
    period_dates = bars.index.normalize()

    completed = period_dates < today
    if market_close_buffer_passed and latest_daily_date == today:
        completed = completed | (period_dates == today)
    return bars.loc[completed]


def evaluate(df: pd.DataFrame, label: str, weight: int, is_daily=False, fast_ema=20, slow_ema=50):
    # Monthly EMA10/20 needs fewer bars than the weekly/daily EMA20/50
    # framework. Validate actual indicator availability instead of using an
    # unnecessarily high 60-bar minimum for every timeframe.
    minimum_bars = 30 if (fast_ema, slow_ema) == (10, 20) else 60
    if len(df) < minimum_bars:
        return None
    r = df.iloc[-1]
    prev = df.iloc[-2]
    required_values = [
        r.Close, r.get(f"EMA{fast_ema}"), r.get(f"EMA{slow_ema}"),
        r.RSI, r.RSI_MA9, r.MACD_HIST, prev.MACD_HIST,
    ]
    if is_daily:
        required_values.append(r.get("SMA200"))
    if any(pd.isna(value) for value in required_values):
        return None

    close_check = f"Close above EMA{fast_ema}"
    ema_stack_check = f"EMA{fast_ema} above EMA{slow_ema}"
    checks = {
        close_check: bool(pd.notna(r.get(f"EMA{fast_ema}")) and r.Close > r.get(f"EMA{fast_ema}")),
        ema_stack_check: bool(pd.notna(r.get(f"EMA{slow_ema}")) and r.get(f"EMA{fast_ema}") > r.get(f"EMA{slow_ema}")),
        "RSI above 50": bool(pd.notna(r.RSI) and r.RSI > 50),
        "RSI above its 9-period average": bool(pd.notna(r.RSI_MA9) and r.RSI > r.RSI_MA9),
        "MACD histogram positive": bool(pd.notna(r.MACD_HIST) and r.MACD_HIST > 0),
        "MACD histogram improving": bool(pd.notna(prev.MACD_HIST) and pd.notna(r.MACD_HIST) and r.MACD_HIST > prev.MACD_HIST),
    }
    if is_daily:
        checks["Close above SMA200"] = bool(pd.notna(r.SMA200) and r.Close > r.SMA200)
    # Group related signals so momentum checks cannot outweigh price structure.
    # The daily SMA200 check belongs to trend.
    trend_checks = [close_check, ema_stack_check]
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
        is_swing_low = d.Low.iloc[i] == low_window.min()
        is_swing_high = d.High.iloc[i] == high_window.max()
        # Daily OHLC cannot reveal which extreme happened first. Exclude
        # outside bars that qualify as both a swing high and a swing low.
        if is_swing_low and is_swing_high:
            continue
        if is_swing_low:
            points.append({"date": d.index[i], "price": float(d.Low.iloc[i]), "kind": "Low"})
        elif is_swing_high:
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

    # Count other confirmed swing points of the same type within a narrow
    # price band. This is a transparent clustering proxy, not a claim that
    # every visit was a successful support/resistance reaction.
    def add_cluster_context(levels, kind, tolerance_pct=1.5):
        enriched = []
        same_kind = [p for p in swings if p["kind"] == kind]
        for level in levels:
            level_price = float(level["price"])
            cluster_count = sum(
                abs(float(candidate["price"]) / level_price - 1) * 100 <= tolerance_pct
                for candidate in same_kind
            )
            item = dict(level)
            item["cluster_count"] = cluster_count
            item["level_context"] = "Repeated swing area" if cluster_count >= 2 else "Single confirmed swing"
            enriched.append(item)
        return enriched

    supports = add_cluster_context(supports, "Low")
    resistances = add_cluster_context(resistances, "High")
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
    atr = wilder_average(true_range, 14).iloc[-1]
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


def make_chart(df, title, reset_revision=0):
    # Keep the full selected history available; use Plotly range controls to
    # navigate long histories without forcing thousands of candles into view.
    d = df.copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.05, row_heights=[.75,.25])
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Price"), row=1,col=1)
    for col, name in [("EMA10","EMA 10"),("EMA20","EMA 20"),("EMA50","EMA 50"),("SMA200","SMA 200")]:
        if col in d and d[col].notna().any(): fig.add_trace(go.Scatter(x=d.index,y=d[col],name=name,mode="lines"),row=1,col=1)
    fig.add_trace(go.Bar(x=d.index,y=d.Volume,name="Volume"),row=2,col=1)
    range_buttons = [
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="1Y / RESET", step="year", stepmode="backward"),
        dict(count=3, label="3Y", step="year", stepmode="backward"),
        dict(count=5, label="5Y", step="year", stepmode="backward"),
        dict(step="all", label="ALL"),
    ]
    fig.update_layout(
        title=title, height=650,
        xaxis=dict(
            rangeselector=dict(
                buttons=range_buttons,
                font=dict(size=12),
                bgcolor="#f2f4f7",
                activecolor="#d7e6f7",
                bordercolor="#9aa4b2",
                borderwidth=1
            ),
            rangeslider=dict(visible=False)
        ),
        dragmode=False,
        uirevision=f"{title}-default-one-year",
        legend_orientation="h", margin=dict(l=15,r=15,t=75,b=15)
    )
    # Keep the default chart on one year. The in-chart 1Y / RESET control
    # restores this view directly without triggering a Streamlit rerun.
    if len(d.index):
        one_year_range = [d.index[-1] - pd.DateOffset(years=1), d.index[-1]]
        fig.update_layout(xaxis_range=one_year_range, xaxis2_range=one_year_range)
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

# Keep the selected analysis visible across Streamlit reruns (for example,
# when the chart reset button is pressed). The analysis is refreshed only
# when the user presses Run GFS analysis or interacts with a rerun-triggering
# control while an analysis is already active.
if run:
    st.session_state.analysis_requested = True

if st.session_state.get("analysis_requested", False):
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
        analysis_raw = completed_daily_ohlcv(raw)
        if len(analysis_raw) < 220:
            st.error(f"Only {len(analysis_raw)} completed daily rows are available after excluding any still-forming session. Please provide at least 220 completed trading sessions for the daily SMA200 and meaningful multi-timeframe analysis.")
            st.stop()
        daily = indicators(analysis_raw)
        weekly = indicators(completed_resample_ohlcv(analysis_raw, "W-FRI"))
        monthly = indicators(completed_resample_ohlcv(analysis_raw, "ME"))
        results = [evaluate(monthly, "Grandfather / Monthly", 35, fast_ema=10, slow_ema=20), evaluate(weekly, "Father / Weekly", 35), evaluate(daily, "Son / Daily", 30, True)]
        if any(x is None for x in results):
            st.error("Insufficient completed timeframe history or required indicator values. Check the selected history period and data source; the app will not convert missing indicator values into a zero score.")
            st.stop()
        g, f, s = results
        total = round(sum(x["Weighted points"] for x in results))
        st.subheader(info.get("longName", symbol) + (f" ({yf_symbol})" if source.startswith("Yahoo") else ""))
        st.caption(f"Source: {source_label}  |  Last completed daily bar used for GFS: {s['Date']}  |  {len(analysis_raw):,} completed daily sessions")
        if not analysis_raw.empty and pd.Timestamp(raw.index[-1]).normalize() > pd.Timestamp(analysis_raw.index[-1]).normalize():
            st.info(f"Today's developing daily bar ({pd.Timestamp(raw.index[-1]).strftime('%d %b %Y')}) is excluded from confirmed daily indicators, scoring and volume until after 16:15 IST. It remains available for the provisional monthly panel.")
        latest_bar_date = pd.Timestamp(raw.index[-1]).normalize()
        today_for_freshness = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
        data_age_days = (today_for_freshness - latest_bar_date).days
        if data_age_days > 5:
            st.error(f"Stale-data warning: the latest daily bar is {data_age_days} calendar days old ({latest_bar_date.strftime('%d %b %Y')}). Do not use the current GFS score for a fresh entry until the data feed is updated.")
        elif data_age_days < 0:
            st.error(f"Data-date warning: the latest daily bar ({latest_bar_date.strftime('%d %b %Y')}) is in the future. Verify the uploaded file or provider data before relying on this analysis.")
        else:
            st.warning("Data quality: verify the latest bar date and corporate-action adjustments. Yahoo Finance is an unofficial source; CSV data quality depends on the uploaded file.")

        cols = st.columns(4)
        for col, item in zip(cols, [g, f, s, {"Score":total, "State":"Composite GFS", "Timeframe":"Overall"}]):
            with col:
                # State is a category, not a numeric delta. Show it separately
                # so Streamlit does not imply that even a "Weak" state is positive.
                st.metric(item["Timeframe"], f"{item['Score']}%" if item["Timeframe"] != "Overall" else f"{total}/100")
                st.caption(item.get("State", ""))
        st.subheader("GFS decision framework")
        # A high composite checklist score is not enough for the strongest
        # label: require confirmation on both shorter timeframes and at least
        # average daily participation. These are decision gates, not score
        # penalties, so the published GFS score remains unchanged.
        daily_volume_row = volume_context(daily)
        daily_relative_volume = (
            daily.iloc[-1].get("Volume", np.nan) / daily.iloc[-1].get("VOL_MA20", np.nan)
            if pd.notna(daily.iloc[-1].get("VOL_MA20", np.nan))
            and daily.iloc[-1].get("VOL_MA20", np.nan) > 0 else np.nan
        )
        confirmation_ready = (
            f["Confirmation score"] >= 50
            and s["Confirmation score"] >= 50
        )
        volume_ready = pd.notna(daily_relative_volume) and daily_relative_volume >= 0.8
        all_timeframes_bullish = (
            g["State"] == "Bullish" and f["State"] == "Bullish" and s["State"] == "Bullish"
        )
        if all_timeframes_bullish and confirmation_ready and volume_ready:
            verdict = "STRONG BUY CANDIDATE — trend, confirmation and volume gates passed; still check entry, valuation and risk"
            st.success(verdict)
        elif g["State"] == "Bullish" and f["State"] in ("Bullish", "Improving") and s["State"] != "Weak":
            verdict = "BUY ON CONFIRMATION — trend is constructive, but wait for stronger daily confirmation and/or volume"
            st.success(verdict)
            missing_gates = []
            if not confirmation_ready:
                missing_gates.append("weekly and/or daily confirmation score is below 50%")
            if not volume_ready:
                missing_gates.append(
                    f"daily volume is {daily_relative_volume:.2f}x its 20-session average"
                    if pd.notna(daily_relative_volume) else "daily volume confirmation is unavailable"
                )
            if missing_gates:
                st.info("Strong-buy gates not yet met: " + "; ".join(missing_gates) + ".")
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

        # Show the developing current-month candle separately from the confirmed
        # Grandfather score. It is informational only and never enters GFS weights.
        developing_monthly_raw = resample_ohlcv(raw, "ME")
        today_local = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
        latest_raw_date = pd.Timestamp(raw.index[-1]).normalize()
        current_month_end = latest_raw_date.to_period("M").to_timestamp("M")
        confirmed_month_ends = set(monthly.index.normalize())
        current_month_is_unconfirmed = (
            latest_raw_date.year == today_local.year
            and latest_raw_date.month == today_local.month
            and current_month_end not in confirmed_month_ends
        )
        if not developing_monthly_raw.empty and current_month_is_unconfirmed:
            developing_monthly = indicators(developing_monthly_raw)
            provisional = developing_monthly.iloc[-1]
            previous_confirmed = monthly.iloc[-1]
            provisional_fast = provisional.get("EMA10", np.nan)
            provisional_slow = provisional.get("EMA20", np.nan)
            provisional_rsi = provisional.get("RSI", np.nan)
            provisional_rsi_ma = provisional.get("RSI_MA9", np.nan)
            provisional_hist = provisional.get("MACD_HIST", np.nan)
            previous_hist = previous_confirmed.get("MACD_HIST", np.nan)

            def provisional_status(condition):
                return "Yes" if condition else "No"

            recovery_rows = [
                {"Provisional monthly measure": "Current month-to-date close (₹)", "Reading": fmt(provisional.Close),
                 "Interpretation": "Unfinished candle; changes until month-end"},
                {"Provisional monthly measure": "Close above EMA10", "Reading": provisional_status(pd.notna(provisional_fast) and provisional.Close > provisional_fast),
                 "Interpretation": f"EMA10: ₹{provisional_fast:,.2f}" if pd.notna(provisional_fast) else "EMA10 unavailable"},
                {"Provisional monthly measure": "Close above EMA20", "Reading": provisional_status(pd.notna(provisional_slow) and provisional.Close > provisional_slow),
                 "Interpretation": f"EMA20: ₹{provisional_slow:,.2f}" if pd.notna(provisional_slow) else "EMA20 unavailable"},
                {"Provisional monthly measure": "EMA10 above EMA20", "Reading": provisional_status(pd.notna(provisional_fast) and pd.notna(provisional_slow) and provisional_fast > provisional_slow),
                 "Interpretation": "Current partial-month EMA alignment"},
                {"Provisional monthly measure": "RSI (14)", "Reading": fmt(provisional_rsi),
                 "Interpretation": "Above 50" if pd.notna(provisional_rsi) and provisional_rsi > 50 else "Below / equal to 50" if pd.notna(provisional_rsi) else "Unavailable"},
                {"Provisional monthly measure": "RSI above RSI MA9", "Reading": provisional_status(pd.notna(provisional_rsi) and pd.notna(provisional_rsi_ma) and provisional_rsi > provisional_rsi_ma),
                 "Interpretation": f"RSI MA9: {fmt(provisional_rsi_ma)}"},
                {"Provisional monthly measure": "MACD histogram", "Reading": fmt(provisional_hist, 4),
                 "Interpretation": "Improving vs last completed month" if pd.notna(provisional_hist) and pd.notna(previous_hist) and provisional_hist > previous_hist else "Not improving vs last completed month" if pd.notna(provisional_hist) and pd.notna(previous_hist) else "Unavailable"},
            ]
            st.subheader("Provisional monthly recovery — not included in GFS score")
            st.caption(f"Developing month: {today_local.strftime('%B %Y')} | Data through {raw.index[-1].strftime('%d %b %Y')}. These readings use the unfinished monthly candle and can change before month-end.")
            st.dataframe(pd.DataFrame(recovery_rows), use_container_width=True, hide_index=True)
            if (pd.notna(provisional.Close) and pd.notna(provisional_fast)
                    and pd.notna(provisional_slow) and provisional.Close > provisional_fast > provisional_slow):
                st.success("Provisional monthly price structure is improving: close is above EMA10 and EMA10 is above EMA20. Wait for month-end confirmation.")
            elif pd.notna(provisional.Close) and pd.notna(provisional_fast) and provisional.Close > provisional_fast:
                st.info("Provisional monthly close is above EMA10, but the full EMA10/EMA20 bullish structure is not yet confirmed.")
            else:
                st.warning("The developing monthly candle has not yet established a bullish close/EMA10 structure. Reassess as the month progresses.")
            st.caption("The confirmed Grandfather score above continues to use completed monthly candles only. Provisional readings are context for a possible recovery, not a replacement score or a buy signal.")


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
            # Use the agreed EMA pair for each GFS timeframe:
            # Monthly 10/20; Weekly and Daily 20/50.
            fast_period, slow_period = (10, 20) if "Monthly" in frame_name else (20, 50)
            fast_ema = current.get(f"EMA{fast_period}", np.nan)
            slow_ema = current.get(f"EMA{slow_period}", np.nan)
            close_value = current.get("Close", np.nan)
            distance_fast_ema = (close_value / fast_ema - 1) * 100 if pd.notna(fast_ema) and fast_ema else np.nan
            distance_slow_ema = (close_value / slow_ema - 1) * 100 if pd.notna(slow_ema) and slow_ema else np.nan
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
                "EMA pair": f"{fast_period}/{slow_period}",
                "Close vs fast EMA": f"{distance_fast_ema:+.1f}%" if pd.notna(distance_fast_ema) else "—",
                "Close vs slow EMA": f"{distance_slow_ema:+.1f}%" if pd.notna(distance_slow_ema) else "—",
                "MACD hist (% of close)": f"{(hist_now / close_value) * 100:+.3f}%" if pd.notna(hist_now) and pd.notna(close_value) and close_value else "—",
                "MACD reading": macd_read,
            })
        st.dataframe(pd.DataFrame(diagnostic_rows), use_container_width=True, hide_index=True)
        st.caption("EMA pairs follow the GFS framework: Monthly 10/20; Weekly and Daily 20/50. RSI bands and moving-average distances add context to the binary checklist. A large positive distance can mean price is extended rather than offering a good entry. MACD readings compare the latest completed bar with the previous completed bar. These diagnostics do not alter the GFS score.")

        st.markdown("**Volume confirmation (separate from GFS score)**")
        volume_rows = []
        for frame, data in [("Grandfather / Monthly", monthly), ("Father / Weekly", weekly), ("Son / Daily", daily)]:
            volume_rows.append({"Timeframe": frame, **volume_context(data)})
        st.dataframe(pd.DataFrame(volume_rows), use_container_width=True, hide_index=True)
        st.caption("Relative volume compares the latest completed bar with its 20-period average. Elevated volume is context, not automatically bullish: interpret it with the bar direction, price structure and the next session. Volume is displayed separately and does not change the GFS score.")

        # Highlight potentially weak participation or possible distribution as
        # entry-timing context only; neither condition changes the GFS score.
        daily_volume = volume_context(daily)
        daily_relative_volume = daily.iloc[-1].get("Volume", np.nan) / daily.iloc[-1].get("VOL_MA20", np.nan) if pd.notna(daily.iloc[-1].get("VOL_MA20", np.nan)) and daily.iloc[-1].get("VOL_MA20", np.nan) > 0 else np.nan
        daily_bar_direction = daily_volume.get("Price direction")
        if pd.notna(daily_relative_volume) and daily_relative_volume < 0.8 and daily_bar_direction == "Up bar":
            st.info(f"Volume confirmation caution — the latest daily up-bar traded at {daily_relative_volume:.2f}x its 20-session average volume. The price rise has below-average participation; wait for stronger volume or a constructive retest before treating it as a confirmed breakout.")
        elif pd.notna(daily_relative_volume) and daily_relative_volume >= 1.5 and daily_bar_direction == "Down bar":
            st.warning(f"High-volume decline — the latest daily down-bar traded at {daily_relative_volume:.2f}x its 20-session average volume. This may indicate meaningful selling pressure; check the next sessions and nearby support before considering an entry.")

        with st.expander("Detailed GFS checklist", expanded=True):
            rows=[]
            for item in results:
                for check, passed in item["Checks"].items():
                    rows.append({"Timeframe":item["Timeframe"], "Category":item["Check groups"][check],
                                 "Condition":check, "Result":"Pass" if passed else "Not met"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Daily chart")
        st.caption("Default view: 1 year. To restore it, tap the chart's 1Y / RESET control above the plot. Use 6M, 3Y, 5Y or ALL to change the visible history. Chart zoom and pan are disabled to reduce accidental movement on touchscreens.")
        st.plotly_chart(
            make_chart(
                daily,
                f"{symbol} — Daily candles and moving averages"
            ),
            use_container_width=True,
            config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False, "staticPlot": False}
        )
        st.subheader("Momentum dashboard")
        mom = daily.tail(250)
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=mom.index, y=mom["RSI"], name="RSI 14", mode="lines"))
        rsi_fig.add_trace(go.Scatter(x=mom.index, y=mom["RSI_MA9"], name="RSI MA 9", mode="lines"))
        for level in (30, 50, 70):
            rsi_fig.add_hline(y=level, line_dash="dot", annotation_text=str(level))
        rsi_fig.update_yaxes(range=[0, 100], title="RSI")
        rsi_fig.update_layout(title="RSI 14 with 9-period average", height=320, margin=dict(l=15, r=15, t=45, b=15), legend_orientation="h", dragmode=False, uirevision=f"{symbol}-rsi-stable")
        st.plotly_chart(rsi_fig, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})

        macd_fig = go.Figure()
        macd_fig.add_trace(go.Bar(x=mom.index, y=mom["MACD_HIST"], name="Histogram"))
        macd_fig.add_trace(go.Scatter(x=mom.index, y=mom["MACD"], name="MACD", mode="lines"))
        macd_fig.add_trace(go.Scatter(x=mom.index, y=mom["MACD_SIGNAL"], name="Signal", mode="lines"))
        macd_fig.update_layout(title="MACD (12, 26, 9)", height=320, margin=dict(l=15, r=15, t=45, b=15), legend_orientation="h", dragmode=False, uirevision=f"{symbol}-macd-stable")
        st.plotly_chart(macd_fig, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})
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
            # Swing points are sourced from this timeframe, but their current
            # support/resistance role is classified against the latest daily
            # close so every table answers the same practical question:
            # which confirmed levels are below/above the price being assessed?
            # The completed timeframe close remains visible as structural context.
            supports, resistances = pivots(level_df, window=swing_window, lookback=swing_lookback, reference_price=latest_daily_close)
            nearby_supports, major_supports = split_nearby_levels(supports, latest_daily_close)
            nearby_resistances, major_resistances = split_nearby_levels(resistances, latest_daily_close)
            nearest_support = supports[0] if supports else None
            nearest_resistance = resistances[0] if resistances else None
            def compact_level(point):
                if not point:
                    return "Not found"
                distance = (float(point["price"]) / latest_daily_close - 1) * 100
                evidence = point.get("cluster_count", 1)
                evidence_label = "Repeated" if evidence >= 2 else "Single"
                return f"₹{float(point['price']):,.2f} ({distance:+.1f}%) · {evidence_label} ×{evidence}"

            nearest_level_rows.append({
                "Timeframe": level_title.split(" — ")[0],
                "Nearest support": compact_level(nearest_support),
                "Nearest resistance": compact_level(nearest_resistance),
            })
            def level_rows(points, prefix):
                rows = []
                for i, point in enumerate(points):
                    level_price = float(point["price"])
                    distance_from_daily = (level_price / latest_daily_close - 1) * 100
                    evidence = point.get("cluster_count", 1)
                    evidence_label = "Repeated" if evidence >= 2 else "Single"
                    rows.append({
                        "Level": f"{prefix} {i+1}",
                        "Swing date": pd.Timestamp(point["date"]).strftime("%d %b %Y"),
                        "Price / distance": f"₹{level_price:,.2f} ({distance_from_daily:+.1f}%)",
                        "Swing evidence": f"{evidence_label} · {evidence} pivot(s) within ±1.5%"
                    })
                return rows
            st.caption(f"Swing source: {level_title.split(" — ")[0]} | Current-price reference: ₹{latest_daily_close:,.2f} | Last completed timeframe close: ₹{timeframe_close:,.2f}. Support/resistance roles are classified against the current daily price; swing dates and timeframe close provide structural context. Nearby band: within 10% of the daily close.")
            st.caption("Nearby confirmed support — potential trading reference. Repeated swing areas have at least two confirmed same-type pivots within 1.5% of the displayed price; this is a clustering clue, not proof of a defended level.")
            support_table = pd.DataFrame(level_rows(nearby_supports, "Support"))
            if support_table.empty:
                st.info("No confirmed support within 10% below the latest daily close.")
            else:
                st.dataframe(support_table, use_container_width=True, hide_index=True)
            st.caption("Nearby confirmed resistance — potential trading reference. Repeated swing areas have at least two confirmed same-type pivots within 1.5% of the displayed price; this is a clustering clue, not proof of a defended level.")
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
        st.caption("Each row uses confirmed swing points from that timeframe, classified against the latest daily close. Repeated ×2 or more means multiple confirmed same-type pivots fall within 1.5% of the reference price; it is a historical clustering measure, not a tested bounce count or guarantee. A single pivot is still a valid reference. Monthly/weekly levels provide structural context; daily levels are more relevant to immediate planning.")

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
                ("Price / book", "priceToBook", "ratio"), ("EV / EBITDA", "enterpriseToEbitda", "ratio"),
                ("Return on equity", "returnOnEquity", "percent"),
                ("Debt / equity", "debtToEquity", "debt_equity_percent"),
                ("Current ratio", "currentRatio", "ratio"), ("Quick ratio", "quickRatio", "ratio"),
                ("Total debt (₹)", "totalDebt", "inr"), ("Total cash (₹)", "totalCash", "inr"),
                ("Profit margin", "profitMargins", "percent"), ("Revenue growth (provider period)", "revenueGrowth", "percent"),
                ("Earnings growth (provider period)", "earningsGrowth", "percent"), ("52-week high (₹)", "fiftyTwoWeekHigh", "rupees"),
                ("52-week low (₹)", "fiftyTwoWeekLow", "rupees")]
            fund_rows=[]
            market_cap = info.get("marketCap")
            if market_cap is not None:
                fund_rows.append({"Metric":"Market cap", "Provider value":format_market_cap_inr(market_cap), "Data-quality note":""})
            for label,key,kind in fields:
                value=info.get(key)
                if value is not None and pd.notna(pd.to_numeric(value, errors="coerce")):
                    numeric_value = float(pd.to_numeric(value, errors="coerce"))
                    note = ""
                    if kind=="percent":
                        value = f"{numeric_value*100:.2f}%"
                        if key in ("revenueGrowth", "earningsGrowth") and abs(numeric_value) >= 1:
                            note = "Provider period/basis not exposed; verify filings and low-base effects"
                    elif kind=="ratio":
                        value=f"{numeric_value:,.2f}x"
                    elif kind=="debt_equity_percent":
                        # Yahoo's debtToEquity field is expressed as a percentage,
                        # not as a plain ratio: 218.56 means 218.56% = 2.19x.
                        value=f"{numeric_value/100:,.2f}x ({numeric_value:,.2f}%)"
                        note = "Yahoo percentage convention; verify against latest balance sheet and reporting basis"
                    elif kind=="inr":
                        value=format_market_cap_inr(numeric_value)
                        note = "Provider-reported amount; debt and cash are not directly comparable to company net-cash measure without matching definitions, dates and restricted-cash treatment"
                    elif kind=="rupees":
                        value=f"₹{numeric_value:,.2f}"
                    fund_rows.append({"Metric":label,"Provider value":value,"Data-quality note":note})
            if fund_rows:
                fund_table = pd.DataFrame(fund_rows).fillna("")
                st.dataframe(fund_table, use_container_width=True, hide_index=True)
            else:
                st.info("No fundamental fields returned by the provider.")
        else:
            st.info("Fundamentals are not included for CSV uploads. Use company annual reports, quarterly results and exchange filings for verification.")
        st.caption("Fundamental data is provider-reported and may be missing or stale. Growth periods and comparison bases are not reliably exposed in this table; verify revenue and earnings growth against quarterly and annual filings. Large growth percentages may reflect a low base or exceptional items. Debt and cash are provider-reported amounts; do not subtract them to infer company net cash without matching reporting dates, definitions, restricted-cash treatment, leases and investments. Yahoo debt/equity is shown using its percentage convention; verify the figure and reporting basis against the latest balance sheet. This snapshot is not a substitute for audited statements, quarterly results, shareholding and exchange announcements.")

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
