"""Non-destructive OHLCV data-quality checks for GFS V2.

This module reports anomalies; it never repairs, drops, adjusts, or rewrites market data.
It is intentionally independent of the GFS indicator/scoring engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def audit_ohlcv(data: pd.DataFrame, *, large_return_pct: float = 25.0) -> dict:
    """Return summary and row-level findings for a daily OHLCV DataFrame.

    A Date/Datetime/Timestamp column or a DatetimeIndex is accepted.
    Calendar gaps are reported, not classified as missing sessions because
    exchange holidays and special sessions require an authoritative calendar.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    frame = data.copy()
    frame.columns = [str(column).strip().title() for column in frame.columns]
    date_col = next(
        (column for column in frame.columns
         if column.lower() in {"date", "datetime", "timestamp"}),
        None,
    )
    if date_col is not None:
        parsed_dates = pd.to_datetime(frame.pop(date_col), errors="coerce")
        frame.index = parsed_dates
    else:
        frame.index = pd.to_datetime(frame.index, errors="coerce")

    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_convert("Asia/Kolkata").tz_localize(None)

    missing_columns = [column for column in REQUIRED_OHLCV if column not in frame.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    original_order = frame.index.copy()
    invalid_dates = frame.index.isna()
    frame = frame.loc[~invalid_dates, list(REQUIRED_OHLCV)].copy()
    for column in REQUIRED_OHLCV:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    duplicate_mask = frame.index.duplicated(keep=False)
    unsorted = not frame.index.is_monotonic_increasing
    frame = frame.sort_index(kind="stable")

    findings = []
    def add(mask, code, severity, detail):
        mask = pd.Series(mask, index=frame.index).fillna(False).astype(bool)
        for date, row in frame.loc[mask].iterrows():
            findings.append({
                "Date": pd.Timestamp(date),
                "Code": code,
                "Severity": severity,
                "Detail": detail(row),
            })

    add(frame[list(REQUIRED_OHLCV)].isna().any(axis=1), "MISSING_OHLCV", "error",
        lambda row: "One or more OHLCV fields are missing or non-numeric")
    finite = np.isfinite(frame[list(REQUIRED_OHLCV)].to_numpy(dtype=float)).all(axis=1)
    add(~finite, "NON_FINITE_VALUE", "error",
        lambda row: "One or more OHLCV fields are infinite")

    prices = frame[["Open", "High", "Low", "Close"]]
    invalid_ohlc = (
        (prices <= 0).any(axis=1)
        | (frame["High"] < prices[["Open", "Close", "Low"]].max(axis=1))
        | (frame["Low"] > prices[["Open", "Close", "High"]].min(axis=1))
    )
    add(invalid_ohlc, "INVALID_OHLC", "error",
        lambda row: "Prices are non-positive or High/Low do not contain Open and Close")
    add(frame["Volume"] < 0, "NEGATIVE_VOLUME", "error",
        lambda row: f"Volume={row['Volume']}")
    add(frame["Volume"].eq(0), "ZERO_VOLUME", "warning",
        lambda row: "Zero volume; verify whether this is a genuine exchange observation")
    flat_zero = frame["Volume"].eq(0) & prices.nunique(axis=1).eq(1)
    add(flat_zero, "ZERO_VOLUME_FLAT_BAR", "warning",
        lambda row: f"All OHLC values are identical ({row['Close']})")

    close_return = frame["Close"].pct_change(fill_method=None) * 100
    large_move = close_return.abs() >= large_return_pct
    add(large_move, "LARGE_DAILY_RETURN", "review",
        lambda row: "Absolute close-to-close move exceeds threshold; verify against market news and corporate actions")

    gap_days = frame.index.to_series().diff().dt.days
    large_gap = gap_days > 4
    add(large_gap, "CALENDAR_GAP_OVER_4D", "info",
        lambda row: "More than four calendar days since prior observation; check exchange calendar before classifying")

    finding_frame = pd.DataFrame(findings, columns=["Date", "Code", "Severity", "Detail"])
    if not finding_frame.empty:
        finding_frame = finding_frame.sort_values(["Date", "Severity", "Code"]).reset_index(drop=True)

    return {
        "rows_input": int(len(data)),
        "rows_with_valid_dates": int(len(frame)),
        "invalid_date_rows": int(invalid_dates.sum()),
        "duplicate_date_rows": int(duplicate_mask.sum()),
        "dates_out_of_order": bool(unsorted),
        "first_date": frame.index.min() if not frame.empty else None,
        "last_date": frame.index.max() if not frame.empty else None,
        "findings": finding_frame,
        "checked_columns": list(REQUIRED_OHLCV),
        "non_destructive": True,
    }


def audit_actions(actions: pd.DataFrame | None) -> dict:
    """Summarize provider-supplied dividend/split event columns when available."""
    if actions is None or actions.empty:
        return {
            "available": False,
            "dividend_events": 0,
            "split_events": 0,
            "note": "No corporate-action event data supplied; absence is not evidence of no events.",
        }
    columns = {str(column).strip().lower(): column for column in actions.columns}
    dividend_col = columns.get("dividends")
    split_col = columns.get("stock splits") or columns.get("stocksplits")
    dividends = pd.to_numeric(actions[dividend_col], errors="coerce") if dividend_col else pd.Series(dtype=float)
    splits = pd.to_numeric(actions[split_col], errors="coerce") if split_col else pd.Series(dtype=float)
    return {
        "available": True,
        "dividend_events": int((dividends.fillna(0) != 0).sum()),
        "split_events": int((splits.fillna(0) != 0).sum()),
        "note": "Event counts reflect only the supplied provider data; verify event dates and terms against exchange/company filings.",
    }
