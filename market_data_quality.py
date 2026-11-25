"""Non-destructive market-data quality checks for GFS V2.

Audits daily OHLCV observations and optional exchange-calendar, corporate-action,
and provider metadata. This module never repairs, drops, adjusts, or rewrites data
and is deliberately independent of the GFS indicator/scoring engine.
"""
from __future__ import annotations

from typing import Iterable, Mapping
import numpy as np
import pandas as pd

REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")
DATE_NAMES = {"date", "datetime", "timestamp"}
FINDING_COLUMNS = ["Date", "Row", "Code", "Severity", "Detail"]


def _date_index(values) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    return idx.normalize()


def audit_ohlcv(
    data: pd.DataFrame,
    *,
    large_return_pct: float = 25.0,
    exchange_calendar: Iterable | pd.DataFrame | None = None,
    exchange: str = "NSE",
    expected_interval_days: int = 4,
) -> dict:
    """Audit OHLCV data; optional calendar is an authoritative caller-supplied list.

    exchange_calendar may be an iterable of non-trading dates or a DataFrame with
    a Date/Datetime/Timestamp column/index and optional Status/Reason column.
    Without it, gaps are informational only and are never called missing sessions.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if large_return_pct <= 0:
        raise ValueError("large_return_pct must be positive")
    if expected_interval_days < 1:
        raise ValueError("expected_interval_days must be at least 1")

    frame = data.copy()
    frame.columns = [str(c).strip().title() for c in frame.columns]
    date_col = next((c for c in frame.columns if c.lower() in DATE_NAMES), None)
    raw_dates = frame.pop(date_col) if date_col else frame.index
    parsed = pd.to_datetime(raw_dates, errors="coerce")
    dates = pd.DatetimeIndex(parsed)
    if dates.tz is not None:
        dates = dates.tz_convert("Asia/Kolkata").tz_localize(None)
    dates = dates.normalize()

    missing_columns = [c for c in REQUIRED_OHLCV if c not in frame.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    invalid_date_mask = dates.isna()
    valid_positions = np.flatnonzero(~invalid_date_mask)
    valid = frame.iloc[valid_positions][list(REQUIRED_OHLCV)].copy()
    valid.index = dates[~invalid_date_mask]
    valid.insert(0, "__source_row__", valid_positions)
    for col in REQUIRED_OHLCV:
        valid[col] = pd.to_numeric(valid[col], errors="coerce")

    original_valid_dates = dates[~invalid_date_mask]
    duplicate_mask = original_valid_dates.duplicated(keep=False)
    out_of_order = not original_valid_dates.is_monotonic_increasing
    duplicate_rows = int(duplicate_mask.sum())

    findings = []

    def add(date, row_number, code, severity, detail):
        findings.append({"Date": date, "Row": int(row_number), "Code": code,
                         "Severity": severity, "Detail": str(detail)})

    for pos in np.flatnonzero(invalid_date_mask):
        add(pd.NaT, pos, "INVALID_DATE", "error", "Date is missing or cannot be parsed")

    for pos, is_dup in zip(valid["__source_row__"].to_numpy(), duplicate_mask):
        if is_dup:
            add(dates[pos], pos, "DUPLICATE_DATE", "error",
                "More than one observation exists for this date; no row was selected")
    if out_of_order:
        add(pd.NaT, -1, "OUT_OF_ORDER_DATES", "warning",
            "Input observations are not in chronological order")

    # Stable sort is used only for sequential diagnostics; source rows remain identified.
    ordered = valid.sort_index(kind="stable")
    for idx, row in ordered.iterrows():
        source_row = row["__source_row__"]
        values = row[list(REQUIRED_OHLCV)]
        missing_fields = values.index[values.isna()].tolist()
        missing_price_fields = [field for field in missing_fields if field in ("Open", "High", "Low", "Close")]
        if missing_price_fields:
            add(idx, source_row, "MISSING_OR_NON_NUMERIC", "error",
                "Missing or non-numeric price field(s): " + ", ".join(missing_price_fields))
        if "Volume" in missing_fields:
            # Missing volume is permitted by clean_ohlcv and does not invalidate
            # price-based indicators. Volume-dependent confirmation remains unavailable.
            add(idx, source_row, "MISSING_VOLUME", "warning",
                "Volume is missing or non-numeric; price analysis may continue, but volume confirmation is unavailable")
        numeric_values = values.dropna().to_numpy(dtype=float)
        if np.isinf(numeric_values).any():
            add(idx, source_row, "NON_FINITE_VALUE", "error",
                "One or more OHLCV fields contain positive or negative infinity")
        prices = row[["Open", "High", "Low", "Close"]]
        if prices.notna().all():
            invalid = bool((prices <= 0).any() or row["High"] < prices.max() or
                           row["Low"] > prices.min())
            if invalid:
                add(idx, source_row, "INVALID_OHLC", "error",
                    "Prices are non-positive or High/Low do not contain Open and Close")
        if pd.notna(row["Volume"]):
            if row["Volume"] < 0:
                add(idx, source_row, "NEGATIVE_VOLUME", "error",
                    f"Volume={row['Volume']}")
            elif row["Volume"] == 0:
                add(idx, source_row, "ZERO_VOLUME", "warning",
                    "Zero volume; verify whether this is a genuine exchange observation")
                if prices.notna().all() and prices.nunique() == 1:
                    add(idx, source_row, "ZERO_VOLUME_FLAT_BAR", "warning",
                        f"All OHLC values are identical ({row['Close']})")

    # Sequential price checks are performed only when dates are unique.
    if not ordered.empty and not ordered.index.has_duplicates:
        returns = ordered["Close"].pct_change(fill_method=None) * 100
        for idx, pct in returns.items():
            if pd.notna(pct) and abs(pct) >= large_return_pct:
                row = ordered.loc[idx]
                add(idx, row["__source_row__"], "LARGE_DAILY_RETURN", "review",
                    f"Close-to-close move is {pct:.2f}%; verify news, actions and source prices")
        gaps = ordered.index.to_series().diff().dt.days
        for idx, gap in gaps.items():
            if pd.notna(gap) and gap > expected_interval_days:
                row = ordered.loc[idx]
                add(idx, row["__source_row__"], "CALENDAR_GAP", "info",
                    f"{int(gap)} calendar days since previous observation; not proof of a missing session")

    calendar_dates = set()
    calendar_reasons = {}
    if exchange_calendar is not None:
        if isinstance(exchange_calendar, pd.DataFrame):
            cal = exchange_calendar.copy()
            cal.columns = [str(c).strip().title() for c in cal.columns]
            cal_date_col = next((c for c in cal.columns if c.lower() in DATE_NAMES), None)
            raw_cal_dates = cal.pop(cal_date_col) if cal_date_col else cal.index
            cal_dates = _date_index(raw_cal_dates)
            reason_col = next((c for c in ("Reason", "Status", "Description") if c in cal.columns), None)
            reasons = cal[reason_col].astype(str).tolist() if reason_col else ["exchange non-trading day"] * len(cal)
        else:
            cal_dates = _date_index(list(exchange_calendar))
            reasons = ["exchange non-trading day"] * len(cal_dates)
        for d, reason in zip(cal_dates, reasons):
            if pd.notna(d):
                calendar_dates.add(d)
                calendar_reasons[d] = reason

        for idx, row in ordered.iterrows():
            if idx in calendar_dates:
                add(idx, row["__source_row__"], "OBSERVATION_ON_NON_TRADING_DATE", "error",
                    f"{exchange} calendar marks this date non-trading: {calendar_reasons[idx]}")
        # A gap is classified as a potential missing session only if caller supplied
        # an explicit session calendar, not merely a holiday list.
        if isinstance(exchange_calendar, pd.DataFrame) and "IsTradingSession" in exchange_calendar.columns:
            cal = exchange_calendar.copy()
            cal.columns = [str(c).strip().title() for c in cal.columns]
            dc = next((c for c in cal.columns if c.lower() in DATE_NAMES), None)
            cdates = _date_index(cal.pop(dc) if dc else cal.index)
            sessions = {d for d, flag in zip(cdates, cal["Istradingsession"]) if bool(flag)}
            observed = set(ordered.index)
            for d in sorted(sessions - observed):
                add(d, -1, "MISSING_TRADING_SESSION", "error",
                    f"{exchange} calendar identifies a trading session with no OHLCV observation")

    result_findings = pd.DataFrame(findings, columns=FINDING_COLUMNS)
    if not result_findings.empty:
        result_findings = result_findings.sort_values(
            ["Date", "Row", "Code"], na_position="last").reset_index(drop=True)

    return {
        "rows_input": int(len(data)),
        "rows_with_valid_dates": int(len(valid)),
        "invalid_date_rows": int(invalid_date_mask.sum()),
        "duplicate_date_rows": duplicate_rows,
        "dates_out_of_order": bool(out_of_order),
        "first_date": ordered.index.min() if not ordered.empty else None,
        "last_date": ordered.index.max() if not ordered.empty else None,
        "findings": result_findings,
        "checked_columns": list(REQUIRED_OHLCV),
        "exchange": exchange,
        "calendar_supplied": exchange_calendar is not None,
        "non_destructive": True,
        "source_rows_preserved": True,
    }


def audit_actions(actions: pd.DataFrame | None) -> dict:
    """Summarize supplied provider dividend/split columns without inferring absence."""
    if actions is None or actions.empty:
        return {"available": False, "dividend_events": 0, "split_events": 0,
                "note": "No corporate-action event data supplied; absence is not evidence of no events."}
    columns = {str(c).strip().lower(): c for c in actions.columns}
    dividend_col = columns.get("dividends")
    split_col = columns.get("stock splits") or columns.get("stocksplits")
    dividends = pd.to_numeric(actions[dividend_col], errors="coerce") if dividend_col else pd.Series(dtype=float)
    splits = pd.to_numeric(actions[split_col], errors="coerce") if split_col else pd.Series(dtype=float)
    return {
        "available": bool(dividend_col or split_col),
        "dividend_events": int((dividends.fillna(0) != 0).sum()),
        "split_events": int((splits.fillna(0) != 0).sum()),
        "note": "Counts reflect only supplied provider data; verify dates and terms against exchange/company filings.",
    }


def audit_price_adjustment_metadata(metadata: Mapping | None) -> dict:
    """Describe price adjustment provenance; never guess undocumented settings."""
    metadata = dict(metadata or {})
    keys = ("provider", "downloaded_at", "interval", "auto_adjust", "back_adjust",
            "actions", "repair", "currency", "exchange", "timezone")
    captured = {key: metadata[key] for key in keys if key in metadata}
    missing = [key for key in ("provider", "interval", "auto_adjust", "actions") if key not in metadata]
    return {
        "metadata": captured,
        "missing_core_fields": missing,
        "adjustment_policy_known": "auto_adjust" in metadata and metadata.get("auto_adjust") is not None,
        "corporate_action_events_requested": metadata.get("actions") is True,
        "note": ("Adjustment settings are partly undocumented; do not infer raw/adjusted status."
                 if missing else "Record adjustment settings with the dataset; this does not independently verify provider behaviour."),
    }
