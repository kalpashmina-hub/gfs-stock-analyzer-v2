"""Helpers for preserving provider financial statements as a portable CSV."""
from __future__ import annotations

from typing import Mapping
import pandas as pd


LONG_COLUMNS = [
    "Ticker", "Statement", "Metric", "Period end", "Reported value",
    "Currency", "Provider", "Data basis",
]


def statement_frames_to_long(
    tables: Mapping[str, pd.DataFrame],
    *,
    ticker: str,
    currency: str | None = None,
    provider: str = "Yahoo Finance / yfinance",
) -> pd.DataFrame:
    """Convert raw provider statement tables into a single tidy, loss-minimizing CSV view.

    Provider line-item labels, values and reporting dates are preserved. No units,
    accounting basis, or missing values are inferred.
    """
    rows: list[dict] = []
    for statement, frame in tables.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        for metric, values in frame.iterrows():
            for period, value in values.items():
                if pd.isna(value):
                    continue
                rows.append({
                    "Ticker": ticker,
                    "Statement": str(statement),
                    "Metric": str(metric),
                    "Period end": pd.to_datetime(period, errors="coerce"),
                    "Reported value": value,
                    "Currency": currency or "Not supplied by provider",
                    "Provider": provider,
                    "Data basis": "Provider-reported; consolidated/standalone basis not confirmed",
                })
    if not rows:
        return pd.DataFrame(columns=LONG_COLUMNS)
    result = pd.DataFrame(rows, columns=LONG_COLUMNS)
    result["Period end"] = pd.to_datetime(result["Period end"], errors="coerce")
    return result.sort_values(
        ["Statement", "Metric", "Period end"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)
