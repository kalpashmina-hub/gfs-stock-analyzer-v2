"""Source-aware fundamental data intake for GFS Stock Analyzer.

This first-stage module deliberately does not scrape NSE/BSE or third-party sites.
It accepts a normalized CSV exported/prepared from official company/exchange filings
and records provenance so that unsupported or stale values are not silently scored.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

REQUIRED_COLUMNS = {
    "metric", "value", "period_end", "basis", "unit",
    "source_name", "source_url", "filing_date",
}
ALLOWED_BASIS = {"consolidated", "standalone"}
PRIMARY_SOURCE_HINTS = ("nse", "bse", "company", "annual report", "exchange")


@dataclass
class FundamentalIntake:
    data: pd.DataFrame
    errors: list[str]
    warnings: list[str]


def _as_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True).dt.normalize()


def validate_fundamental_upload(raw: pd.DataFrame) -> FundamentalIntake:
    """Validate a normalized filing CSV; preserve source and period per metric."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        return FundamentalIntake(pd.DataFrame(), [f"Missing columns: {', '.join(missing)}"], [])

    df["metric"] = df["metric"].astype(str).str.strip().str.lower()
    df["basis"] = df["basis"].astype(str).str.strip().str.lower()
    df["unit"] = df["unit"].astype(str).str.strip()
    df["source_name"] = df["source_name"].astype(str).str.strip()
    df["source_url"] = df["source_url"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period_end"] = _as_date(df["period_end"])
    df["filing_date"] = _as_date(df["filing_date"])

    errors: list[str] = []
    warnings: list[str] = []
    if df["metric"].eq("").any() or df["metric"].isna().any():
        errors.append("Every row must identify a metric.")
    if df["value"].isna().any():
        errors.append("Some values are blank or non-numeric.")
    if df["period_end"].isna().any():
        errors.append("Some period_end dates are invalid.")
    if df["filing_date"].isna().any():
        errors.append("Some filing_date dates are invalid.")
    if (~df["basis"].isin(ALLOWED_BASIS)).any():
        errors.append("basis must be consolidated or standalone; do not mix them.")
    if df["source_name"].eq("").any() or df["source_url"].eq("").any():
        errors.append("Every value must include its source name and source URL.")

    # A metric/period/basis tuple must not silently contain duplicate competing values.
    duplicate_groups = df.groupby(["metric", "period_end", "basis", "unit"], dropna=False)
    conflicting_duplicates = any(
        group["value"].dropna().nunique() > 1
        for _, group in duplicate_groups
        if len(group) > 1
    )
    if conflicting_duplicates:
        warnings.append(
            "Conflicting values exist for the same metric, period, basis and unit. "
            "Check original versus revised filings and source definitions; do not select "
            "a value automatically."
        )
    elif any(len(group) > 1 for _, group in duplicate_groups):
        warnings.append(
            "Multiple sources report the same metric, period, basis and unit with matching "
            "values. This is agreement in the uploaded file, not proof of independent "
            "verification; inspect each source link."
        )
    non_primary = ~df["source_name"].str.lower().apply(
        lambda x: any(hint in x for hint in PRIMARY_SOURCE_HINTS)
    )
    if non_primary.any():
        warnings.append(
            "Some rows are secondary-source data. Cross-check them against the original "
            "company or exchange filing before treating them as authoritative."
        )
    if df["filing_date"].gt(pd.Timestamp.today().normalize()).any():
        warnings.append("A filing date is in the future; check the source data.")
    df = df.sort_values(["metric", "period_end", "basis"], ascending=[True, False, True])
    return FundamentalIntake(df, errors, warnings)


def compare_sources(data: pd.DataFrame) -> pd.DataFrame:
    """Show disagreements across sources for the same metric, period, basis and unit."""
    if data.empty:
        return pd.DataFrame()
    keys = ["metric", "period_end", "basis", "unit"]
    rows = []
    for key, group in data.groupby(keys, dropna=False):
        values = group["value"].dropna()
        if len(values) < 2:
            continue
        low, high = float(values.min()), float(values.max())
        spread_pct = ((high - low) / max(abs(low), abs(high), 1e-12)) * 100
        rows.append({
            "Metric": key[0],
            "Period end": key[1].strftime("%d %b %Y") if pd.notna(key[1]) else "—",
            "Basis": key[2],
            "Unit": key[3],
            "Sources": ", ".join(group["source_name"].astype(str).unique()),
            "Low": low,
            "High": high,
            "Difference (%)": round(spread_pct, 2),
            "Review required": "Yes" if spread_pct > 1 else "No",
        })
    return pd.DataFrame(rows)


def latest_filing_status(data: pd.DataFrame) -> pd.DataFrame:
    """Return latest filing and financial-period metadata, without guessing freshness."""
    if data.empty:
        return pd.DataFrame()
    out = data.groupby(["basis"], dropna=False).agg(
        Latest_period=("period_end", "max"),
        Latest_filing=("filing_date", "max"),
        Metrics=("metric", "nunique"),
        Sources=("source_name", lambda x: ", ".join(sorted(set(map(str, x))))),
    ).reset_index()
    return out
