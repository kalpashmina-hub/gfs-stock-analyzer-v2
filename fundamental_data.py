"""Source-aware fundamental data intake for GFS Stock Analyzer.

This first-stage module deliberately does not scrape NSE/BSE or third-party sites.
It accepts a normalized CSV exported/prepared from official company/exchange filings
and records provenance so that unsupported or stale values are not silently scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
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



def classify_source(source_name: str) -> str:
    name = str(source_name).strip().lower()
    if "nse" in name or "bse" in name or "exchange" in name:
        return "Primary — stock exchange filing"
    if "company" in name or "tcs" in name or "annual report" in name or "investor relations" in name:
        return "Primary — company disclosure"
    return "Secondary — cross-check only"


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
    invalid_urls = df["source_url"].apply(
        lambda value: urlparse(str(value)).scheme not in {"http", "https"}
        or not bool(urlparse(str(value)).netloc)
    )
    if invalid_urls.any():
        errors.append("Every source_url must be a complete HTTP or HTTPS link.")
    df["source_class"] = df["source_name"].apply(classify_source)

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
            "Review required": "Yes" if spread_pct > 0.01 else "No",
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

def period_growth_table(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate like-for-like quarter growth only for matching metric/unit/basis rows."""
    if data.empty:
        return pd.DataFrame()
    eligible = {
        "revenue_from_operations", "profit_before_tax",
        "profit_after_tax_before_non_controlling_interests",
        "net_profit_attributable_to_owners",
        "net_cash_from_operating_activities", "free_cash_flow",
    }
    d = data[data["metric"].isin(eligible)].copy()
    if d.empty:
        return pd.DataFrame()
    # Do not calculate growth when multiple filings compete for the same observation.
    keys = ["metric", "period_end", "basis", "unit"]
    d = d.drop_duplicates(keys, keep=False)
    rows = []
    for (metric, basis, unit), group in d.groupby(["metric", "basis", "unit"]):
        values = {pd.Timestamp(r.period_end).strftime("%Y-%m-%d"): float(r.value)
                  for r in group.itertuples()}
        periods = sorted(values)
        for current in periods:
            current_date = pd.Timestamp(current)
            comparisons = [
                ("YoY", (current_date - pd.DateOffset(years=1)).strftime("%Y-%m-%d")),
                ("QoQ", (current_date - pd.DateOffset(months=3)).strftime("%Y-%m-%d")),
            ]
            for label, prior in comparisons:
                if prior in values and values[prior] != 0:
                    rows.append({
                        "Metric": metric,
                        "Current period": current_date.strftime("%d %b %Y"),
                        "Comparison": label,
                        "Prior period": pd.Timestamp(prior).strftime("%d %b %Y"),
                        "Basis": basis,
                        "Unit": unit,
                        "Growth (%)": round((values[current] / values[prior] - 1) * 100, 2),
                    })
    return pd.DataFrame(rows)

