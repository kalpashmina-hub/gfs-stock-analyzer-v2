"""Source-aware fundamental filing intake for GFS Stock Analyzer.

The module accepts normalized filing CSVs; it does not scrape or authenticate
external documents. Source classification is a provenance hint based on the
URL host, never proof that a linked filing was reviewed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from urllib.parse import urlparse

import pandas as pd

REQUIRED_COLUMNS = {
    "metric", "value", "period_end", "basis", "unit",
    "source_name", "source_url", "filing_date",
}
ALLOWED_BASIS = {"consolidated", "standalone"}
PRIMARY_HOSTS = {
    "nseindia.com": "Primary — NSE exchange domain",
    "bseindia.com": "Primary — BSE exchange domain",
    "tcs.com": "Primary — company disclosure domain",
}
SECONDARY_HOSTS = {
    "screener.in", "trendlyne.com", "tijorifinance.com",
    "moneycontrol.com", "tickertape.in",
}


@dataclass
class FundamentalIntake:
    data: pd.DataFrame
    errors: list[str]
    warnings: list[str]


def _as_date(series: pd.Series) -> pd.Series:
    """Parse common Indian filing dates and ISO dates without guessing invalids."""
    return pd.to_datetime(series, errors="coerce", dayfirst=True).dt.normalize()


def _host_matches(host: str, domain: str) -> bool:
    host = host.lower().rstrip(".")
    return host == domain or host.endswith("." + domain)


def classify_source(source_name: str, source_url: str | None = None) -> str:
    """Classify by exact URL-domain boundaries; labels alone are not authority."""
    host = (urlparse(str(source_url or "")).hostname or "").lower().rstrip(".")
    for domain, label in PRIMARY_HOSTS.items():
        if _host_matches(host, domain):
            return label
    if any(_host_matches(host, domain) for domain in SECONDARY_HOSTS):
        return "Secondary — cross-check only"
    return "Unclassified — verify source"


def _duplicate_observations(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["metric", "period_end", "basis", "unit"]
    if data.empty:
        return data.copy()
    return data[data.duplicated(keys, keep=False)].copy()


def validate_fundamental_upload(raw: pd.DataFrame) -> FundamentalIntake:
    """Validate normalized filing rows while preserving provenance and accounting basis."""
    empty = pd.DataFrame()
    if raw is None or raw.empty:
        return FundamentalIntake(empty, ["CSV contains no data rows."], [])

    df = raw.copy()
    normalized_columns = [str(c).strip().lower() for c in df.columns]
    if len(normalized_columns) != len(set(normalized_columns)):
        return FundamentalIntake(empty, ["Column names become duplicated after normalization."], [])
    df.columns = normalized_columns

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        return FundamentalIntake(empty, [f"Missing columns: {', '.join(missing)}"], [])

    errors: list[str] = []
    warnings: list[str] = []

    # Treat whitespace-only strings as missing rather than valid metadata.
    for column in ("metric", "basis", "unit", "source_name", "source_url"):
        df[column] = df[column].astype("string").str.strip()

    df["metric"] = df["metric"].str.lower()
    df["basis"] = df["basis"].str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period_end"] = _as_date(df["period_end"])
    df["filing_date"] = _as_date(df["filing_date"])

    if df["metric"].isna().any() or df["metric"].eq("").any():
        errors.append("Every row must identify a metric.")
    if df["value"].isna().any():
        errors.append("Some values are blank or non-numeric.")
    elif not df["value"].map(lambda value: math.isfinite(float(value))).all():
        errors.append("Values must be finite numbers; NaN and infinity are not allowed.")
    if df["period_end"].isna().any():
        errors.append("Some period_end dates are invalid.")
    if df["filing_date"].isna().any():
        errors.append("Some filing_date dates are invalid.")
    if (~df["basis"].isin(ALLOWED_BASIS)).any():
        errors.append("basis must be consolidated or standalone; do not mix them.")
    for column, label in (("unit", "unit"), ("source_name", "source name"), ("source_url", "source URL")):
        if df[column].isna().any() or df[column].eq("").any():
            errors.append(f"Every value must include a non-empty {label}.")

    def valid_http_url(value: object) -> bool:
        parsed = urlparse(str(value))
        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
        )

    invalid_urls = ~df["source_url"].map(valid_http_url)
    if invalid_urls.any():
        errors.append("Every source_url must be a complete HTTP or HTTPS link without embedded credentials.")

    today = pd.Timestamp(date.today())
    if df["filing_date"].notna().any() and df["filing_date"].gt(today).any():
        errors.append("A filing date is in the future; verify the source data.")
    if df["period_end"].notna().any() and df["period_end"].gt(today).any():
        errors.append("A financial period_end is in the future; verify the source data.")
    date_order = df["period_end"].notna() & df["filing_date"].notna() & df["filing_date"].lt(df["period_end"])
    if date_order.any():
        errors.append("A filing_date precedes its period_end; verify the reporting dates.")

    df["source_class"] = df.apply(
        lambda row: classify_source(row["source_name"], row["source_url"]), axis=1
    )
    secondary = df["source_class"].eq("Secondary — cross-check only")
    if secondary.any():
        names = sorted(df.loc[secondary, "source_name"].astype(str).unique())
        warnings.append(
            "Secondary-source rows detected: " + "; ".join(names)
            + ". Use these only as cross-checks, not primary filing evidence."
        )
    unclassified = df["source_class"].eq("Unclassified — verify source")
    if unclassified.any():
        names = sorted(df.loc[unclassified, "source_name"].astype(str).unique())
        warnings.append(
            "Unclassified source domains detected: " + "; ".join(names)
            + ". Verify the publisher independently; a descriptive source label is not authentication."
        )

    duplicates = _duplicate_observations(df)
    if not duplicates.empty:
        keys = ["metric", "period_end", "basis", "unit"]
        conflicting = []
        matching = []
        for key, group in duplicates.groupby(keys, dropna=False):
            if group["value"].nunique(dropna=True) > 1:
                conflicting.append(key)
            else:
                matching.append(key)
        if conflicting:
            warnings.append(
                f"{len(conflicting)} metric/period/basis/unit observation group(s) contain competing values. "
                "No value is selected automatically; inspect original, revised and comparative filings."
            )
        if matching:
            warnings.append(
                f"{len(matching)} observation group(s) have duplicate matching values. "
                "Agreement in an uploaded file is not proof of independent verification."
            )

    if errors:
        # Return normalized rows for diagnostics, but the UI must not treat them as validated.
        return FundamentalIntake(df, errors, warnings)

    return FundamentalIntake(
        df.sort_values(["metric", "period_end", "basis"], ascending=[True, False, True]).reset_index(drop=True),
        errors,
        warnings,
    )



def _publisher_domain(source_url: object) -> str:
    """Normalize publisher domains so subdomains count as one publisher."""
    host = (urlparse(str(source_url or "")).hostname or "").lower().rstrip(".")
    for domain in (*PRIMARY_HOSTS.keys(), *SECONDARY_HOSTS):
        if _host_matches(host, domain):
            return domain
    return host


def source_reconciliation_coverage(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize exact metric/period/basis/unit coverage across distinct publishers."""
    columns = ["Comparable metric-period groups", "Single-publisher groups", "Same-publisher duplicate groups", "Total metric-period groups"]
    if data.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame()
    keys = ["metric", "period_end", "basis", "unit"]
    rows = []
    for key, group in data.groupby(keys, dropna=False):
        urls = group["source_url"] if "source_url" in group else pd.Series(index=group.index, dtype="object")
        publishers = sorted({domain for domain in (_publisher_domain(url) for url in urls) if domain})
        count = len(group)
        secondary_present = (
            group["source_class"].astype(str).str.startswith("Secondary").any()
            if "source_class" in group.columns else False
        )
        if len(publishers) >= 2 and secondary_present:
            status = "Comparable publishers — includes secondary source; verify original filing"
        elif len(publishers) >= 2:
            status = "Comparable publishers — review values and provenance"
        elif count > 1:
            status = "Multiple records, same publisher — not independent"
        else:
            status = "Single publisher — no corroboration"
        rows.append({"Metric": key[0], "Period end": pd.Timestamp(key[1]).strftime("%d %b %Y") if pd.notna(key[1]) else "—", "Basis": key[2], "Unit": key[3], "Source records": count, "Publisher count": len(publishers), "Publishers": ", ".join(publishers) if publishers else "Unknown", "Status": status})
    details = pd.DataFrame(rows)
    summary = pd.DataFrame([{"Comparable metric-period groups": int(details["Status"].str.startswith("Comparable publishers").sum()), "Single-publisher groups": int(details["Status"].str.startswith("Single publisher").sum()), "Same-publisher duplicate groups": int(details["Status"].str.startswith("Multiple records").sum()), "Total metric-period groups": len(details)}], columns=columns)
    return summary, details[details["Publisher count"] > 1].reset_index(drop=True)

def compare_sources(data: pd.DataFrame) -> pd.DataFrame:
    """Show differences across multiple rows for identical metric/period/basis/unit."""
    if data.empty:
        return pd.DataFrame()
    keys = ["metric", "period_end", "basis", "unit"]
    rows = []
    for key, group in data.groupby(keys, dropna=False):
        if len(group) < 2:
            continue
        urls = group["source_url"] if "source_url" in group else pd.Series(index=group.index, dtype="object")
        publishers = {_publisher_domain(url) for url in urls}
        publishers.discard("")
        if len(publishers) < 2:
            continue
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
            "Source classes": ", ".join(sorted(set(group.get("source_class", pd.Series(["Unclassified"] * len(group), index=group.index)).astype(str)))),
            "Low": low,
            "High": high,
            "Difference (%)": round(spread_pct, 2),
            "Review required": "Yes" if spread_pct > 0.01 else "No",
        })
    return pd.DataFrame(rows)



def fundamental_quality_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build descriptive quality and balance-sheet views from reported observations.

    This deliberately does not assign a score or infer missing debt/EBITDA values.
    Only validated primary-source rows are used for the displayed snapshot.
    """
    operating_columns = ["Metric", "Period", "Reported value", "Unit", "Interpretation note"]
    balance_columns = ["Metric", "Period", "Value", "Unit", "Interpretation note"]
    missing_columns = ["Metric not available", "Why it matters"]

    if data.empty:
        return (
            pd.DataFrame(columns=operating_columns),
            pd.DataFrame(columns=balance_columns),
            pd.DataFrame(columns=missing_columns),
        )

    d = data.copy()
    if "source_class" in d.columns:
        d = d[d["source_class"].astype(str).str.startswith("Primary")].copy()
    if d.empty:
        return (
            pd.DataFrame(columns=operating_columns),
            pd.DataFrame(columns=balance_columns),
            pd.DataFrame(columns=missing_columns),
        )

    def latest(metric: str, unit_contains: str | None = None) -> pd.Series | None:
        rows = d[(d["metric"] == metric) & (d["basis"] == "consolidated")].copy()
        if unit_contains:
            rows = rows[rows["unit"].astype(str).str.contains(unit_contains, case=False, regex=False)]
        if rows.empty:
            return None
        max_period = rows["period_end"].max()
        rows = rows[rows["period_end"] == max_period]
        # If multiple primary rows disagree, do not choose one silently.
        if rows["value"].nunique(dropna=True) != 1:
            return None
        return rows.iloc[0]

    operating_specs = [
        ("operating_margin", "IFRS; quarter", "Reported quarterly operating margin; one quarter is not a trend."),
        ("net_margin", "IFRS; quarter", "Reported quarterly net margin; confirm the issuer's definition before peer comparison."),
        ("net_cash_from_operating_activities", "IFRS; quarter", "Quarterly operating cash flow; assess across several periods."),
        ("free_cash_flow", "IFRS; quarter", "Reported quarterly free cash flow; verify the issuer's capex definition."),
        ("capital_expenditure", "IFRS; quarter", "Quarterly capital expenditure; compare with operating cash flow over time."),
        ("operating_cash_flow_to_sales", "IFRS; quarter", "Issuer-reported operating cash flow to sales ratio."),
        ("operating_cash_flow_to_net_profit", "IFRS; quarter", "Issuer-reported cash flow to net profit ratio; above 100% in one quarter is not a standalone quality verdict."),
    ]
    operating_rows = []
    for metric, unit_hint, note in operating_specs:
        row = latest(metric, unit_hint)
        if row is not None:
            operating_rows.append({
                "Metric": metric.replace("_", " ").title(),
                "Period": pd.Timestamp(row["period_end"]).strftime("%d %b %Y"),
                "Reported value": float(row["value"]),
                "Unit": row["unit"],
                "Interpretation note": note,
            })

    balance_specs = [
        ("total_cash_and_investments", "Cash and investments"),
        ("total_assets", "Total assets"),
        ("shareholders_funds", "Shareholders' funds"),
        ("current_liabilities", "Current liabilities"),
        ("non_current_liabilities", "Non-current liabilities"),
    ]
    balance_values: dict[str, pd.Series] = {}
    for metric, _ in balance_specs:
        row = latest(metric, "IFRS balance sheet")
        if row is not None:
            balance_values[metric] = row

    balance_rows = []
    periods = {pd.Timestamp(row["period_end"]) for row in balance_values.values()}
    units = {str(row["unit"]) for row in balance_values.values()}
    # Raw balances are displayed even if one input is missing; derived ratios
    # are calculated only when every required input shares one date and unit.
    for metric, label in balance_specs:
        row = balance_values.get(metric)
        if row is not None:
            balance_rows.append({
                "Metric": label,
                "Period": pd.Timestamp(row["period_end"]).strftime("%d %b %Y"),
                "Value": float(row["value"]),
                "Unit": row["unit"],
                "Interpretation note": "Reported IFRS consolidated balance-sheet amount; point-in-time, not a quarterly flow.",
            })

    if len(periods) == 1 and len(units) == 1 and "total_assets" in balance_values:
        assets = float(balance_values["total_assets"]["value"])
        if assets > 0:
            derived = []
            if "total_cash_and_investments" in balance_values:
                derived.append(("Cash and investments / total assets", "total_cash_and_investments"))
            if "shareholders_funds" in balance_values:
                derived.append(("Shareholders' funds / total assets", "shareholders_funds"))
            liabilities_present = all(k in balance_values for k in ("current_liabilities", "non_current_liabilities"))
            if liabilities_present:
                total_liabilities = (
                    float(balance_values["current_liabilities"]["value"])
                    + float(balance_values["non_current_liabilities"]["value"])
                )
                derived.append(("Current plus non-current liabilities / total assets", None))
            for label, key in derived:
                numerator = total_liabilities if key is None else float(balance_values[key]["value"])
                balance_rows.append({
                    "Metric": label,
                    "Period": next(iter(periods)).strftime("%d %b %Y"),
                    "Value": round(numerator / assets * 100, 2),
                    "Unit": "% of total assets",
                    "Interpretation note": "Derived from same-date IFRS balance-sheet lines; not a debt ratio or liquidity score.",
                })

    missing_specs = [
        ("Interest-bearing borrowings and lease liabilities", "Needed to assess gross debt and net debt; total liabilities are not a substitute."),
        ("Finance costs / interest expense", "Needed to assess interest coverage and financing burden."),
        ("EBITDA or operating profit reconciliation", "Needed for consistent operating-profitability and EV/EBITDA analysis."),
        ("Current assets and inventories", "Needed for current and quick liquidity ratios."),
        ("Trade receivables and contract assets", "Needed to assess collection quality and working-capital intensity."),
        ("At least 8 quarters of comparable results", "Needed to distinguish durable growth and margin trends from quarterly volatility."),
    ]
    missing_rows = [{"Metric not available": metric, "Why it matters": why} for metric, why in missing_specs]
    return (
        pd.DataFrame(operating_rows, columns=operating_columns),
        pd.DataFrame(balance_rows, columns=balance_columns),
        pd.DataFrame(missing_rows, columns=missing_columns),
    )


def latest_filing_status(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize latest period and filing metadata separately by reporting basis."""
    if data.empty:
        return pd.DataFrame()
    out = data.groupby(["basis"], dropna=False).agg(
        Latest_period=("period_end", "max"),
        Latest_filing=("filing_date", "max"),
        Metrics=("metric", "nunique"),
        Sources=("source_name", lambda values: ", ".join(sorted(set(map(str, values))))),
    ).reset_index()
    return out


def period_growth_table(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate exact-period growth only for unique, like-for-like observations."""
    if data.empty:
        return pd.DataFrame()
    eligible = {
        "revenue_from_operations",
        "net_cash_from_operating_activities",
        "free_cash_flow",
    }
    subset = data[data["metric"].isin(eligible)].copy()
    if subset.empty:
        return pd.DataFrame()

    keys = ["metric", "period_end", "basis", "unit"]
    # For duplicate observations, retain exactly one primary row only when
    # secondary publisher rows agree with it. Conflicts remain excluded.
    selected = []
    for _, group in subset.groupby(keys, dropna=False):
        if len(group) == 1:
            selected.append(group.index[0])
            continue
        if "source_class" not in group.columns:
            continue
        primary = group[group["source_class"].astype(str).str.startswith("Primary")]
        if len(primary) == 1 and group["value"].nunique(dropna=True) == 1:
            selected.append(primary.index[0])
    subset = subset.loc[selected]
    rows = []
    for (metric, basis, unit), group in subset.groupby(["metric", "basis", "unit"]):
        values = {
            pd.Timestamp(row.period_end).strftime("%Y-%m-%d"): float(row.value)
            for row in group.itertuples()
        }
        for current in sorted(values):
            current_date = pd.Timestamp(current)
            comparisons = (
                ("YoY", (current_date - pd.DateOffset(years=1)).strftime("%Y-%m-%d")),
                ("QoQ", (current_date - pd.DateOffset(months=3)).strftime("%Y-%m-%d")),
            )
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
