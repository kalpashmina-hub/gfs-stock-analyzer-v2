"""Regression tests for the source-aware fundamental filing intake."""
import unittest
from pathlib import Path

import pandas as pd

from fundamental_data import (
    classify_source,
    compare_sources,
    period_growth_table,
    validate_fundamental_upload,
)

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "tcs_q2_fy27_fundamentals.csv"


class FundamentalIntakeTests(unittest.TestCase):
    def setUp(self):
        self.valid = pd.DataFrame([{
            "metric": "revenue_from_operations",
            "value": 100,
            "period_end": "30-09-2026",
            "basis": "consolidated",
            "unit": "INR crore (IndAS; quarter)",
            "source_name": "NSE filing",
            "source_url": "https://nsearchives.nseindia.com/filing.html",
            "filing_date": "08-10-2026",
        }])

    def test_tcs_pilot_validates_and_keeps_provenance(self):
        raw = pd.read_csv(CSV_PATH)
        result = validate_fundamental_upload(raw)
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.data), len(raw))
        self.assertTrue(result.data["source_class"].str.startswith("Primary").all())
        self.assertEqual(set(result.data["basis"]), {"consolidated"})
        q1 = result.data[result.data["period_end"] == pd.Timestamp("2026-06-30")]
        profit = q1.set_index("metric")["value"]
        self.assertEqual(float(profit["profit_after_tax_before_non_controlling_interests"]), 13420.0)
        self.assertEqual(float(profit["net_profit_attributable_to_owners"]), 13349.0)
        self.assertTrue(q1["source_url"].str.contains("173420_09072026").all())

    def test_future_period_and_filing_dates_are_rejected(self):
        future = self.valid.copy()
        future.loc[0, "period_end"] = "30-12-2026"
        result = validate_fundamental_upload(future)
        self.assertTrue(any("period_end is in the future" in error for error in result.errors))

    def test_non_finite_and_blank_values_are_rejected(self):
        bad = self.valid.copy()
        bad.loc[0, "value"] = float("inf")
        result = validate_fundamental_upload(bad)
        self.assertTrue(result.errors)

    def test_lookalike_domain_is_not_primary(self):
        self.assertEqual(
            classify_source("NSE filing", "https://nseindia.com.attacker.example/file"),
            "Unclassified — verify source",
        )

    def test_secondary_domain_is_flagged(self):
        self.assertEqual(
            classify_source("Research site", "https://www.screener.in/company/TCS/"),
            "Secondary — cross-check only",
        )

    def test_growth_omits_duplicate_observation_groups(self):
        data = pd.DataFrame([
            {"metric": "revenue_from_operations", "value": 100, "period_end": pd.Timestamp("2025-09-30"), "basis": "consolidated", "unit": "INR crore"},
            {"metric": "revenue_from_operations", "value": 120, "period_end": pd.Timestamp("2026-09-30"), "basis": "consolidated", "unit": "INR crore"},
            {"metric": "revenue_from_operations", "value": 120, "period_end": pd.Timestamp("2026-09-30"), "basis": "consolidated", "unit": "INR crore"},
        ])
        self.assertTrue(period_growth_table(data).empty)

    def test_source_comparison_reports_spread(self):
        data = pd.DataFrame([
            {"metric": "revenue", "value": 100, "period_end": pd.Timestamp("2026-09-30"), "basis": "consolidated", "unit": "INR crore", "source_name": "NSE"},
            {"metric": "revenue", "value": 101, "period_end": pd.Timestamp("2026-09-30"), "basis": "consolidated", "unit": "INR crore", "source_name": "Company"},
        ])
        comparison = compare_sources(data)
        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison.iloc[0]["Review required"], "Yes")


if __name__ == "__main__":
    unittest.main()
