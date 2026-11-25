"""Regression tests for generic provider-statement CSV conversion."""
import unittest
import pandas as pd

from provider_fundamentals import statement_frames_to_long


class ProviderFundamentalExportTests(unittest.TestCase):
    def test_converts_multiple_statement_tables_without_inventing_units(self):
        income = pd.DataFrame(
            {pd.Timestamp("2026-09-30"): [100.0], pd.Timestamp("2026-06-30"): [90.0]},
            index=["Total Revenue"],
        )
        balance = pd.DataFrame(
            {pd.Timestamp("2026-09-30"): [250.0]},
            index=["Total Assets"],
        )
        result = statement_frames_to_long(
            {"Quarterly income statement": income, "Quarterly balance sheet": balance},
            ticker="INFY.NS",
            currency="INR",
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(set(result["Ticker"]), {"INFY.NS"})
        self.assertIn("Total Revenue", set(result["Metric"]))
        self.assertIn("Total Assets", set(result["Metric"]))
        self.assertTrue((result["Currency"] == "INR").all())
        self.assertTrue(result["Data basis"].str.contains("not confirmed").all())

    def test_empty_provider_tables_return_export_schema(self):
        result = statement_frames_to_long({"Quarterly income statement": pd.DataFrame()}, ticker="TCS.NS")
        self.assertTrue(result.empty)
        self.assertIn("Reported value", result.columns)


if __name__ == "__main__":
    unittest.main()
