"""Tests for non-destructive OHLCV quality checks."""
import unittest
import pandas as pd

from market_data_quality import audit_ohlcv, audit_actions


class MarketDataQualityTests(unittest.TestCase):
    def test_valid_data_has_no_findings(self):
        data = pd.DataFrame({
            "Date": ["2026-01-01", "2026-01-02", "2026-01-05"],
            "Open": [100, 101, 102], "High": [102, 103, 104],
            "Low": [99, 100, 101], "Close": [101, 102, 103],
            "Volume": [1000, 1200, 900],
        })
        result = audit_ohlcv(data)
        self.assertEqual(result["rows_input"], 3)
        self.assertTrue(result["findings"].empty)
        self.assertTrue(result["non_destructive"])

    def test_flags_zero_volume_flat_bar_without_dropping_it(self):
        data = pd.DataFrame({
            "Date": ["2026-01-01", "2026-01-02"],
            "Open": [100, 101], "High": [100, 101],
            "Low": [100, 101], "Close": [100, 101],
            "Volume": [0, 1000],
        })
        result = audit_ohlcv(data)
        self.assertEqual(result["rows_with_valid_dates"], 2)
        self.assertIn("ZERO_VOLUME", set(result["findings"]["Code"]))
        self.assertIn("ZERO_VOLUME_FLAT_BAR", set(result["findings"]["Code"]))

    def test_flags_invalid_ohlc_and_negative_volume(self):
        data = pd.DataFrame({
            "Date": ["2026-01-01"], "Open": [105], "High": [104],
            "Low": [99], "Close": [103], "Volume": [-1],
        })
        codes = set(audit_ohlcv(data)["findings"]["Code"])
        self.assertIn("INVALID_OHLC", codes)
        self.assertIn("NEGATIVE_VOLUME", codes)

    def test_duplicate_and_order_diagnostics(self):
        data = pd.DataFrame({
            "Date": ["2026-01-02", "2026-01-01", "2026-01-02"],
            "Open": [101, 100, 102], "High": [102, 101, 103],
            "Low": [100, 99, 101], "Close": [101, 100, 102],
            "Volume": [1000, 1000, 1000],
        })
        result = audit_ohlcv(data)
        self.assertEqual(result["duplicate_date_rows"], 2)
        self.assertTrue(result["dates_out_of_order"])

    def test_large_move_is_review_not_auto_rejection(self):
        data = pd.DataFrame({
            "Date": ["2026-01-01", "2026-01-02"],
            "Open": [100, 70], "High": [101, 71],
            "Low": [99, 69], "Close": [100, 70],
            "Volume": [1000, 2000],
        })
        result = audit_ohlcv(data)
        self.assertIn("LARGE_DAILY_RETURN", set(result["findings"]["Code"]))
        self.assertEqual(result["rows_with_valid_dates"], 2)

    def test_missing_action_data_is_explicit(self):
        result = audit_actions(None)
        self.assertFalse(result["available"])
        self.assertIn("not evidence", result["note"])


if __name__ == "__main__":
    unittest.main()
