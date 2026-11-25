"""Regression tests for non-destructive market-data quality checks."""
import unittest
import pandas as pd
from market_data_quality import audit_ohlcv, audit_actions, audit_price_adjustment_metadata


def sample(dates=None):
    return pd.DataFrame({
        "Date": dates or ["2026-01-01", "2026-01-02"],
        "Open": [100, 101], "High": [102, 103],
        "Low": [99, 100], "Close": [101, 102],
        "Volume": [1000, 1200],
    })


class MarketDataQualityTests(unittest.TestCase):
    def test_clean_data(self):
        result = audit_ohlcv(sample())
        self.assertTrue(result["findings"].empty)
        self.assertTrue(result["non_destructive"])

    def test_invalid_date_is_reported_not_silently_lost(self):
        data = sample(["bad-date", "2026-01-02"])
        result = audit_ohlcv(data)
        self.assertEqual(result["invalid_date_rows"], 1)
        self.assertIn("INVALID_DATE", set(result["findings"].Code))

    def test_duplicate_and_order(self):
        data = sample(["2026-01-02", "2026-01-01"])
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        result = audit_ohlcv(data)
        self.assertEqual(result["duplicate_date_rows"], 2)
        self.assertTrue(result["dates_out_of_order"])
        self.assertIn("DUPLICATE_DATE", set(result["findings"].Code))

    def test_invalid_ohlc_and_negative_volume(self):
        data = sample(["2026-01-01", "2026-01-02"])
        data.loc[0, "High"] = 90
        data.loc[1, "Volume"] = -1
        codes = set(audit_ohlcv(data).findings.Code) if False else set(audit_ohlcv(data)["findings"].Code)
        self.assertIn("INVALID_OHLC", codes)
        self.assertIn("NEGATIVE_VOLUME", codes)

    def test_missing_volume_is_warning_not_blocking_error(self):
        data = sample()
        data.loc[1, "Volume"] = float("nan")
        findings = audit_ohlcv(data)["findings"]
        row = findings[findings["Date"] == pd.Timestamp("2026-01-02")]
        self.assertIn("MISSING_VOLUME", set(row["Code"]))
        self.assertFalse((row["Severity"] == "error").any())
        self.assertNotIn("NON_FINITE_VALUE", set(row["Code"]))

    def test_missing_price_is_blocking_without_duplicate_non_finite_flag(self):
        data = sample()
        data.loc[1, "Close"] = float("nan")
        findings = audit_ohlcv(data)["findings"]
        row = findings[findings["Date"] == pd.Timestamp("2026-01-02")]
        self.assertIn("MISSING_OR_NON_NUMERIC", set(row["Code"]))
        self.assertNotIn("NON_FINITE_VALUE", set(row["Code"]))

    def test_infinite_value_is_non_finite_error(self):
        data = sample()
        data.loc[1, "Volume"] = float("inf")
        findings = audit_ohlcv(data)["findings"]
        row = findings[findings["Date"] == pd.Timestamp("2026-01-02")]
        self.assertIn("NON_FINITE_VALUE", set(row["Code"]))

    def test_zero_volume_flat_bar(self):
        data = sample(["2026-01-01", "2026-01-02"])
        data.loc[0, ["Open", "High", "Low", "Close", "Volume"]] = [100, 100, 100, 100, 0]
        codes = set(audit_ohlcv(data)["findings"].Code)
        self.assertIn("ZERO_VOLUME_FLAT_BAR", codes)

    def test_large_move_review(self):
        data = sample()
        data.loc[1, ["Open", "High", "Low", "Close"]] = [70, 71, 69, 70]
        self.assertIn("LARGE_DAILY_RETURN", set(audit_ohlcv(data)["findings"].Code))

    def test_calendar_holiday_observation(self):
        data = sample(["2026-01-15", "2026-01-16"])
        result = audit_ohlcv(data, exchange_calendar=["2026-01-15"], exchange="NSE")
        row = result["findings"].query("Code == 'OBSERVATION_ON_NON_TRADING_DATE'")
        self.assertEqual(len(row), 1)

    def test_calendar_missing_session(self):
        data = sample(["2026-01-01", "2026-01-05"])
        calendar = pd.DataFrame({"Date": ["2026-01-01", "2026-01-02", "2026-01-05"],
                                 "IsTradingSession": [True, True, True]})
        codes = set(audit_ohlcv(data, exchange_calendar=calendar)["findings"].Code)
        self.assertIn("MISSING_TRADING_SESSION", codes)

    def test_actions_and_metadata(self):
        actions = pd.DataFrame({"Dividends": [0, 2], "Stock Splits": [0, 2]})
        result = audit_actions(actions)
        self.assertEqual(result["dividend_events"], 1)
        self.assertEqual(result["split_events"], 1)
        metadata = audit_price_adjustment_metadata({"provider": "sample", "interval": "1d",
                                                      "auto_adjust": True, "actions": False})
        self.assertTrue(metadata["adjustment_policy_known"])
        self.assertEqual(metadata["missing_core_fields"], [])

    def test_missing_actions_is_not_no_actions(self):
        self.assertFalse(audit_actions(None)["available"])
        self.assertIn("not evidence", audit_actions(None)["note"])


if __name__ == "__main__":
    unittest.main()
