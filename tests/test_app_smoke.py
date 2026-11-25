"""Streamlit startup smoke test; does not fetch market data or alter GFS calculations."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppStartupTests(unittest.TestCase):
    def test_app_renders_landing_page_without_exception(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
        self.assertEqual(len(app.exception), 0, [exception.message for exception in app.exception])
        self.assertTrue(
            any("GFS Stock Analyzer" in element.value for element in app.title),
            "Expected the GFS Stock Analyzer title on the landing page.",
        )


if __name__ == "__main__":
    unittest.main()
