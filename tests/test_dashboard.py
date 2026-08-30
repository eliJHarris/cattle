import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DashboardSmokeTest(unittest.TestCase):
    def test_generated_page_is_self_contained_and_interactive(self):
        page = ROOT / "docs" / "index.html"
        self.assertTrue(page.exists(), "Run python build_dashboard.py first")
        text = page.read_text(encoding="utf-8")
        self.assertIn("Cattle Futures", text)
        self.assertIn('id="theme-toggle"', text)
        self.assertIn('id="start-date"', text)
        self.assertIn('id="end-date"', text)
        self.assertIn("Plotly.newPlot", text)
        self.assertNotIn('src="https://cdn.plot.ly', text)
        self.assertNotIn('href="/assets/', text)

    def test_build_metadata_is_valid_json(self):
        metadata = ROOT / "docs" / "build-metadata.json"
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        self.assertIn("data_through", payload)
        self.assertIn("source_status", payload)
        self.assertGreaterEqual(payload["chart_count"], 8)


if __name__ == "__main__":
    unittest.main()
