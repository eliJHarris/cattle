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
        self.assertIn('id="range-preset"', text)
        for preset in ("1d", "5d", "1m", "6m", "1y", "5y", "all"):
            self.assertIn(f'value="{preset}"', text)
        self.assertIn("applyDateRange", text)
        self.assertIn("'.autorange'", text)
        self.assertIn("preloaded in this page", text)
        self.assertIn('id="topic-filter"', text)
        self.assertIn("Plotly.newPlot", text)
        self.assertIn('class="chart-wrap"', text)
        self.assertIn('data-panel-count="4"', text)
        self.assertIn('applyMobileChartLayout', text)
        self.assertIn('touch-action:pan-y', text)
        self.assertIn("'dragmode': false", text)
        self.assertIn('class="table-scroll"', text)
        self.assertIn('@media (max-width:680px)', text)
        self.assertNotIn('src="https://cdn.plot.ly', text)
        self.assertNotIn('href="/assets/', text)

    def test_build_metadata_is_valid_json(self):
        metadata = ROOT / "docs" / "build-metadata.json"
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        self.assertIn("data_through", payload)
        self.assertIn("source_status", payload)
        self.assertEqual(payload["range_presets"], ["1d", "5d", "1m", "6m", "1y", "5y", "all"])
        self.assertEqual(payload["range_data"]["price_frequency"], "daily")
        self.assertGreaterEqual(payload["chart_count"], 8)

    def test_pages_quote_snapshot_is_valid_json(self):
        snapshot = ROOT / "docs" / "live-market.json"
        self.assertTrue(snapshot.exists(), "Run python build_dashboard.py first")
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        self.assertEqual(payload["feed_mode"], "snapshot")
        self.assertIn("generated_at", payload)
        self.assertIn("GF=F", payload["quotes"])


if __name__ == "__main__":
    unittest.main()
