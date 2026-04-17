import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_data import MetricSpec, normalize_metric, score_stock, signal_from_score

BUY_SIGNAL = "진입"
HOLD_SIGNAL = "보유"
SELL_SIGNAL = "매도"


class PortfolioDataTests(unittest.TestCase):
    def test_normalize_lower_better(self):
        spec = MetricSpec("PE", 10, 30, False, "src", "formula")
        self.assertAlmostEqual(normalize_metric(10, spec), 1.0)
        self.assertAlmostEqual(normalize_metric(30, spec), 0.0)

    def test_normalize_missing_is_neutral(self):
        spec = MetricSpec("Any", 0, 1, True, "src", "formula")
        self.assertAlmostEqual(normalize_metric(None, spec), 0.5)

    def test_score_and_signal(self):
        metrics = {
            "pe_ratio": 15,
            "peg_ratio": 1.0,
            "revenue_growth_yoy": 0.2,
            "rd_ratio": 0.15,
            "asset_turnover": 0.8,
            "tech_cycle": 0.2,
        }
        weights = {
            "pe_ratio": 0.18,
            "peg_ratio": 0.20,
            "revenue_growth_yoy": 0.22,
            "rd_ratio": 0.18,
            "asset_turnover": 0.12,
            "tech_cycle": 0.10,
        }
        score, normalized = score_stock(metrics, "AI", weights)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertEqual(set(normalized.keys()), set(metrics.keys()))
        for value in normalized.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertIn(signal_from_score(score), {BUY_SIGNAL, HOLD_SIGNAL, SELL_SIGNAL})


if __name__ == "__main__":
    unittest.main()
