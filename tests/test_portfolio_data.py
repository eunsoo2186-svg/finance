import unittest
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_data import MetricSpec, load_all_tickers, normalize_metric, score_stock, signal_from_score

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

    def test_load_all_tickers_reads_market_db_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "market_db.csv"
            csv_path.write_text(
                "ticker,company_name_ko,company_name_en,exchange,sector,sub_sector\n"
                "MSFT,마이크로소프트,Microsoft,NASDAQ,Technology,Software\n"
                "005930,삼성전자,Samsung Electronics,KOSPI,Technology,Semiconductors\n",
                encoding="utf-8",
            )
            with patch("portfolio_data.MARKET_DB_PATH", csv_path):
                load_all_tickers.cache_clear()
                rows = load_all_tickers()
                load_all_tickers.cache_clear()
        self.assertEqual(rows[0]["ticker"], "005930")
        self.assertEqual(rows[1]["exchange"], "NASDAQ")
        self.assertEqual(rows[0]["company_name_ko"], "삼성전자")

    @patch("portfolio_data._rows_from_market_csv", return_value=[])
    @patch(
        "portfolio_data.load_watchlist",
        return_value=pd.DataFrame(
            [{"ticker": "AAPL", "company_name_en": "Apple", "exchange": "NASDAQ", "sector": "Technology", "sub_sector": "Consumer Electronics"}]
        ),
    )
    def test_load_all_tickers_falls_back_to_watchlist(self, *_):
        load_all_tickers.cache_clear()
        rows = load_all_tickers()
        load_all_tickers.cache_clear()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "AAPL")
        self.assertEqual(rows[0]["company_name_en"], "Apple")


if __name__ == "__main__":
    unittest.main()
