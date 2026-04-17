import unittest
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_data import (
    MetricSpec,
    build_portfolio_dataframe,
    get_sector_info,
    load_all_tickers,
    normalize_metric,
    score_stock,
    signal_from_score,
    stock_metrics,
)

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

    @unittest.mock.patch("portfolio_data._ticker_info", return_value={})
    @unittest.mock.patch(
        "portfolio_data.metadata_for_ticker",
        return_value={
            "company_name": "Sample",
            "company_name_ko": "샘플",
            "company_name_en": "Sample",
            "sector": "Technology",
            "sub_sector": "Software",
            "market": "NASDAQ",
        },
    )
    @unittest.mock.patch(
        "portfolio_data.stock_metrics",
        return_value={
            "pe_ratio": 20.0,
            "revenue_growth_yoy": 0.2,
            "asset_turnover": 0.8,
            "debt_ratio": 0.3,
            "dividend_yield": 0.01,
            "tech_cycle": 0.2,
            "current_price": 100.0,
        },
    )
    @unittest.mock.patch("portfolio_data.get_sector", return_value="MARKET")
    @unittest.mock.patch("portfolio_data.get_sector_info", return_value=("Technology", "Software"))
    def test_dataframe_contains_all_metrics_and_sector_display(self, *_):
        df = build_portfolio_dataframe(["MSFT"])
        self.assertEqual(df.loc[0, "sector"], "Technology")
        self.assertEqual(df.loc[0, "sub_sector"], "Software")
        self.assertIn("peg_ratio", df.columns)
        self.assertIn("peg_ratio_normalized", df.columns)
        self.assertNotIn("current_price_normalized", df.columns)

    def test_get_sector_info_prefers_stock_metadata(self):
        sector, sub_sector = get_sector_info("MSFT")
        self.assertTrue(sector)
        self.assertTrue(sub_sector)

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
        row_map = {row["ticker"]: row for row in rows}
        self.assertIn("005930", row_map)
        self.assertIn("MSFT", row_map)
        self.assertEqual(row_map["MSFT"]["exchange"], "NASDAQ")
        self.assertEqual(row_map["005930"]["company_name_ko"], "삼성전자")

    def test_load_all_tickers_normalizes_krx_suffix_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "market_db.csv"
            csv_path.write_text(
                "ticker,company_name_ko,company_name_en,exchange,sector,sub_sector\n"
                "930.KS,삼성전자,Samsung Electronics,KOSPI,Technology,Semiconductors\n",
                encoding="utf-8",
            )
            with patch("portfolio_data.MARKET_DB_PATH", csv_path):
                load_all_tickers.cache_clear()
                rows = load_all_tickers()
                load_all_tickers.cache_clear()
        self.assertEqual(rows[0]["ticker"], "000930")

    @patch("portfolio_data._rows_from_market_csv", return_value=[])
    @patch(
        "portfolio_data.load_watchlist",
        return_value=pd.DataFrame(
            [{"ticker": "AAPL", "company_name_en": "Apple", "exchange": "NASDAQ", "sector": "Technology", "sub_sector": "Consumer Electronics"}]
        ),
    )
    def test_load_all_tickers_falls_back_to_watchlist(self, mock_load_watchlist, mock_rows_from_market_csv):
        load_all_tickers.cache_clear()
        rows = load_all_tickers()
        load_all_tickers.cache_clear()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "AAPL")
        self.assertEqual(rows[0]["company_name_en"], "Apple")

    @patch("portfolio_data._finnhub_metrics", return_value={})
    @patch("portfolio_data._ticker_object")
    @patch("portfolio_data._ticker_info", return_value={"trailingPE": 20.0, "pegRatio": 1.2})
    @patch("portfolio_data.metadata_for_ticker", return_value={"market": "NASDAQ"})
    def test_market_stock_metrics_include_peg_ratio(self, *_):
        metrics = stock_metrics("MSFT", "MARKET")
        self.assertIn("peg_ratio", metrics)
        self.assertEqual(metrics["peg_ratio"], 1.2)


if __name__ == "__main__":
    unittest.main()
