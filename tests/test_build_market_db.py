import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_market_db


class BuildMarketDbTests(unittest.TestCase):
    @patch("build_market_db._load_krx_rows_from_cache", return_value=[])
    def test_load_krx_rows_returns_empty_on_pykrx_failure(self, *_):
        fake_stock = MagicMock()
        fake_stock.get_market_ticker_list.side_effect = requests.ConnectionError("krx down")
        with patch.object(build_market_db, "stock", fake_stock):
            rows = build_market_db._load_krx_rows("KOSPI")
        self.assertEqual(rows, [])

    @patch("build_market_db._load_nasdaq_rows", side_effect=RuntimeError("nasdaq fail"))
    @patch("build_market_db._load_nyse_rows", return_value=[])
    @patch("build_market_db._load_krx_rows", return_value=[])
    @patch(
        "build_market_db.load_watchlist",
        return_value=pd.DataFrame(
            [
                {
                    "ticker": "005930",
                    "company_name_ko": "삼성전자",
                    "company_name_en": "Samsung Electronics",
                    "exchange": "KOSPI",
                    "sector": "Technology",
                    "sub_sector": "Semiconductors",
                }
            ]
        ),
    )
    def test_build_market_db_keeps_running_with_watchlist_fallback(self, *_):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "market_db.csv"
            frame = build_market_db.build_market_db(output_path, enrich_sectors=False, delay_seconds=0.0)

            self.assertTrue(output_path.exists())
            self.assertFalse(frame.empty)
            row = frame.loc[frame["ticker"] == "005930"].iloc[0]
            self.assertEqual(row["exchange"], "KOSPI")
            self.assertEqual(row["sector"], "Technology")
            self.assertEqual(row["sub_sector"], "Semiconductors")

    @patch("build_market_db._load_nasdaq_rows", side_effect=RuntimeError("nasdaq fail"))
    @patch("build_market_db._load_nyse_rows", return_value=[])
    @patch(
        "build_market_db._load_krx_rows_from_cache",
        side_effect=[
            [
                {
                    "ticker": "005930",
                    "company_name_ko": "삼성전자",
                    "company_name_en": "삼성전자",
                    "exchange": "KOSPI",
                    "sector": "전기, 전자",
                    "sub_sector": "전자부품 제조업",
                }
            ],
            [
                {
                    "ticker": "035720",
                    "company_name_ko": "카카오",
                    "company_name_en": "카카오",
                    "exchange": "KOSDAQ",
                    "sector": "서비스업",
                    "sub_sector": "포털 및 기타 인터넷 정보매개 서비스업",
                }
            ],
        ],
    )
    @patch("build_market_db.load_watchlist", return_value=pd.DataFrame())
    def test_build_market_db_uses_krx_cache_fallback(self, *_):
        fake_stock = MagicMock()
        fake_stock.get_market_ticker_list.side_effect = requests.ConnectionError("krx down")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "market_db.csv"
            with patch.object(build_market_db, "stock", fake_stock):
                frame = build_market_db.build_market_db(output_path, enrich_sectors=False, delay_seconds=0.0)

            self.assertEqual(len(frame), 2)
            self.assertSetEqual(set(frame["exchange"].tolist()), {"KOSPI", "KOSDAQ"})
            self.assertIn("005930", set(frame["ticker"].tolist()))
            self.assertIn("035720", set(frame["ticker"].tolist()))


if __name__ == "__main__":
    unittest.main()
