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
    def test_load_krx_rows_returns_empty_on_pykrx_failure(self):
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


if __name__ == "__main__":
    unittest.main()
