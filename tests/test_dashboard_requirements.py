import unittest
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_analysis import recommendation_from_score
from portfolio_data import build_portfolio_dataframe, metric_basis_table
from watchlist_manager import load_watchlist, save_holdings, load_holdings


class DashboardRequirementTests(unittest.TestCase):
    def test_recommendation_split_for_held_vs_unheld(self):
        self.assertEqual(recommendation_from_score(80, held=True), "매도")
        self.assertEqual(recommendation_from_score(60, held=True), "보유")
        self.assertEqual(recommendation_from_score(50, held=True), "추매")
        self.assertEqual(recommendation_from_score(30, held=True), "손절")
        self.assertEqual(recommendation_from_score(80, held=False), "매수")
        self.assertEqual(recommendation_from_score(60, held=False), "관심")
        self.assertEqual(recommendation_from_score(30, held=False), "회피")

    def test_metric_labels_are_english_only(self):
        ai_basis = metric_basis_table("AI")
        self.assertFalse(ai_basis["Metric"].str.contains(r"[가-힣]").any())

    @patch("portfolio_data._ticker_info", return_value={})
    @patch(
        "portfolio_data.metadata_for_ticker",
        return_value={
            "company_name": "Microsoft",
            "company_name_ko": "마이크로소프트",
            "company_name_en": "Microsoft",
            "sector": "AI",
            "sub_sector": "LLM",
            "market": "NASDAQ",
        },
    )
    @patch(
        "portfolio_data.stock_metrics",
        return_value={
            "pe_ratio": 20,
            "peg_ratio": 1.1,
            "revenue_growth_yoy": 0.2,
            "rd_ratio": 0.1,
            "asset_turnover": 0.7,
            "tech_cycle": 0.2,
        },
    )
    @patch("portfolio_data.get_sector", return_value="AI")
    def test_dataframe_includes_ticker_display(self, *_):
        df = build_portfolio_dataframe(["MSFT"])
        self.assertIn("ticker_display", df.columns)
        self.assertEqual(df.loc[0, "ticker_display"], "MSFT")
        self.assertEqual(df.loc[0, "company_name_ko"], "마이크로소프트")

    def test_watchlist_loaded_and_holdings_persist(self):
        watchlist = load_watchlist()
        self.assertIn("ticker", watchlist.columns)
        self.assertIn("company_name_ko", watchlist.columns)
        self.assertIn("company_name_en", watchlist.columns)

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir) / "holdings.json"
            payload = {"MSFT": {"purchase_price": 100.0, "quantity": 3, "currency": "USD", "target_price": 140.0}}
            with patch("watchlist_manager.HOLDINGS_PATH", temp_path):
                save_holdings(payload)
                loaded = load_holdings()
            self.assertIn("MSFT", loaded)
            self.assertEqual(loaded["MSFT"]["target_price"], 140.0)


if __name__ == "__main__":
    unittest.main()
