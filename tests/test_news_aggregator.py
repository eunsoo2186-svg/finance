import unittest
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_aggregator import NewsAggregator


class NewsAggregatorTests(unittest.TestCase):
    @patch("news_aggregator.requests.get")
    def test_get_stock_news_returns_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [{"headline": "test"}]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        rows = NewsAggregator("dummy").get_stock_news("MSFT")
        self.assertEqual(len(rows), 1)

    @patch("news_aggregator.requests.get")
    def test_get_sector_news_aggregates(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "a", "headline": "first", "datetime": 10, "url": "https://a"},
            {"id": "a", "headline": "first", "datetime": 10, "url": "https://a"},
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        rows = NewsAggregator("dummy").get_sector_news("Technology")
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
