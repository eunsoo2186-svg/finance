from __future__ import annotations

import logging
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Union
from xml.etree import ElementTree

import requests

ENGLISH_STOPWORDS = {"the", "and", "for", "with", "from", "stock", "shares", "will", "this", "that", "company"}
POSITIVE_WORDS = {"beat", "growth", "surge", "win", "record", "upgrade", "strong", "profit", "optimistic"}
NEGATIVE_WORDS = {"miss", "fall", "drop", "risk", "downgrade", "loss", "delay", "lawsuit", "weak"}
POSITIVE_PATTERN = re.compile(rf"\b({'|'.join(re.escape(w) for w in POSITIVE_WORDS)})\b", re.IGNORECASE)
NEGATIVE_PATTERN = re.compile(rf"\b({'|'.join(re.escape(w) for w in NEGATIVE_WORDS)})\b", re.IGNORECASE)
LOGGER = logging.getLogger(__name__)

SECTOR_NEWS_SYMBOLS = {
    "Technology": ["MSFT", "NVDA", "TSM"],
    "Communication Services": ["META", "GOOGL", "NFLX"],
    "Healthcare": ["LLY", "JNJ", "PFE"],
    "Financial": ["JPM", "BAC", "MS"],
    "Industrials": ["BA", "LMT", "CAT"],
    "Energy": ["XOM", "CVX", "COP"],
    "Materials": ["RIO", "BHP", "FCX"],
    "Utilities": ["NEE", "DUK", "SO"],
    "Consumer Defensive": ["KO", "PG", "WMT"],
    "Consumer Cyclical": ["TSLA", "AMZN", "HD"],
}
MAX_SECTOR_NEWS_SYMBOLS = 3
# Keep request latency bounded for interactive dashboard refreshes.
FINNHUB_REQUEST_TIMEOUT = 8
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"


def _sentiment_label(headline: str, summary: str) -> str:
    text = f"{headline} {summary}"
    positive = len(POSITIVE_PATTERN.findall(text))
    negative = len(NEGATIVE_PATTERN.findall(text))
    if positive > negative:
        return "Positive"
    if negative > positive:
        return "Negative"
    return "Neutral"


class NewsAggregator:
    def __init__(self, finnhub_key: str | None):
        self.finnhub_key = finnhub_key or ""
        self.finnhub_url = "https://finnhub.io/api/v1/company-news"

    def _fetch_company_news(self, symbol: str, days: int) -> List[Dict]:
        if not self.finnhub_key:
            return []

        now = datetime.now(timezone.utc)
        from_date = (now - timedelta(days=max(1, days))).date().isoformat()
        to_date = now.date().isoformat()
        try:
            response = requests.get(
                self.finnhub_url,
                params={
                    "symbol": symbol.upper(),
                    "from": from_date,
                    "to": to_date,
                    "token": self.finnhub_key,
                },
                timeout=FINNHUB_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as exc:
            LOGGER.warning("Failed to fetch news for %s: %s", symbol, exc)
            return []

    def _fetch_google_news(self, symbol: str, days: int) -> List[Dict]:
        query = f"{symbol} 주식" if symbol.isdigit() else symbol.upper()
        try:
            response = requests.get(
                GOOGLE_NEWS_RSS_URL,
                params={"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"},
                timeout=FINNHUB_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            root = ElementTree.fromstring(response.text)
        except Exception as exc:
            LOGGER.warning("Failed to fetch Google RSS for %s: %s", symbol, exc)
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))
        rows: List[Dict] = []
        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date_raw = (item.findtext("pubDate") or "").strip()
            if not title or not link:
                continue
            try:
                dt = parsedate_to_datetime(pub_date_raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt = dt.astimezone(timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)
            if dt < cutoff:
                continue
            rows.append(
                {
                    "id": link,
                    "datetime": int(dt.timestamp()),
                    "source": "Google News",
                    "headline": title,
                    "summary": "",
                    "url": link,
                }
            )
        return rows

    def get_stock_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get raw Finnhub news for a specific stock."""
        rows = self._fetch_company_news(symbol, days)
        if rows:
            return rows
        return self._fetch_google_news(symbol, days)

    def get_sector_news(self, sector: str, days: int = 7) -> List[Dict]:
        """Get aggregated sector news using representative symbols."""
        symbols = SECTOR_NEWS_SYMBOLS.get(sector, [])
        if not symbols:
            symbols = [sector]

        merged: Dict[str, Dict] = {}
        for symbol in symbols[:MAX_SECTOR_NEWS_SYMBOLS]:
            for item in self._fetch_company_news(symbol, days):
                if not isinstance(item, dict):
                    continue
                key = str(item.get("id") or item.get("url") or item.get("headline") or "")
                if not key:
                    continue
                if key not in merged:
                    merged[key] = item

        return sorted(
            merged.values(),
            key=lambda x: int(x.get("datetime", 0) or 0),
            reverse=True,
        )


def _to_dashboard_rows(payload: List[Dict], keyword: str = "") -> List[Dict[str, str]]:
    needle = keyword.strip().lower()
    rows: List[Dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        headline = str(item.get("headline") or "")
        summary = str(item.get("summary") or "")
        content = f"{headline} {summary}".lower()
        if needle and needle not in content:
            continue
        rows.append(
            {
                "datetime": datetime.fromtimestamp(int(item.get("datetime", 0)), tz=timezone.utc).strftime("%Y-%m-%d"),
                "source": str(item.get("source") or ""),
                "headline": headline,
                "summary": summary,
                "sentiment": _sentiment_label(headline, summary),
                "url": str(item.get("url") or ""),
            }
        )
    return rows


def recent_news(ticker: str, keyword: str = "", months: int = 3) -> List[Dict[str, str]]:
    api_key = os.getenv("FINNHUB_API_KEY", "")
    payload = NewsAggregator(api_key).get_stock_news(ticker, days=max(1, months) * 30)
    return _to_dashboard_rows(payload, keyword=keyword)


def recent_sector_news(sector: str, days: int = 7) -> List[Dict[str, str]]:
    api_key = os.getenv("FINNHUB_API_KEY", "")
    payload = NewsAggregator(api_key).get_sector_news(sector, days=days)
    return _to_dashboard_rows(payload)


def extract_keywords(news_rows: List[Dict[str, str]], top_n: int = 8) -> List[Dict[str, Union[str, int]]]:
    tokens: List[str] = []

    for row in news_rows:
        headline = str(row.get("headline") or "").lower()
        tokens.extend([w for w in re.findall(r"[a-z]{3,}", headline) if w not in ENGLISH_STOPWORDS])

    counts = Counter(tokens)
    return [{"keyword": k, "count": v} for k, v in counts.most_common(top_n)]
