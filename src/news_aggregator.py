from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Union

import requests

ENGLISH_STOPWORDS = {"the", "and", "for", "with", "from", "stock", "shares", "will", "this", "that", "company"}
POSITIVE_WORDS = {"beat", "growth", "surge", "win", "record", "upgrade", "strong", "profit", "optimistic"}
NEGATIVE_WORDS = {"miss", "fall", "drop", "risk", "downgrade", "loss", "delay", "lawsuit", "weak"}
POSITIVE_PATTERN = re.compile(rf"\b({'|'.join(sorted(re.escape(w) for w in POSITIVE_WORDS))})\b")
NEGATIVE_PATTERN = re.compile(rf"\b({'|'.join(sorted(re.escape(w) for w in NEGATIVE_WORDS))})\b")


def _sentiment_label(headline: str, summary: str) -> str:
    text = f"{headline} {summary}".lower()
    positive = len(POSITIVE_PATTERN.findall(text))
    negative = len(NEGATIVE_PATTERN.findall(text))
    if positive > negative:
        return "Positive"
    if negative > positive:
        return "Negative"
    return "Neutral"


def recent_news(ticker: str, keyword: str = "", months: int = 3) -> List[Dict[str, str]]:
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return []

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=30 * max(1, months))

    try:
        response = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker.upper(),
                "from": start.date().isoformat(),
                "to": now.date().isoformat(),
                "token": api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    if not isinstance(payload, list):
        return []

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


def extract_keywords(news_rows: List[Dict[str, str]], top_n: int = 8) -> List[Dict[str, Union[str, int]]]:
    tokens: List[str] = []

    for row in news_rows:
        headline = str(row.get("headline") or "").lower()
        tokens.extend([w for w in re.findall(r"[a-z]{3,}", headline) if w not in ENGLISH_STOPWORDS])

    counts = Counter(tokens)
    return [{"keyword": k, "count": v} for k, v in counts.most_common(top_n)]
