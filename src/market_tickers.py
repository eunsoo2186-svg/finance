from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import requests

from watchlist_manager import load_watchlist


STATIC_MARKET_TICKERS = [
    {"ticker": "AAPL", "company_name": "Apple", "market": "NASDAQ"},
    {"ticker": "MSFT", "company_name": "Microsoft", "market": "NASDAQ"},
    {"ticker": "NVDA", "company_name": "NVIDIA", "market": "NASDAQ"},
    {"ticker": "TSLA", "company_name": "Tesla", "market": "NASDAQ"},
    {"ticker": "JPM", "company_name": "JPMorgan Chase", "market": "NYSE"},
    {"ticker": "BA", "company_name": "Boeing", "market": "NYSE"},
    {"ticker": "005930.KS", "company_name": "Samsung Electronics", "market": "KOSPI"},
    {"ticker": "000660.KS", "company_name": "SK hynix", "market": "KOSPI"},
    {"ticker": "035420.KS", "company_name": "NAVER", "market": "KOSPI"},
    {"ticker": "035720.KQ", "company_name": "Kakao", "market": "KOSDAQ"},
]


def _map_us_market(mic_exchange: str | None) -> str:
    mic_exchange = (mic_exchange or "").upper()
    if mic_exchange == "XNAS":
        return "NASDAQ"
    if mic_exchange == "XNYS":
        return "NYSE"
    return "US"


@lru_cache(maxsize=1)
def _finnhub_symbols() -> List[Dict[str, str]]:
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return []

    merged: Dict[str, Dict[str, str]] = {}

    for exchange in ["US", "KRX"]:
        try:
            response = requests.get(
                "https://finnhub.io/api/v1/stock/symbol",
                params={"exchange": exchange, "token": api_key},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            continue

        if not isinstance(payload, list):
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("symbol") or "").upper().strip()
            if not ticker:
                continue
            if exchange == "US":
                market = _map_us_market(str(item.get("mic") or item.get("micExchange") or ""))
                if market not in {"NASDAQ", "NYSE"}:
                    continue
            else:
                market = "KOSPI" if ticker.endswith(".KS") else "KOSDAQ" if ticker.endswith(".KQ") else "KRX"
            merged[ticker] = {
                "ticker": ticker,
                "company_name": str(item.get("description") or ticker),
                "market": market,
            }

    return list(merged.values())


def load_market_tickers() -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}

    for row in load_watchlist().to_dict(orient="records"):
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue
        merged[ticker] = {
            "ticker": ticker,
            "company_name": str(row.get("company_name", ticker)),
            "market": str(row.get("market", "UNKNOWN")).upper(),
        }

    for row in STATIC_MARKET_TICKERS + _finnhub_symbols():
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue
        if ticker not in merged:
            merged[ticker] = {
                "ticker": ticker,
                "company_name": str(row.get("company_name", ticker)),
                "market": str(row.get("market", "UNKNOWN")).upper(),
            }

    return sorted(merged.values(), key=lambda x: x["ticker"])
