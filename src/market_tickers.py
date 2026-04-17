from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import requests

from watchlist_manager import load_watchlist


STATIC_MARKET_TICKERS = [
    {"ticker": "AAPL", "company_name_ko": "애플", "company_name_en": "Apple", "market": "NASDAQ"},
    {"ticker": "MSFT", "company_name_ko": "마이크로소프트", "company_name_en": "Microsoft", "market": "NASDAQ"},
    {"ticker": "NVDA", "company_name_ko": "엔비디아", "company_name_en": "NVIDIA", "market": "NASDAQ"},
    {"ticker": "TSLA", "company_name_ko": "테슬라", "company_name_en": "Tesla", "market": "NASDAQ"},
    {"ticker": "JPM", "company_name_ko": "JP모건", "company_name_en": "JPMorgan Chase", "market": "NYSE"},
    {"ticker": "BA", "company_name_ko": "보잉", "company_name_en": "Boeing", "market": "NYSE"},
    {"ticker": "005930", "company_name_ko": "삼성전자", "company_name_en": "Samsung Electronics", "market": "KOSPI"},
    {"ticker": "000660", "company_name_ko": "SK하이닉스", "company_name_en": "SK hynix", "market": "KOSPI"},
    {"ticker": "035420", "company_name_ko": "NAVER", "company_name_en": "NAVER", "market": "KOSPI"},
    {"ticker": "035720", "company_name_ko": "카카오", "company_name_en": "Kakao", "market": "KOSDAQ"},
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
                # Finnhub may return either `mic` or legacy `micExchange`; support both for compatibility.
                market = _map_us_market(str(item.get("mic") or item.get("micExchange") or ""))
                if market not in {"NASDAQ", "NYSE"}:
                    continue
            else:
                market = "KOSPI" if ticker.endswith(".KS") else "KOSDAQ" if ticker.endswith(".KQ") else "KRX"
                ticker = ticker.split(".")[0]
            merged[ticker] = {
                "ticker": ticker,
                "company_name_ko": "",
                "company_name_en": str(item.get("description") or ticker),
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
            "company_name_ko": str(row.get("company_name_ko", "")),
            "company_name_en": str(row.get("company_name_en", row.get("company_name", ticker))),
            "market": str(row.get("exchange", row.get("market", "UNKNOWN"))).upper(),
        }

    for row in STATIC_MARKET_TICKERS + _finnhub_symbols():
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue
        if ticker not in merged:
            company_name_ko = str(row.get("company_name_ko", ""))
            company_name_en = str(row.get("company_name_en", row.get("company_name", ticker)))
            merged[ticker] = {
                "ticker": ticker,
                "company_name": company_name_ko or company_name_en or ticker,
                "company_name_ko": company_name_ko,
                "company_name_en": company_name_en,
                "market": str(row.get("market", "UNKNOWN")).upper(),
            }

    return sorted(merged.values(), key=lambda x: x["ticker"])
