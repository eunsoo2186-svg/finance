from __future__ import annotations

from pathlib import Path
from typing import Dict

from watchlist_manager import load_watchlist

DEFAULT_SECTOR_HIERARCHY: Dict[str, Dict[str, list[str]]] = {
    "AI": {
        "LLM": ["MSFT", "GOOGL", "META", "AMZN"],
        "Semiconductors": ["NVDA", "AMD", "TSM", "INTC"],
        "Computer Vision": ["TSLA", "ADSK"],
    },
    "SPACE": {
        "Aerospace": ["BA", "LMT", "NOC", "RTX"],
        "Space Infrastructure": ["RKLB", "SPCE", "PL"],
    },
    "MARKET": {
        "General": [],
    },
}

AI_SECTORS = {"Technology", "Communication Services", "Information Technology"}
SPACE_SECTORS = {"Aerospace & Defense", "Industrials", "Defense"}

BASE_METADATA: Dict[str, Dict[str, str]] = {
    "MSFT": {"company_name": "Microsoft", "sector": "AI", "sub_sector": "LLM", "market": "NASDAQ"},
    "NVDA": {"company_name": "NVIDIA", "sector": "AI", "sub_sector": "Semiconductors", "market": "NASDAQ"},
    "GOOGL": {"company_name": "Alphabet", "sector": "AI", "sub_sector": "LLM", "market": "NASDAQ"},
    "TSLA": {"company_name": "Tesla", "sector": "AI", "sub_sector": "Computer Vision", "market": "NASDAQ"},
    "RTX": {"company_name": "RTX", "sector": "SPACE", "sub_sector": "Aerospace", "market": "NYSE"},
    "LMT": {"company_name": "Lockheed Martin", "sector": "SPACE", "sub_sector": "Aerospace", "market": "NYSE"},
    "NOC": {"company_name": "Northrop Grumman", "sector": "SPACE", "sub_sector": "Aerospace", "market": "NYSE"},
    "BA": {"company_name": "Boeing", "sector": "SPACE", "sub_sector": "Aerospace", "market": "NYSE"},
}


def _build_stock_metadata() -> Dict[str, Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for row in load_watchlist().to_dict(orient="records"):
        ticker = _normalize_symbol(str(row.get("ticker", "")))
        if not ticker:
            continue
        merged[ticker] = {
            "name_ko": str(row.get("company_name_ko") or ""),
            "name_en": str(row.get("company_name_en") or row.get("company_name") or ticker),
            "sector": str(row.get("sector") or "Unknown"),
            "sub_sector": str(row.get("sub_sector") or "Unknown"),
            "market": str(row.get("exchange") or row.get("market") or "UNKNOWN").upper(),
        }

    for ticker, payload in BASE_METADATA.items():
        if ticker not in merged:
            merged[ticker] = {
                "name_ko": "",
                "name_en": payload.get("company_name", ticker),
                "sector": payload.get("sector", "Unknown"),
                "sub_sector": payload.get("sub_sector", "Unknown"),
                "market": payload.get("market", "UNKNOWN"),
            }
    return merged


def _normalize_symbol(symbol: str) -> str:
    symbol = symbol.upper().strip()
    if symbol.endswith(".KS") or symbol.endswith(".KQ"):
        return symbol.split(".")[0]
    return symbol


STOCK_METADATA: Dict[str, Dict[str, str]] = _build_stock_metadata()


_WATCHLIST_CACHE: Dict[str, object] = {"mtime": None, "data": {}}


def _watchlist_metadata_map() -> Dict[str, Dict[str, str]]:
    watchlist_path = Path(__file__).resolve().parents[1] / "config" / "watchlist.csv"
    mtime = watchlist_path.stat().st_mtime if watchlist_path.exists() else None
    if _WATCHLIST_CACHE["mtime"] == mtime:
        return _WATCHLIST_CACHE["data"]  # type: ignore[return-value]

    mapping: Dict[str, Dict[str, str]] = {}
    for row in load_watchlist().to_dict(orient="records"):
        ticker = _normalize_symbol(str(row.get("ticker", "")))
        if not ticker:
            continue
        mapping[ticker] = {
            "company_name": str(row.get("company_name") or ticker),
            "company_name_ko": str(row.get("company_name_ko") or ""),
            "company_name_en": str(row.get("company_name_en") or row.get("company_name") or ticker),
            "sector": str(row.get("sector") or "MARKET"),
            "sub_sector": str(row.get("sub_sector") or "General"),
            "market": str(row.get("exchange") or row.get("market") or "UNKNOWN").upper(),
        }
    _WATCHLIST_CACHE["mtime"] = mtime
    _WATCHLIST_CACHE["data"] = mapping
    return mapping


def _market_from_exchange(exchange: str | None) -> str:
    exchange = (exchange or "").upper()
    if "KOSPI" in exchange:
        return "KOSPI"
    if "KOSDAQ" in exchange:
        return "KOSDAQ"
    if exchange in {"NMS", "NAS", "NASDAQ", "XNAS"}:
        return "NASDAQ"
    if exchange in {"NYQ", "NYS", "NYSE", "XNYS"}:
        return "NYSE"
    return "UNKNOWN"


def metadata_for_ticker(symbol: str, info: Dict[str, object] | None = None) -> Dict[str, str]:
    symbol = _normalize_symbol(symbol)
    info = info or {}

    watchlist_meta = _watchlist_metadata_map().get(symbol)
    if watchlist_meta:
        return {"ticker": symbol, **watchlist_meta}

    known = BASE_METADATA.get(symbol)
    if known:
        return {
            "ticker": symbol,
            **known,
            "company_name_ko": "",
            "company_name_en": known.get("company_name", symbol),
        }

    company_name = str(info.get("longName") or info.get("shortName") or symbol)
    inferred_sector = str(info.get("sector") or "MARKET")
    inferred_sub_sector = str(info.get("industry") or "General")
    market = _market_from_exchange(str(info.get("exchange") or info.get("fullExchangeName") or ""))

    if inferred_sector in AI_SECTORS:
        sector = "AI"
    elif inferred_sector in SPACE_SECTORS:
        sector = "SPACE"
    else:
        sector = "MARKET"

    return {
        "ticker": symbol,
        "company_name": company_name,
        "company_name_ko": "",
        "company_name_en": company_name,
        "sector": sector,
        "sub_sector": inferred_sub_sector,
        "market": market,
    }
