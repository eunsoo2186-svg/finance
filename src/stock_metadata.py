from __future__ import annotations

from typing import Dict

DEFAULT_SECTOR_HIERARCHY: Dict[str, Dict[str, list[str]]] = {
    "AI": {
        "LLM": ["MSFT", "GOOGL", "META", "AMZN"],
        "반도체": ["NVDA", "AMD", "TSM", "INTC"],
        "컴퓨터 비전": ["TSLA", "ADSK"],
    },
    "SPACE": {
        "항공우주": ["BA", "LMT", "NOC", "RTX"],
        "우주 인프라": ["RKLB", "SPCE", "PL"],
    },
    "MARKET": {
        "기타": [],
    },
}

BASE_METADATA: Dict[str, Dict[str, str]] = {
    "MSFT": {"company_name": "Microsoft", "sector": "AI", "sub_sector": "LLM", "market": "NASDAQ"},
    "NVDA": {"company_name": "NVIDIA", "sector": "AI", "sub_sector": "반도체", "market": "NASDAQ"},
    "GOOGL": {"company_name": "Alphabet", "sector": "AI", "sub_sector": "LLM", "market": "NASDAQ"},
    "TSLA": {"company_name": "Tesla", "sector": "AI", "sub_sector": "컴퓨터 비전", "market": "NASDAQ"},
    "RTX": {"company_name": "RTX", "sector": "SPACE", "sub_sector": "항공우주", "market": "NYSE"},
    "LMT": {"company_name": "Lockheed Martin", "sector": "SPACE", "sub_sector": "항공우주", "market": "NYSE"},
    "NOC": {"company_name": "Northrop Grumman", "sector": "SPACE", "sub_sector": "항공우주", "market": "NYSE"},
    "BA": {"company_name": "Boeing", "sector": "SPACE", "sub_sector": "항공우주", "market": "NYSE"},
}


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
    symbol = symbol.upper()
    info = info or {}

    known = BASE_METADATA.get(symbol)
    if known:
        return {"ticker": symbol, **known}

    company_name = str(info.get("longName") or info.get("shortName") or symbol)
    inferred_sector = str(info.get("sector") or "MARKET")
    inferred_sub_sector = str(info.get("industry") or "기타")
    market = _market_from_exchange(str(info.get("exchange") or info.get("fullExchangeName") or ""))

    if inferred_sector in {"Technology", "Communication Services", "Information Technology"}:
        sector = "AI"
    elif inferred_sector in {"Aerospace & Defense", "Industrials", "Defense"}:
        sector = "SPACE"
    else:
        sector = "MARKET"

    return {
        "ticker": symbol,
        "company_name": company_name,
        "sector": sector,
        "sub_sector": inferred_sub_sector,
        "market": market,
    }
