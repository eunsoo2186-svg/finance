from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WATCHLIST_PATH = ROOT / "config" / "watchlist.csv"
FAVORITES_PATH = ROOT / "config" / "favorites.json"
NOTES_PATH = ROOT / "config" / "notes.json"
HOLDINGS_PATH = ROOT / "config" / "holdings.json"
LOGGER = logging.getLogger(__name__)
WATCHLIST_COLUMNS = [
    "ticker",
    "company_name",
    "company_name_ko",
    "company_name_en",
    "market",
    "exchange",
    "sector",
    "sub_sector",
]


def _normalize_ticker(raw: str) -> str:
    ticker = str(raw or "").upper().strip()
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        return ticker.split(".")[0]
    return ticker


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("JSON decode failed for %s. Falling back to empty payload.", path)
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_watchlist() -> pd.DataFrame:
    if not WATCHLIST_PATH.exists():
        default = pd.DataFrame(
            [
                {
                    "ticker": "MSFT",
                    "company_name_ko": "마이크로소프트",
                    "company_name_en": "Microsoft",
                    "exchange": "NASDAQ",
                    "sector": "Technology",
                    "sub_sector": "Software",
                },
                {
                    "ticker": "TSLA",
                    "company_name_ko": "테슬라",
                    "company_name_en": "Tesla",
                    "exchange": "NASDAQ",
                    "sector": "Technology",
                    "sub_sector": "Electric Vehicles",
                },
            ]
        )
        WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        default.to_csv(WATCHLIST_PATH, index=False)
        return load_watchlist()

    df = pd.read_csv(WATCHLIST_PATH)
    if "ticker" not in df.columns:
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)

    if "exchange" not in df.columns and "market" in df.columns:
        df["exchange"] = df["market"]
    if "company_name_en" not in df.columns and "company_name" in df.columns:
        df["company_name_en"] = df["company_name"]
    if "company_name_ko" not in df.columns:
        df["company_name_ko"] = ""

    df["ticker"] = df["ticker"].apply(_normalize_ticker)
    df["company_name_ko"] = df["company_name_ko"].fillna("").astype(str).str.strip()
    df["company_name_en"] = df["company_name_en"].fillna("").astype(str).str.strip()
    df["exchange"] = df["exchange"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    df["market"] = df["exchange"]
    df["company_name"] = (
        df["company_name_ko"]
        .where(df["company_name_ko"].str.len() > 0, df["company_name_en"])
        .where(lambda s: s.str.len() > 0, df["ticker"])
    )
    if "sector" not in df.columns:
        df["sector"] = "MARKET"
    if "sub_sector" not in df.columns:
        df["sub_sector"] = "General"
    df["sector"] = df["sector"].fillna("MARKET").astype(str).str.strip()
    df["sub_sector"] = df["sub_sector"].fillna("General").astype(str).str.strip()

    return (
        df[WATCHLIST_COLUMNS]
        .dropna(subset=["ticker"])
        .drop_duplicates(subset=["ticker"])
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def load_watchlist_tickers() -> List[str]:
    df = load_watchlist()
    return df["ticker"].tolist()


def load_favorites() -> List[str]:
    raw = _read_json(FAVORITES_PATH)
    favorites = raw.get("favorites", []) if isinstance(raw, dict) else []
    return sorted({str(t).upper() for t in favorites})


def save_favorites(tickers: List[str]) -> None:
    _write_json(FAVORITES_PATH, {"favorites": sorted({str(t).upper() for t in tickers})})


def load_notes() -> Dict[str, str]:
    raw = _read_json(NOTES_PATH)
    if not isinstance(raw, dict):
        return {}
    return {str(k).upper(): str(v) for k, v in raw.items()}


def save_notes(notes: Dict[str, str]) -> None:
    clean = {str(k).upper(): str(v) for k, v in notes.items()}
    _write_json(NOTES_PATH, clean)


def load_holdings() -> Dict[str, Dict[str, Any]]:
    raw = _read_json(HOLDINGS_PATH)
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for ticker, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        out[str(ticker).upper()] = {
            "purchase_price": float(payload.get("purchase_price", 0) or 0),
            "quantity": float(payload.get("quantity", 0) or 0),
            "currency": str(payload.get("currency", "USD") or "USD"),
            "target_price": float(payload.get("target_price", 0) or 0),
        }
    return out


def save_holdings(holdings: Dict[str, Dict[str, Any]]) -> None:
    _write_json(HOLDINGS_PATH, holdings)
