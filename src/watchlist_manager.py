from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WATCHLIST_PATH = ROOT / "config" / "watchlist.csv"
FAVORITES_PATH = ROOT / "config" / "favorites.json"
NOTES_PATH = ROOT / "config" / "notes.json"
HOLDINGS_PATH = ROOT / "config" / "holdings.json"


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_watchlist() -> pd.DataFrame:
    if not WATCHLIST_PATH.exists():
        default = pd.DataFrame(
            [
                {"ticker": "MSFT", "company_name": "Microsoft", "market": "NASDAQ"},
                {"ticker": "NVDA", "company_name": "NVIDIA", "market": "NASDAQ"},
                {"ticker": "GOOGL", "company_name": "Alphabet", "market": "NASDAQ"},
                {"ticker": "TSLA", "company_name": "Tesla", "market": "NASDAQ"},
                {"ticker": "RTX", "company_name": "RTX", "market": "NYSE"},
                {"ticker": "LMT", "company_name": "Lockheed Martin", "market": "NYSE"},
                {"ticker": "NOC", "company_name": "Northrop Grumman", "market": "NYSE"},
                {"ticker": "BA", "company_name": "Boeing", "market": "NYSE"},
            ]
        )
        WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        default.to_csv(WATCHLIST_PATH, index=False)
        return default

    df = pd.read_csv(WATCHLIST_PATH)
    if "ticker" not in df.columns:
        return pd.DataFrame(columns=["ticker", "company_name", "market"])

    if "company_name" not in df.columns:
        df["company_name"] = df["ticker"]
    if "market" not in df.columns:
        df["market"] = "UNKNOWN"

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["company_name"] = df["company_name"].astype(str).str.strip()
    df["market"] = df["market"].astype(str).str.upper().str.strip()
    return df[["ticker", "company_name", "market"]].dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])


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


def load_holdings() -> Dict[str, Dict[str, object]]:
    raw = _read_json(HOLDINGS_PATH)
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for ticker, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        out[str(ticker).upper()] = {
            "purchase_price": float(payload.get("purchase_price", 0) or 0),
            "quantity": float(payload.get("quantity", 0) or 0),
            "currency": str(payload.get("currency", "USD") or "USD"),
        }
    return out


def save_holdings(holdings: Dict[str, Dict[str, object]]) -> None:
    _write_json(HOLDINGS_PATH, holdings)
