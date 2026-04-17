from __future__ import annotations

import argparse
import io
import logging
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests
import yfinance as yf
from watchlist_manager import load_watchlist

try:
    from pykrx import stock  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    stock = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT / "config" / "market_db.csv"
LOGGER = logging.getLogger(__name__)
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
OTHER_LISTED_ACT_SYMBOL_COLUMN = "ACT Symbol"


def _non_negative_float(raw: str) -> float:
    return max(0.0, float(raw))


def _read_pipe_separated(url: str) -> pd.DataFrame:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        LOGGER.warning("Failed to fetch %s", url)
        return pd.DataFrame()
    rows = [line for line in response.text.splitlines() if line and "File Creation Time" not in line]
    if not rows:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO("\n".join(rows)), sep="|")


def _clean_us_ticker(raw: object) -> str:
    ticker = str(raw or "").strip().upper()
    if not ticker:
        return ""
    if "$" in ticker:
        return ""
    return ticker.replace(".", "-")


def _load_nasdaq_rows() -> List[Dict[str, str]]:
    frame = _read_pipe_separated(NASDAQ_LISTED_URL)
    if frame.empty:
        return []
    out: List[Dict[str, str]] = []
    for row in frame.to_dict(orient="records"):
        ticker = _clean_us_ticker(row.get("Symbol"))
        if not ticker:
            continue
        out.append(
            {
                "ticker": ticker,
                "company_name_ko": "",
                "company_name_en": str(row.get("Security Name") or ticker).strip(),
                "exchange": "NASDAQ",
                "sector": "Unknown",
                "sub_sector": "Unknown",
            }
        )
    return out


def _load_nyse_rows() -> List[Dict[str, str]]:
    frame = _read_pipe_separated(OTHER_LISTED_URL)
    if frame.empty:
        return []
    out: List[Dict[str, str]] = []
    for row in frame.to_dict(orient="records"):
        if str(row.get("Exchange") or "").strip().upper() != "N":
            continue
        ticker = _clean_us_ticker(row.get(OTHER_LISTED_ACT_SYMBOL_COLUMN))
        if not ticker:
            continue
        out.append(
            {
                "ticker": ticker,
                "company_name_ko": "",
                "company_name_en": str(row.get("Security Name") or ticker).strip(),
                "exchange": "NYSE",
                "sector": "Unknown",
                "sub_sector": "Unknown",
            }
        )
    return out


def _load_krx_rows(market: str) -> List[Dict[str, str]]:
    if stock is None:
        LOGGER.warning("pykrx is not installed. Skipping %s.", market)
        return []
    try:
        tickers = stock.get_market_ticker_list(market=market)
    except (
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        JSONDecodeError,
        requests.RequestException,
    ) as exc:
        LOGGER.warning("Failed to fetch KRX ticker list for %s: %s", market, exc)
        return []
    except Exception as exc:  # pragma: no cover - pykrx can raise non-standard runtime errors
        LOGGER.warning("Unexpected pykrx failure while loading %s: %s", market, exc)
        return []
    out: List[Dict[str, str]] = []
    for ticker in tickers:
        try:
            name = stock.get_market_ticker_name(ticker) or ticker
        except (TypeError, ValueError) as exc:
            LOGGER.warning("Failed to resolve KRX ticker name for %s (%s): %s", ticker, market, exc)
            continue
        out.append(
            {
                "ticker": str(ticker).zfill(6),
                "company_name_ko": str(name).strip(),
                "company_name_en": "",
                "exchange": market.upper(),
                "sector": "Unknown",
                "sub_sector": "Unknown",
            }
        )
    return out


def _load_watchlist_rows() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in load_watchlist().to_dict(orient="records"):
        ticker = _clean_us_ticker(row.get("ticker"))
        exchange = str(row.get("exchange") or row.get("market") or "").upper().strip()
        if not ticker:
            continue
        if exchange in {"KOSPI", "KOSDAQ"} and ticker.isdigit():
            ticker = ticker.zfill(6)
        out.append(
            {
                "ticker": ticker,
                "company_name_ko": str(row.get("company_name_ko") or "").strip(),
                "company_name_en": str(row.get("company_name_en") or row.get("company_name") or ticker).strip(),
                "exchange": exchange or "UNKNOWN",
                "sector": str(row.get("sector") or "Unknown").strip() or "Unknown",
                "sub_sector": str(row.get("sub_sector") or "Unknown").strip() or "Unknown",
            }
        )
    return out


def _collect_rows(stage_name: str, loader) -> List[Dict[str, str]]:
    LOGGER.info("📥 %s 종목 수집 중...", stage_name)
    try:
        rows = loader()
    except Exception as exc:  # pragma: no cover - defensive catch for unstable external APIs
        LOGGER.warning("%s 수집 실패: %s", stage_name, exc)
        rows = []
    LOGGER.info("  ✅ %s: %s개", stage_name, len(rows))
    return rows


def _enrich_sectors(rows: Iterable[Dict[str, str]], delay_seconds: float) -> List[Dict[str, str]]:
    enriched: List[Dict[str, str]] = []
    for row in rows:
        exchange = str(row.get("exchange", "")).upper()
        if exchange not in {"NASDAQ", "NYSE"}:
            enriched.append(row)
            continue
        ticker = str(row.get("ticker", "")).upper()
        try:
            # yfinance may intermittently return malformed payloads depending on symbol/state.
            info = yf.Ticker(ticker).info or {}
        except (KeyError, TypeError, ValueError, AttributeError, requests.RequestException) as exc:
            LOGGER.warning("Failed to fetch yfinance info for %s: %s", ticker, exc)
            info = {}
        row["sector"] = str(info.get("sector") or row.get("sector") or "").strip()
        row["sub_sector"] = str(info.get("industry") or row.get("sub_sector") or "").strip()
        enriched.append(row)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return enriched


def build_market_db(output_path: Path, enrich_sectors: bool, delay_seconds: float) -> pd.DataFrame:
    merged: Dict[str, Dict[str, str]] = {}

    all_rows = (
        _collect_rows("NASDAQ", _load_nasdaq_rows)
        + _collect_rows("NYSE", _load_nyse_rows)
        + _collect_rows("KOSPI", lambda: _load_krx_rows("KOSPI"))
        + _collect_rows("KOSDAQ", lambda: _load_krx_rows("KOSDAQ"))
        + _collect_rows("WATCHLIST fallback", _load_watchlist_rows)
    )

    for row in all_rows:
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        if ticker not in merged:
            merged[ticker] = row
            continue
        current = merged[ticker]
        for key in ("company_name_ko", "company_name_en", "exchange", "sector", "sub_sector"):
            existing = str(current.get(key, "") or "").strip()
            incoming = str(row.get(key, "") or "").strip()
            if not existing or existing in {"Unknown", "UNKNOWN"}:
                if incoming:
                    current[key] = incoming

    rows = list(merged.values())
    if enrich_sectors:
        rows = _enrich_sectors(rows, delay_seconds=delay_seconds)

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["ticker", "company_name_ko", "company_name_en", "exchange", "sector", "sub_sector"])
    else:
        frame = frame.sort_values(["exchange", "ticker"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    LOGGER.info("\n✅ 완료! 총 %s개 종목", len(frame))
    LOGGER.info("📁 저장: %s", output_path)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified market DB CSV for NASDAQ/NYSE/KOSPI/KOSDAQ.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV path")
    parser.add_argument("--enrich-sectors", action="store_true", help="Fetch sector/sub-sector from yfinance info")
    parser.add_argument("--delay-seconds", type=_non_negative_float, default=0.1, help="Delay between yfinance info calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frame = build_market_db(args.output, enrich_sectors=args.enrich_sectors, delay_seconds=args.delay_seconds)
    counts = frame["exchange"].value_counts().to_dict() if not frame.empty else {}
    LOGGER.info("✅ NASDAQ: %s", counts.get("NASDAQ", 0))
    LOGGER.info("✅ NYSE: %s", counts.get("NYSE", 0))
    LOGGER.info("✅ KOSPI: %s", counts.get("KOSPI", 0))
    LOGGER.info("✅ KOSDAQ: %s", counts.get("KOSDAQ", 0))
    LOGGER.info("Total %s tickers in market DB -> %s", len(frame), args.output)


if __name__ == "__main__":
    main()
