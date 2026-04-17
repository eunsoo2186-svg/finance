from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests
import yfinance as yf

try:
    from pykrx import stock  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    stock = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT / "config" / "market_db.csv"
LOGGER = logging.getLogger(__name__)
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


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
                "sector": "",
                "sub_sector": "",
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
        ticker = _clean_us_ticker(row.get("ACT Symbol"))
        if not ticker:
            continue
        out.append(
            {
                "ticker": ticker,
                "company_name_ko": "",
                "company_name_en": str(row.get("Security Name") or ticker).strip(),
                "exchange": "NYSE",
                "sector": "",
                "sub_sector": "",
            }
        )
    return out


def _load_krx_rows(market: str) -> List[Dict[str, str]]:
    if stock is None:
        LOGGER.warning("pykrx is not installed. Skipping %s.", market)
        return []
    tickers = stock.get_market_ticker_list(market=market)
    out: List[Dict[str, str]] = []
    for ticker in tickers:
        try:
            name = stock.get_market_ticker_name(ticker) or ticker
        except Exception:
            continue
        out.append(
            {
                "ticker": str(ticker).zfill(6),
                "company_name_ko": str(name).strip(),
                "company_name_en": "",
                "exchange": market.upper(),
                "sector": "",
                "sub_sector": "",
            }
        )
    return out


def _enrich_sectors(rows: Iterable[Dict[str, str]], delay_seconds: float) -> List[Dict[str, str]]:
    enriched: List[Dict[str, str]] = []
    for row in rows:
        exchange = str(row.get("exchange", "")).upper()
        if exchange not in {"NASDAQ", "NYSE"}:
            enriched.append(row)
            continue
        ticker = str(row.get("ticker", "")).upper()
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        row["sector"] = str(info.get("sector") or row.get("sector") or "").strip()
        row["sub_sector"] = str(info.get("industry") or row.get("sub_sector") or "").strip()
        enriched.append(row)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return enriched


def build_market_db(output_path: Path, enrich_sectors: bool, delay_seconds: float) -> pd.DataFrame:
    merged: Dict[str, Dict[str, str]] = {}

    for row in _load_nasdaq_rows() + _load_nyse_rows() + _load_krx_rows("KOSPI") + _load_krx_rows("KOSDAQ"):
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        merged[ticker] = row

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
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified market DB CSV for NASDAQ/NYSE/KOSPI/KOSDAQ.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV path")
    parser.add_argument("--enrich-sectors", action="store_true", help="Fetch sector/sub-sector from yfinance info")
    parser.add_argument("--delay-seconds", type=float, default=0.1, help="Delay between yfinance info calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frame = build_market_db(args.output, enrich_sectors=args.enrich_sectors, delay_seconds=max(0.0, args.delay_seconds))
    counts = frame["exchange"].value_counts().to_dict() if not frame.empty else {}
    LOGGER.info("✅ NASDAQ: %s", counts.get("NASDAQ", 0))
    LOGGER.info("✅ NYSE: %s", counts.get("NYSE", 0))
    LOGGER.info("✅ KOSPI: %s", counts.get("KOSPI", 0))
    LOGGER.info("✅ KOSDAQ: %s", counts.get("KOSDAQ", 0))
    LOGGER.info("총 %s 종목 DB 구축 완료! -> %s", len(frame), args.output)


if __name__ == "__main__":
    main()
