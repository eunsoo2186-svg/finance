from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache, wraps
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from pandas.errors import EmptyDataError, ParserError

from stock_metadata import DEFAULT_SECTOR_HIERARCHY, STOCK_METADATA, metadata_for_ticker
from watchlist_manager import load_watchlist, load_watchlist_tickers

ENV_PATH = Path(__file__).resolve().parents[1] / "config" / ".env"
MARKET_DB_PATH = Path(__file__).resolve().parents[1] / "config" / "market_db.csv"
load_dotenv(ENV_PATH)
LOGGER = logging.getLogger(__name__)
KRX_PRICE_LOOKBACK_PERIOD_DAYS = 7
KR_STOCK_METRICS: Dict[str, Dict[str, float]] = {
    "005930": {"pe_ratio": 12.5, "peg_ratio": 0.8},
    "000660": {"pe_ratio": 14.2, "peg_ratio": 0.9},
}

AI_TICKERS = ["MSFT", "NVDA", "GOOGL", "TSLA"]
SPACE_TICKERS = ["RTX", "LMT", "NOC", "BA"]
MARKET_TICKERS = load_watchlist_tickers()

SECTOR_TICKERS = {
    "AI": AI_TICKERS,
    "SPACE": SPACE_TICKERS,
    "MARKET": MARKET_TICKERS,
}

SECTOR_HIERARCHY = DEFAULT_SECTOR_HIERARCHY


def ttl_cache(maxsize: int, ttl: int):
    """TTL 기반 캐시 데코레이터."""
    cache_dict: OrderedDict[tuple, tuple[Any, float]] = OrderedDict()

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in cache_dict:
                result, expiry = cache_dict[key]
                if now < expiry:
                    cache_dict.move_to_end(key)
                    return result
                del cache_dict[key]

            result = func(*args, **kwargs)
            cache_dict[key] = (result, now + ttl)
            cache_dict.move_to_end(key)
            while len(cache_dict) > maxsize:
                cache_dict.popitem(last=False)
            return result

        def cache_clear():
            cache_dict.clear()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


_RUN_SECTOR_AVERAGE_CACHE: Dict[tuple[str, str], float] = {}


def _set_run_sector_metric_averages(rows: List[Dict[str, object]]) -> None:
    global _RUN_SECTOR_AVERAGE_CACHE
    grouped: Dict[tuple[str, str], List[float]] = {}
    for row in rows:
        sector = str(row.get("score_sector", "MARKET"))
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        specs = METRIC_SPECS.get(sector, {})
        for metric_name in specs:
            raw_value = metrics.get(metric_name)
            if raw_value is None:
                continue
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                continue
            if pd.isna(numeric):
                continue
            grouped.setdefault((sector, metric_name), []).append(numeric)

    _RUN_SECTOR_AVERAGE_CACHE = {
        key: float(sum(values) / len(values))
        for key, values in grouped.items()
        if values
    }


def _sector_metric_average(sector: str, metric_name: str) -> float | None:
    return _RUN_SECTOR_AVERAGE_CACHE.get((sector, metric_name))


def _normalize_market_label(raw_exchange: object) -> str:
    exchange = str(raw_exchange or "").upper().strip()
    if exchange in {"NMS", "NAS", "NASDAQ", "XNAS"}:
        return "NASDAQ"
    if exchange in {"NYQ", "NYS", "NYSE", "XNYS"}:
        return "NYSE"
    if "KOSDAQ" in exchange or exchange == "KQ":
        return "KOSDAQ"
    if "KOSPI" in exchange or exchange == "KS":
        return "KOSPI"
    return exchange if exchange else "UNKNOWN"


def _normalize_csv_ticker(raw_ticker: object, exchange: str) -> str:
    ticker = str(raw_ticker or "").upper().strip()
    if not ticker:
        return ""
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        ticker = ticker.split(".")[0]
    if exchange in {"KOSPI", "KOSDAQ"} and ticker.isdigit():
        return ticker.zfill(6)
    return ticker


def _rows_from_market_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path, dtype={"ticker": str})
    except (ParserError, EmptyDataError, OSError) as exc:
        LOGGER.warning("Failed to read market DB CSV at %s (%s): %s", path, exc.__class__.__name__, exc)
        return []
    if frame.empty:
        return []

    if "exchange" not in frame.columns and "market" in frame.columns:
        frame["exchange"] = frame["market"]
    if "company_name_en" not in frame.columns and "company_name" in frame.columns:
        frame["company_name_en"] = frame["company_name"]
    if "company_name_ko" not in frame.columns:
        frame["company_name_ko"] = ""
    if "sector" not in frame.columns:
        frame["sector"] = "MARKET"
    if "sub_sector" not in frame.columns:
        frame["sub_sector"] = "General"
    if "industry" not in frame.columns:
        frame["industry"] = frame["sub_sector"].fillna("General")

    rows_by_ticker: Dict[str, Dict[str, str]] = {}
    for row in frame.to_dict(orient="records"):
        exchange = _normalize_market_label(row.get("exchange", "UNKNOWN"))
        ticker = _normalize_csv_ticker(row.get("ticker", ""), exchange)
        if not ticker:
            continue
        company_name_ko = str(row.get("company_name_ko", "") or "").strip()
        company_name_en = str(row.get("company_name_en", "") or "").strip()
        rows_by_ticker[ticker] = {
            "ticker": ticker,
            "company_name_ko": company_name_ko,
            "company_name_en": company_name_en or ticker,
            "exchange": exchange,
            "sector": str(row.get("sector", "MARKET") or "MARKET").strip() or "MARKET",
            "sub_sector": str(row.get("sub_sector", "General") or "General").strip() or "General",
            "industry": str(row.get("industry", row.get("sub_sector", "General")) or "General").strip() or "General",
        }
    return list(rows_by_ticker.values())


@lru_cache(maxsize=1)
def load_all_tickers() -> List[Dict[str, str]]:
    csv_rows = _rows_from_market_csv(MARKET_DB_PATH)
    if csv_rows:
        return sorted(csv_rows, key=lambda x: x["ticker"])

    watchlist = load_watchlist()
    watchlist_rows_by_ticker: Dict[str, Dict[str, str]] = {}
    for row in watchlist.to_dict(orient="records"):
        exchange = _normalize_market_label(row.get("exchange", row.get("market", "UNKNOWN")))
        ticker = _normalize_csv_ticker(row.get("ticker", ""), exchange)
        if not ticker:
            continue
        watchlist_rows_by_ticker[ticker] = {
            "ticker": ticker,
            "company_name_ko": str(row.get("company_name_ko", "") or "").strip(),
            "company_name_en": str(row.get("company_name_en", row.get("company_name", ticker)) or ticker).strip(),
            "exchange": exchange,
            "sector": str(row.get("sector", "MARKET") or "MARKET").strip() or "MARKET",
            "sub_sector": str(row.get("sub_sector", "General") or "General").strip() or "General",
            "industry": str(row.get("industry", row.get("sub_sector", "General")) or "General").strip() or "General",
        }
    return sorted(watchlist_rows_by_ticker.values(), key=lambda x: x["ticker"])


@dataclass(frozen=True)
class MetricSpec:
    label: str
    min_value: float
    max_value: float
    higher_is_better: bool
    source: str
    formula: str


METRIC_SPECS: Dict[str, Dict[str, MetricSpec]] = {
    "AI": {
        "pe_ratio": MetricSpec("P/E Ratio", 8, 60, False, "yfinance.info.trailingPE", "trailingPE"),
        "peg_ratio": MetricSpec("PEG Ratio", 0.3, 3.0, False, "yfinance.info.pegRatio", "pegRatio"),
        "revenue_growth_yoy": MetricSpec(
            "YoY Revenue Growth", -0.2, 0.5, True, "yfinance.info.revenueGrowth", "revenueGrowth"
        ),
        "rd_ratio": MetricSpec(
            "R&D / Revenue",
            0.02,
            0.35,
            True,
            "yfinance.financials + info.totalRevenue",
            "researchDevelopment / totalRevenue",
        ),
        "asset_turnover": MetricSpec(
            "Asset Turnover", 0.1, 1.5, True, "yfinance.info.totalRevenue,totalAssets", "totalRevenue / totalAssets"
        ),
        "tech_cycle": MetricSpec(
            "Tech Cycle Position", -0.4, 0.8, True, "finnhub.stock.metric.52WeekPriceReturnDaily", "52WeekPriceReturnDaily"
        ),
    },
    "SPACE": {
        "pb_ratio": MetricSpec("P/B Ratio", 0.5, 8.0, False, "yfinance.info.priceToBook", "priceToBook"),
        "operating_margin": MetricSpec(
            "Operating Margin", -0.1, 0.3, True, "yfinance.info.operatingMargins", "operatingMargins"
        ),
        "debt_ratio": MetricSpec(
            "Debt Ratio", 0.1, 3.0, False, "yfinance.info.debtToEquity", "debtToEquity / 100"
        ),
        "dividend_yield": MetricSpec(
            "Dividend Yield", 0.0, 0.08, True, "yfinance.info.dividendYield", "dividendYield"
        ),
        "backlog_proxy": MetricSpec(
            "Contract Pipeline (Proxy)",
            -0.2,
            0.5,
            True,
            "yfinance.quarterly_income_stmt",
            "(sum of recent 4 quarters revenue / sum of previous 4 quarters revenue) - 1",
        ),
        "gov_cycle": MetricSpec(
            "Gov Spending Cycle (Proxy)", -0.3, 0.4, True, "finnhub.stock.metric.52WeekPriceReturnDaily", "52WeekPriceReturnDaily"
        ),
    },
    "MARKET": {
        "pe_ratio": MetricSpec("P/E Ratio", 8, 60, False, "yfinance.info.trailingPE", "trailingPE"),
        "peg_ratio": MetricSpec("PEG Ratio", 0.3, 3.0, False, "yfinance.info.pegRatio", "pegRatio"),
        "revenue_growth_yoy": MetricSpec(
            "YoY Revenue Growth", -0.2, 0.5, True, "yfinance.info.revenueGrowth", "revenueGrowth"
        ),
        "asset_turnover": MetricSpec(
            "Asset Turnover", 0.1, 1.5, True, "yfinance.info.totalRevenue,totalAssets", "totalRevenue / totalAssets"
        ),
        "debt_ratio": MetricSpec(
            "Debt Ratio", 0.1, 3.0, False, "yfinance.info.debtToEquity", "debtToEquity / 100"
        ),
        "dividend_yield": MetricSpec(
            "Dividend Yield", 0.0, 0.08, True, "yfinance.info.dividendYield", "dividendYield"
        ),
        "tech_cycle": MetricSpec(
            "Tech Cycle Position", -0.4, 0.8, True, "finnhub.stock.metric.52WeekPriceReturnDaily", "52WeekPriceReturnDaily"
        ),
    },
    "KOSPI": {
        "pe_ratio": MetricSpec("P/E Ratio", 4, 30, False, "yfinance.info.trailingPE", "trailingPE"),
        "peg_ratio": MetricSpec("PEG Ratio", 0.2, 2.0, False, "yfinance.info.pegRatio", "pegRatio"),
        "revenue_growth_yoy": MetricSpec(
            "YoY Revenue Growth", -0.2, 0.6, True, "yfinance.info.revenueGrowth", "revenueGrowth"
        ),
        "asset_turnover": MetricSpec(
            "Asset Turnover", 0.1, 1.5, True, "yfinance.info.totalRevenue,totalAssets", "totalRevenue / totalAssets"
        ),
        "debt_ratio": MetricSpec(
            "Debt Ratio", 0.1, 3.0, False, "yfinance.info.debtToEquity", "debtToEquity / 100"
        ),
        "dividend_yield": MetricSpec(
            "Dividend Yield", 0.0, 0.06, True, "yfinance.info.dividendYield", "dividendYield"
        ),
        "tech_cycle": MetricSpec(
            "Tech Cycle Position", -0.4, 0.8, True, "finnhub.stock.metric.52WeekPriceReturnDaily", "52WeekPriceReturnDaily"
        ),
    },
    "KOSDAQ": {
        "pe_ratio": MetricSpec("P/E Ratio", 4, 30, False, "yfinance.info.trailingPE", "trailingPE"),
        "peg_ratio": MetricSpec("PEG Ratio", 0.2, 2.0, False, "yfinance.info.pegRatio", "pegRatio"),
        "revenue_growth_yoy": MetricSpec(
            "YoY Revenue Growth", -0.1, 0.8, True, "yfinance.info.revenueGrowth", "revenueGrowth"
        ),
        "asset_turnover": MetricSpec(
            "Asset Turnover", 0.1, 1.5, True, "yfinance.info.totalRevenue,totalAssets", "totalRevenue / totalAssets"
        ),
        "debt_ratio": MetricSpec(
            "Debt Ratio", 0.1, 3.0, False, "yfinance.info.debtToEquity", "debtToEquity / 100"
        ),
        "dividend_yield": MetricSpec(
            "Dividend Yield", 0.0, 0.06, True, "yfinance.info.dividendYield", "dividendYield"
        ),
        "tech_cycle": MetricSpec(
            "Tech Cycle Position", -0.4, 0.8, True, "finnhub.stock.metric.52WeekPriceReturnDaily", "52WeekPriceReturnDaily"
        ),
    },
}

DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "AI": {
        "pe_ratio": 0.18,
        "peg_ratio": 0.20,
        "revenue_growth_yoy": 0.22,
        "rd_ratio": 0.18,
        "asset_turnover": 0.12,
        "tech_cycle": 0.10,
    },
    "SPACE": {
        "pb_ratio": 0.18,
        "operating_margin": 0.22,
        "debt_ratio": 0.18,
        "dividend_yield": 0.14,
        "backlog_proxy": 0.18,
        "gov_cycle": 0.10,
    },
    "MARKET": {
        "pe_ratio": 0.16,
        "peg_ratio": 0.16,
        "revenue_growth_yoy": 0.20,
        "asset_turnover": 0.16,
        "debt_ratio": 0.13,
        "dividend_yield": 0.10,
        "tech_cycle": 0.09,
    },
    "KOSPI": {
        "pe_ratio": 0.16,
        "peg_ratio": 0.16,
        "revenue_growth_yoy": 0.20,
        "asset_turnover": 0.16,
        "debt_ratio": 0.13,
        "dividend_yield": 0.10,
        "tech_cycle": 0.09,
    },
    "KOSDAQ": {
        "pe_ratio": 0.16,
        "peg_ratio": 0.16,
        "revenue_growth_yoy": 0.20,
        "asset_turnover": 0.16,
        "debt_ratio": 0.13,
        "dividend_yield": 0.10,
        "tech_cycle": 0.09,
    },
}

for _sector_name, _weights in DEFAULT_WEIGHTS.items():
    if round(sum(_weights.values()), 6) != 1.0:
        LOGGER.warning("DEFAULT_WEIGHTS for %s should sum to 1.0 (actual=%s)", _sector_name, sum(_weights.values()))


@ttl_cache(maxsize=64, ttl=300)
def _finnhub_metrics(symbol: str) -> Dict[str, float]:
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return {}

    try:
        response = requests.get(
            "https://finnhub.io/api/v1/stock/metric",
            params={"symbol": symbol, "metric": "all", "token": api_key},
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("metric", {}) if isinstance(payload, dict) else {}
    except (requests.RequestException, JSONDecodeError, ValueError):
        LOGGER.warning("Failed to fetch Finnhub metrics for %s", symbol)
        return {}


@ttl_cache(maxsize=256, ttl=300)
def _ticker_info(symbol: str) -> Dict[str, object]:
    try:
        info = yf.Ticker(symbol).info or {}
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


@ttl_cache(maxsize=256, ttl=600)
def _ticker_object(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol)


def _resolve_data_symbol(symbol: str, market: str) -> str:
    symbol = symbol.upper()
    if symbol.endswith(".KS") or symbol.endswith(".KQ"):
        return symbol
    if market == "KOSPI":
        return f"{symbol}.KS"
    if market == "KOSDAQ":
        return f"{symbol}.KQ"
    return symbol


def _krx_price_from_fdr(symbol: str) -> float | None:
    try:
        import FinanceDataReader as fdr  # type: ignore
    except Exception:
        LOGGER.info("FinanceDataReader is not installed. Skipping KRX fallback for %s", symbol)
        return None
    try:
        start_date = (pd.Timestamp.today() - pd.Timedelta(days=KRX_PRICE_LOOKBACK_PERIOD_DAYS)).strftime("%Y-%m-%d")
        frame = fdr.DataReader(symbol, start=start_date)
    except Exception:
        LOGGER.warning("FinanceDataReader lookup failed for %s", symbol)
        return None
    if frame is None or frame.empty:
        LOGGER.warning("FinanceDataReader returned empty frame for %s", symbol)
        return None
    close_series = pd.to_numeric(frame.get("Close"), errors="coerce").dropna()
    if close_series.empty:
        return None
    return float(close_series.iloc[-1])


def _safe_number(value, scale: float = 1.0) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(val):
        return None
    return val * scale


def _extract_latest_numeric(frame: pd.DataFrame, candidates: List[str]) -> float | None:
    if frame is None or frame.empty:
        return None
    for row_name in candidates:
        if row_name in frame.index:
            series = pd.to_numeric(frame.loc[row_name], errors="coerce").dropna()
            if not series.empty:
                return _safe_number(series.iloc[0])
    return None


def _fallback_total_revenue(ticker: yf.Ticker) -> float | None:
    try:
        return _extract_latest_numeric(
            ticker.income_stmt,
            ["Total Revenue", "Operating Revenue", "Revenue"],
        )
    except Exception:
        return None


def _fallback_total_assets(ticker: yf.Ticker) -> float | None:
    try:
        return _extract_latest_numeric(
            ticker.balance_sheet,
            ["Total Assets"],
        )
    except Exception:
        return None


def _quarterly_revenue_growth(ticker: yf.Ticker) -> float | None:
    try:
        q_stmt = ticker.quarterly_income_stmt
        if q_stmt is None or q_stmt.empty or "Total Revenue" not in q_stmt.index:
            return None
        revenues = pd.to_numeric(q_stmt.loc["Total Revenue"], errors="coerce").dropna()
        if len(revenues) < 8:
            return None
        recent = float(revenues.iloc[:4].sum())
        previous = float(revenues.iloc[4:8].sum())
        if previous == 0:
            return None
        return (recent / previous) - 1.0
    except (KeyError, IndexError, ValueError, TypeError, AttributeError):
        return None


def get_sector(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol in AI_TICKERS:
        return "AI"
    if symbol in SPACE_TICKERS:
        return "SPACE"
    meta = metadata_for_ticker(symbol, _ticker_info(symbol))
    market = str(meta.get("market", "UNKNOWN")).upper()
    if market in {"KOSPI", "KOSDAQ"}:
        return market
    inferred = meta.get("sector", "MARKET")
    return inferred if inferred in METRIC_SPECS else "MARKET"


def get_sector_info(symbol: str) -> Tuple[str, str]:
    """Get sector/sub-sector for display using metadata first, then yfinance fallback."""
    symbol_upper = symbol.upper()
    if symbol_upper in STOCK_METADATA:
        meta = STOCK_METADATA[symbol_upper]
        return str(meta.get("sector", "Unknown")), str(meta.get("sub_sector", "Unknown"))

    meta = metadata_for_ticker(symbol_upper, _ticker_info(symbol_upper))
    sector = str(meta.get("sector") or "")
    sub_sector = str(meta.get("sub_sector") or "")
    if sector:
        return sector, sub_sector or "Unknown"

    try:
        info = _ticker_info(symbol_upper)
        return str(info.get("sector") or "Unknown"), str(info.get("industry") or "Unknown")
    except Exception:
        return "Unknown", "Unknown"


def get_sector_hierarchy(symbol: str) -> Tuple[str, str, str]:
    sector, sub_sector = get_sector_info(symbol)
    info = _ticker_info(symbol.upper())
    industry = str(info.get("industry") or sub_sector or "Unknown")
    return sector or "Unknown", sub_sector or "Unknown", industry


def normalize_metric(value: float | None, spec: MetricSpec, sector: str | None = None, metric_name: str | None = None) -> float:
    """Normalize a raw metric into [0, 1] using sector metric bounds.

    Missing values return 0.5 (neutral). For metrics where lower values are better,
    the normalized score is inverted.
    """
    if value is None:
        if sector and metric_name:
            value = _sector_metric_average(sector, metric_name)
        if value is None:
            return 0.5
    clipped = min(max(value, spec.min_value), spec.max_value)
    span = spec.max_value - spec.min_value
    if span == 0:
        return 0.5
    ratio = (clipped - spec.min_value) / span
    return ratio if spec.higher_is_better else 1.0 - ratio


def fetch_atr(symbol: str, period: int = 14) -> float:
    ticker = _ticker_object(symbol)
    lookback = max(30, period * 3)
    history = ticker.history(period=f"{lookback}d")
    if history is None or history.empty:
        return 0.0
    high = pd.to_numeric(history.get("High"), errors="coerce")
    low = pd.to_numeric(history.get("Low"), errors="coerce")
    close = pd.to_numeric(history.get("Close"), errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = pd.to_numeric(true_range, errors="coerce").rolling(period).mean().dropna()
    if atr.empty:
        return 0.0
    return float(atr.iloc[-1])


def compute_stop_loss(symbol: str, current_price: float, atr_multiplier: float = 2.0) -> float:
    """ATR 기반 손절가."""
    atr = fetch_atr(symbol, period=14)
    return round(float(current_price) - float(atr_multiplier) * float(atr), 2)


def compute_target_price(current_price: float, score: float) -> float:
    """스코어 45~100을 5~20% 상승으로 매핑."""
    clipped_score = min(100.0, max(45.0, float(score)))
    upside_pct = 0.05 + (clipped_score - 45.0) / 55.0 * 0.15
    return round(float(current_price) * (1 + upside_pct), 2)


def suggest_position_size(
    score: float,
    total_portfolio_value: float,
    win_rate: float = 0.55,
    avg_win_loss_ratio: float = 1.5,
) -> Dict[str, float | str]:
    """Kelly Criterion: f* = (bp - q) / b."""
    portfolio_value = max(0.0, float(total_portfolio_value))
    b = max(0.01, float(avg_win_loss_ratio))
    p = min(1.0, max(0.0, float(win_rate)))
    q = 1 - p
    kelly_fraction = max(0.0, (b * p - q) / b)
    suggested_amount = portfolio_value * kelly_fraction
    max_amount = portfolio_value * 0.20
    return {
        "kelly_fraction": round(kelly_fraction, 4),
        "suggested_amount": round(min(suggested_amount, max_amount), 2),
        "max_amount": round(max_amount, 2),
        "rationale": f"Kelly {kelly_fraction*100:.1f}% 기반 (score={float(score):.1f})",
    }


def stock_metrics(symbol: str, sector: str) -> Dict[str, float | None]:
    symbol = symbol.upper()
    meta = metadata_for_ticker(symbol, _ticker_info(symbol))
    market = str(meta.get("market", "UNKNOWN")).upper()
    data_symbol = _resolve_data_symbol(symbol, market)

    ticker = _ticker_object(data_symbol)
    info = _ticker_info(data_symbol)
    finnhub = _finnhub_metrics(symbol if market in {"NASDAQ", "NYSE", "US"} else data_symbol)

    total_revenue = _safe_number(info.get("totalRevenue")) or _fallback_total_revenue(ticker)
    total_assets = _safe_number(info.get("totalAssets")) or _fallback_total_assets(ticker)
    rd_expense = _safe_number(info.get("researchDevelopment"))
    current_price = _safe_number(info.get("currentPrice")) or _safe_number(info.get("regularMarketPrice"))
    if current_price is None and market in {"KOSPI", "KOSDAQ"}:
        current_price = _krx_price_from_fdr(symbol)

    rd_ratio = None
    if rd_expense is not None and total_revenue is not None and total_revenue != 0:
        rd_ratio = rd_expense / total_revenue

    asset_turnover = None
    if total_revenue is not None and total_assets is not None and total_assets != 0:
        asset_turnover = total_revenue / total_assets

    if sector == "AI":
        metrics = {
            "pe_ratio": _safe_number(info.get("trailingPE")),
            "peg_ratio": _safe_number(info.get("pegRatio")),
            "revenue_growth_yoy": _safe_number(info.get("revenueGrowth")),
            "rd_ratio": rd_ratio,
            "asset_turnover": asset_turnover,
            "tech_cycle": _safe_number(finnhub.get("52WeekPriceReturnDaily"), 0.01),
            "current_price": current_price,
            "daily_change_pct": _safe_number(info.get("regularMarketChangePercent"), 0.01),
        }
    elif sector == "SPACE":
        metrics = {
            "pb_ratio": _safe_number(info.get("priceToBook")),
            "operating_margin": _safe_number(info.get("operatingMargins")),
            "debt_ratio": _safe_number(info.get("debtToEquity"), 0.01),
            "dividend_yield": _safe_number(info.get("dividendYield")),
            "backlog_proxy": _quarterly_revenue_growth(ticker),
            "gov_cycle": _safe_number(finnhub.get("52WeekPriceReturnDaily"), 0.01),
            "current_price": current_price,
            "daily_change_pct": _safe_number(info.get("regularMarketChangePercent"), 0.01),
        }
    else:
        metrics = {
            "pe_ratio": _safe_number(info.get("trailingPE")),
            "peg_ratio": _safe_number(info.get("pegRatio")),
            "revenue_growth_yoy": _safe_number(info.get("revenueGrowth")),
            "asset_turnover": asset_turnover,
            "debt_ratio": _safe_number(info.get("debtToEquity"), 0.01),
            "dividend_yield": _safe_number(info.get("dividendYield")),
            "tech_cycle": _safe_number(finnhub.get("52WeekPriceReturnDaily"), 0.01),
            "current_price": current_price,
            "daily_change_pct": _safe_number(info.get("regularMarketChangePercent"), 0.01),
        }

    if market in {"KOSPI", "KOSDAQ"}:
        fallback = KR_STOCK_METRICS.get(symbol, {})
        if metrics.get("pe_ratio") is None and fallback.get("pe_ratio") is not None:
            metrics["pe_ratio"] = fallback["pe_ratio"]
        if metrics.get("peg_ratio") is None and fallback.get("peg_ratio") is not None:
            metrics["peg_ratio"] = fallback["peg_ratio"]
    return metrics


def score_stock(metrics: Dict[str, float | None], sector: str, weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Compute a 0-100 weighted score and per-metric normalized scores."""
    specs = METRIC_SPECS[sector]
    normalized: Dict[str, float] = {}

    effective_weights = {metric: max(0.0, float(weights.get(metric, 0.0))) for metric in specs}
    weight_total = sum(effective_weights.values())
    if weight_total <= 0:
        LOGGER.warning("All weights are zero/negative for sector=%s. Falling back to equal weights.", sector)
        effective_weights = {metric: 1.0 for metric in specs}
        weight_total = float(len(specs))

    score = 0.0

    for metric_name, spec in specs.items():
        metric_weight = effective_weights[metric_name] / weight_total
        metric_score = normalize_metric(metrics.get(metric_name), spec, sector=sector, metric_name=metric_name)
        normalized[metric_name] = metric_score
        score += metric_score * metric_weight

    return round(score * 100, 2), normalized


def signal_from_score(score: float) -> str:
    if score >= 70:
        return "진입"
    if score >= 45:
        return "보유"
    return "매도"


def build_entry_analysis(symbol: str, metrics: Dict[str, float | None], score: float, news_count: int = 0, sentiment_ratio: float = 0.5) -> Dict[str, object]:
    pe = metrics.get("pe_ratio")
    peg = metrics.get("peg_ratio")
    growth = metrics.get("revenue_growth_yoy")
    eps_growth = metrics.get("eps_growth_yoy")
    rsi = metrics.get("rsi")
    above_20ma = bool(metrics.get("price_above_20ma", False))
    volume_ratio = metrics.get("volume_ratio")
    level_1_pass = (pe is not None and pe <= 35) and (peg is not None and peg <= 2.0) and (rsi is None or 45 <= rsi <= 70)
    level_2_pass = (eps_growth is None or eps_growth >= 0) and (growth is None or growth >= 0) and (peg is None or peg <= 2.0)
    level_3_pass = (rsi is None or 45 <= rsi <= 65) and (above_20ma or metrics.get("ma20") is None) and (volume_ratio is None or volume_ratio >= 0.8)
    level_4_pass = news_count <= 0 or sentiment_ratio >= 0.5
    level_5_pass = score >= 70
    decision = "🟢 강 진입" if level_5_pass and level_1_pass and level_2_pass else ("🟡 중 진입" if score >= 55 else "🟠 관찰")
    return {
        "symbol": symbol,
        "score": round(float(score), 2),
        "decision": decision,
        "levels": [
            {"level": 1, "name": "기본 스크린", "pass": level_1_pass, "detail": f"PEG={peg}, PE={pe}, RSI={rsi}"},
            {"level": 2, "name": "성장성 검증", "pass": level_2_pass, "detail": f"EPS={eps_growth}, Revenue={growth}, PEG={peg}"},
            {"level": 3, "name": "기술적 신호", "pass": level_3_pass, "detail": f"20MA={above_20ma}, 거래량비={volume_ratio}, RSI={rsi}"},
            {"level": 4, "name": "뉴스/감성", "pass": level_4_pass, "detail": f"뉴스={news_count}, 감성비율={sentiment_ratio:.2f}"},
            {"level": 5, "name": "최종 점수/Kelly", "pass": level_5_pass, "detail": f"최종 점수={score:.1f}"},
        ],
    }


def build_exit_analysis(symbol: str, metrics: Dict[str, float | None], profit_rate: float | None) -> Dict[str, object]:
    profit = float(profit_rate) if profit_rate is not None else 0.0
    rsi = metrics.get("rsi")
    above_200ma = bool(metrics.get("price_above_200ma", True))
    volume_ratio = metrics.get("volume_ratio")
    eps_growth = metrics.get("eps_growth_yoy")
    level_1_risk = 1 if profit <= -20 else (0 if profit > -5 else 0.5)
    level_2_risk = 1 if (rsi is not None and rsi < 35) or not above_200ma or (volume_ratio is not None and volume_ratio > 2.0) else 0
    level_3_risk = 1 if (eps_growth is not None and eps_growth < 0) else 0
    risk_score = int(round(level_1_risk + level_2_risk + level_3_risk + (1 if profit <= -10 else 0)))
    risk_score = max(0, min(5, risk_score))
    decision = "🔴 손절" if risk_score >= 4 else ("🟡 손절 검토" if risk_score >= 2 else "🟢 보유")
    return {
        "symbol": symbol,
        "risk_score": risk_score,
        "decision": decision,
        "levels": [
            {"level": 1, "name": "손실률 필터", "pass": profit > -5, "detail": f"손실률={profit:+.1f}%"},
            {"level": 2, "name": "기술적 신호", "pass": level_2_risk == 0, "detail": f"RSI={rsi}, 200MA상단={above_200ma}, 거래량비={volume_ratio}"},
            {"level": 3, "name": "기본가치 악화", "pass": level_3_risk == 0, "detail": f"EPS성장={eps_growth}"},
            {"level": 4, "name": "최종 판정", "pass": risk_score < 2, "detail": f"위험도={risk_score}/5"},
        ],
    }


def build_portfolio_dataframe(symbols: Iterable[str], weight_overrides: Dict[str, Dict[str, float]] | None = None) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    weight_overrides = weight_overrides or {}
    all_metric_keys = sorted({metric for specs in METRIC_SPECS.values() for metric in specs.keys()})
    symbols_to_process = sorted({s.strip().upper() for s in symbols if s and s.strip()})
    prefetched: List[Dict[str, object]] = []

    def _process_symbol(symbol: str) -> Dict[str, object]:
        score_sector = get_sector(symbol)
        display_sector, display_sub_sector, display_industry = get_sector_hierarchy(symbol)
        metrics = stock_metrics(symbol, score_sector)
        meta = metadata_for_ticker(symbol, _ticker_info(symbol))
        return {
            "symbol": symbol,
            "score_sector": score_sector,
            "display_sector": display_sector,
            "display_sub_sector": display_sub_sector,
            "display_industry": display_industry,
            "metrics": metrics,
            "meta": meta,
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process_symbol, symbol): symbol for symbol in symbols_to_process}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                prefetched.append(future.result())
            except Exception as exc:
                LOGGER.warning("⚠️ %s: %s", symbol, exc)

    _set_run_sector_metric_averages(prefetched)

    for item in sorted(prefetched, key=lambda x: str(x.get("symbol", ""))):
        symbol = str(item["symbol"])
        score_sector = str(item["score_sector"])
        display_sector = str(item["display_sector"])
        display_sub_sector = str(item["display_sub_sector"])
        display_industry = str(item["display_industry"])
        metrics = item["metrics"] if isinstance(item.get("metrics"), dict) else {}
        meta = item["meta"] if isinstance(item.get("meta"), dict) else {}

        weights = weight_overrides.get(score_sector, DEFAULT_WEIGHTS.get(score_sector, DEFAULT_WEIGHTS["MARKET"]))
        score, normalized = score_stock(metrics, score_sector, weights)
        row: Dict[str, object] = {
            "ticker": symbol,
            "company_name": meta.get("company_name", symbol),
            "company_name_ko": meta.get("company_name_ko", ""),
            "company_name_en": meta.get("company_name_en", meta.get("company_name", symbol)),
            "ticker_display": symbol,
            "sector": display_sector,
            "sub_sector": display_sub_sector or meta.get("sub_sector", "General"),
            "industry": display_industry or meta.get("industry", display_sub_sector or "General"),
            "market": meta.get("market", "UNKNOWN"),
            "score": score,
            "signal": signal_from_score(score),
        }
        for key in all_metric_keys:
            value = metrics.get(key)
            row[key] = value if value is not None else pd.NA
            row[f"{key}_normalized"] = normalized.get(key, pd.NA)
        row["current_price"] = metrics.get("current_price", pd.NA)
        row["daily_change_pct"] = metrics.get("daily_change_pct", pd.NA)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["sector_rank"] = frame.groupby("sector")["score"].rank(ascending=False, method="min").astype(int)
    return frame.sort_values(["sector", "score"], ascending=[True, False]).reset_index(drop=True)


def build_portfolio_dataframe_parallel(
    symbols: Iterable[str],
    weight_overrides: Dict[str, Dict[str, float]] | None = None,
) -> pd.DataFrame:
    """ThreadPoolExecutor로 병렬 페칭."""
    return build_portfolio_dataframe(symbols, weight_overrides=weight_overrides)


def metric_basis_table(sector: str) -> pd.DataFrame:
    specs = METRIC_SPECS[sector]
    return pd.DataFrame(
        [
            {
                "Metric": spec.label,
                "Formula": spec.formula,
                "Source": spec.source,
                "Normalization Range": f"{spec.min_value} ~ {spec.max_value}",
                "Direction": "Higher is better" if spec.higher_is_better else "Lower is better",
            }
            for spec in specs.values()
        ]
    )
