from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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

AI_TICKERS = ["MSFT", "NVDA", "GOOGL", "TSLA"]
SPACE_TICKERS = ["RTX", "LMT", "NOC", "BA"]
MARKET_TICKERS = load_watchlist_tickers()

SECTOR_TICKERS = {
    "AI": AI_TICKERS,
    "SPACE": SPACE_TICKERS,
    "MARKET": MARKET_TICKERS,
}

SECTOR_HIERARCHY = DEFAULT_SECTOR_HIERARCHY


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
}

for _sector_name, _weights in DEFAULT_WEIGHTS.items():
    if round(sum(_weights.values()), 6) != 1.0:
        LOGGER.warning("DEFAULT_WEIGHTS for %s should sum to 1.0 (actual=%s)", _sector_name, sum(_weights.values()))


@lru_cache(maxsize=64)
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


@lru_cache(maxsize=256)
def _ticker_info(symbol: str) -> Dict[str, object]:
    try:
        info = yf.Ticker(symbol).info or {}
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=256)
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


def normalize_metric(value: float | None, spec: MetricSpec) -> float:
    """Normalize a raw metric into [0, 1] using sector metric bounds.

    Missing values return 0.5 (neutral). For metrics where lower values are better,
    the normalized score is inverted.
    """
    if value is None:
        return 0.5
    clipped = min(max(value, spec.min_value), spec.max_value)
    span = spec.max_value - spec.min_value
    if span == 0:
        return 0.5
    ratio = (clipped - spec.min_value) / span
    return ratio if spec.higher_is_better else 1.0 - ratio


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
        return {
            "pe_ratio": _safe_number(info.get("trailingPE")),
            "peg_ratio": _safe_number(info.get("pegRatio")),
            "revenue_growth_yoy": _safe_number(info.get("revenueGrowth")),
            "rd_ratio": rd_ratio,
            "asset_turnover": asset_turnover,
            "tech_cycle": _safe_number(finnhub.get("52WeekPriceReturnDaily"), 0.01),
            "current_price": current_price,
        }

    if sector == "SPACE":
        return {
            "pb_ratio": _safe_number(info.get("priceToBook")),
            "operating_margin": _safe_number(info.get("operatingMargins")),
            "debt_ratio": _safe_number(info.get("debtToEquity"), 0.01),
            "dividend_yield": _safe_number(info.get("dividendYield")),
            "backlog_proxy": _quarterly_revenue_growth(ticker),
            "gov_cycle": _safe_number(finnhub.get("52WeekPriceReturnDaily"), 0.01),
            "current_price": current_price,
        }

    return {
        "pe_ratio": _safe_number(info.get("trailingPE")),
        "peg_ratio": _safe_number(info.get("pegRatio")),
        "revenue_growth_yoy": _safe_number(info.get("revenueGrowth")),
        "asset_turnover": asset_turnover,
        "debt_ratio": _safe_number(info.get("debtToEquity"), 0.01),
        "dividend_yield": _safe_number(info.get("dividendYield")),
        "tech_cycle": _safe_number(finnhub.get("52WeekPriceReturnDaily"), 0.01),
        "current_price": current_price,
    }


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
        metric_score = normalize_metric(metrics.get(metric_name), spec)
        normalized[metric_name] = metric_score
        score += metric_score * metric_weight

    return round(score * 100, 2), normalized


def signal_from_score(score: float) -> str:
    if score >= 70:
        return "진입"
    if score >= 45:
        return "보유"
    return "매도"


def build_portfolio_dataframe(symbols: Iterable[str], weight_overrides: Dict[str, Dict[str, float]] | None = None) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    weight_overrides = weight_overrides or {}
    all_metric_keys = sorted({metric for specs in METRIC_SPECS.values() for metric in specs.keys()})

    for symbol in sorted({s.strip().upper() for s in symbols if s and s.strip()}):
        score_sector = get_sector(symbol)
        display_sector, display_sub_sector = get_sector_info(symbol)
        metrics = stock_metrics(symbol, score_sector)
        weights = weight_overrides.get(score_sector, DEFAULT_WEIGHTS[score_sector])
        score, normalized = score_stock(metrics, score_sector, weights)
        meta = metadata_for_ticker(symbol, _ticker_info(symbol))

        row: Dict[str, object] = {
            "ticker": symbol,
            "company_name": meta.get("company_name", symbol),
            "company_name_ko": meta.get("company_name_ko", ""),
            "company_name_en": meta.get("company_name_en", meta.get("company_name", symbol)),
            "ticker_display": symbol,
            "sector": display_sector,
            "sub_sector": display_sub_sector or meta.get("sub_sector", "General"),
            "market": meta.get("market", "UNKNOWN"),
            "score": score,
            "signal": signal_from_score(score),
        }
        for key in all_metric_keys:
            value = metrics.get(key)
            row[key] = value if value is not None else pd.NA
            row[f"{key}_normalized"] = normalized.get(key, pd.NA)
        row["current_price"] = metrics.get("current_price", pd.NA)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["sector_rank"] = frame.groupby("sector")["score"].rank(ascending=False, method="min").astype(int)
    return frame.sort_values(["sector", "score"], ascending=[True, False]).reset_index(drop=True)


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
