from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().parents[1] / "config" / ".env"
load_dotenv(ENV_PATH)

AI_TICKERS = ["MSFT", "NVDA", "GOOGL", "TSLA"]
SPACE_TICKERS = ["RTX", "LMT", "NOC", "BA"]

SECTOR_TICKERS = {
    "AI": AI_TICKERS,
    "SPACE": SPACE_TICKERS,
}


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
            "R&D / Revenue", 0.02, 0.35, True, "yfinance.financials + info.totalRevenue", "researchDevelopment / totalRevenue"
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
            "Contract Pipeline (Proxy)", -0.2, 0.5, True, "yfinance.quarterly_income_stmt", "(최근4개 분기 매출합 / 이전4개 분기 매출합) - 1"
        ),
        "gov_cycle": MetricSpec(
            "Gov Spending Cycle (Proxy)", -0.3, 0.4, True, "finnhub.stock.metric.52WeekPriceReturnDaily", "52WeekPriceReturnDaily"
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
}


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
    except Exception:
        return {}


def _safe_number(value, scale: float = 1.0) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if pd.isna(val):
        return None
    return val * scale


def _quarterly_revenue_growth(ticker: yf.Ticker) -> float | None:
    try:
        q_stmt = ticker.quarterly_income_stmt
        if q_stmt is None or q_stmt.empty or "Total Revenue" not in q_stmt.index:
            return None
        revenues = q_stmt.loc["Total Revenue"].dropna()
        if len(revenues) < 8:
            return None
        recent = float(revenues.iloc[:4].sum())
        previous = float(revenues.iloc[4:8].sum())
        if previous == 0:
            return None
        return (recent / previous) - 1.0
    except Exception:
        return None


def get_sector(symbol: str) -> str:
    symbol = symbol.upper()
    for sector, members in SECTOR_TICKERS.items():
        if symbol in members:
            return sector
    return "CUSTOM"


def normalize_metric(value: float | None, spec: MetricSpec) -> float:
    if value is None:
        return 0.5
    clipped = min(max(value, spec.min_value), spec.max_value)
    ratio = (clipped - spec.min_value) / (spec.max_value - spec.min_value)
    return ratio if spec.higher_is_better else 1.0 - ratio


def stock_metrics(symbol: str, sector: str) -> Dict[str, float | None]:
    symbol = symbol.upper()
    ticker = yf.Ticker(symbol)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    finnhub = _finnhub_metrics(symbol)

    total_revenue = _safe_number(info.get("totalRevenue"))
    total_assets = _safe_number(info.get("totalAssets"))
    rd_expense = _safe_number(info.get("researchDevelopment"))

    rd_ratio = None
    if rd_expense is not None and total_revenue and total_revenue != 0:
        rd_ratio = rd_expense / total_revenue

    asset_turnover = None
    if total_revenue is not None and total_assets and total_assets != 0:
        asset_turnover = total_revenue / total_assets

    if sector == "AI":
        return {
            "pe_ratio": _safe_number(info.get("trailingPE")),
            "peg_ratio": _safe_number(info.get("pegRatio")),
            "revenue_growth_yoy": _safe_number(info.get("revenueGrowth")),
            "rd_ratio": rd_ratio,
            "asset_turnover": asset_turnover,
            "tech_cycle": _safe_number(finnhub.get("52WeekPriceReturnDaily"), 0.01),
        }

    return {
        "pb_ratio": _safe_number(info.get("priceToBook")),
        "operating_margin": _safe_number(info.get("operatingMargins")),
        "debt_ratio": _safe_number(info.get("debtToEquity"), 0.01),
        "dividend_yield": _safe_number(info.get("dividendYield")),
        "backlog_proxy": _quarterly_revenue_growth(ticker),
        "gov_cycle": _safe_number(finnhub.get("52WeekPriceReturnDaily"), 0.01),
    }


def score_stock(metrics: Dict[str, float | None], sector: str, weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    specs = METRIC_SPECS[sector]
    normalized: Dict[str, float] = {}

    weight_total = sum(max(0.0, float(w)) for w in weights.values()) or 1.0
    score = 0.0

    for metric_name, spec in specs.items():
        metric_weight = max(0.0, float(weights.get(metric_name, 0.0))) / weight_total
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

    for symbol in sorted({s.strip().upper() for s in symbols if s and s.strip()}):
        sector = get_sector(symbol)
        if sector not in {"AI", "SPACE"}:
            continue
        metrics = stock_metrics(symbol, sector)
        weights = weight_overrides.get(sector, DEFAULT_WEIGHTS[sector])
        score, normalized = score_stock(metrics, sector, weights)

        row: Dict[str, object] = {
            "ticker": symbol,
            "sector": sector,
            "score": score,
            "signal": signal_from_score(score),
        }
        for key, value in metrics.items():
            row[key] = value
            row[f"{key}_normalized"] = normalized.get(key)
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
                "지표": spec.label,
                "계산 근거": spec.formula,
                "데이터 출처": spec.source,
                "정규화 범위": f"{spec.min_value} ~ {spec.max_value}",
                "평가 방향": "높을수록 좋음" if spec.higher_is_better else "낮을수록 좋음",
            }
            for spec in specs.values()
        ]
    )
