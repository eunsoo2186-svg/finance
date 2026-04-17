from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from ai_analysis import blended_signal_score
from news_aggregator import NewsAggregator, extract_keywords, recent_news
from portfolio_data import (
    DEFAULT_WEIGHTS,
    METRIC_SPECS,
    build_portfolio_dataframe,
    get_sector,
    get_sector_hierarchy,
    load_all_tickers,
    metric_basis_table,
    stock_metrics,
)
from watchlist_manager import (
    load_favorites,
    load_holdings,
    load_notes,
    load_watchlist,
    load_watchlist_tickers,
    save_favorites,
    save_holdings,
    save_notes,
    save_watchlist,
)

st.set_page_config(page_title="Portfolio Dashboard", page_icon="📊", layout="wide")
st.title("📊 포트폴리오 대시보드")
st.caption("시장 전체 티커 검색 + 보유/미보유 통합 분석 + 재무/뉴스 통합 모니터링")


def _format_number(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _format_currency(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{amount:,.0f}"


def _format_us_price(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _format_kr_price(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"₩{float(value):,.0f}"
    except (TypeError, ValueError):
        return "-"


def _is_kr_exchange(exchange: object) -> bool:
    return str(exchange or "").upper() in {"KOSPI", "KOSDAQ", "KRX"}


def _format_price_by_exchange(value: object, exchange: object) -> str:
    return _format_kr_price(value) if _is_kr_exchange(exchange) else _format_us_price(value)


def _display_name(row: pd.Series) -> str:
    ko = str(row.get("company_name_ko") or "").strip()
    en = str(row.get("company_name_en") or row.get("company_name") or row.get("ticker") or "").strip()
    return ko or en


def _display_name_ko(row: pd.Series) -> str:
    ko = str(row.get("company_name_ko") or "").strip()
    return ko or _display_name(row)


def _row_background_style(row: pd.Series) -> list[str]:
    color = "background-color: #eaf3ff;" if row.get("보유여부") == "✅" else ""
    return [color] * len(row)


def _return_style(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if val > 0:
        return "color: #2e7d32; font-weight: 600;"
    if val < 0:
        return "color: #d32f2f; font-weight: 600;"
    return ""


def _metric_value_style(value: object, metric_name: str) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""

    thresholds = {
        "P/E": (10, 35),
        "PEG": (0.8, 2.0),
        "Revenue Growth(%)": (5, 15),
        "R&D Ratio(%)": (5, 15),
        "Operating Margin(%)": (8, 20),
        "Debt Ratio(%)": (30, 60),
        "Dividend Yield(%)": (1, 4),
    }
    if metric_name not in thresholds:
        return ""

    low, high = thresholds[metric_name]
    if metric_name in {"P/E", "PEG", "Debt Ratio(%)"}:
        if val <= low:
            return "color: #1565c0; font-weight: 600;"
        if val <= high:
            return "color: #f57c00; font-weight: 600;"
        return "color: #c62828; font-weight: 600;"

    if val >= high:
        return "color: #1565c0; font-weight: 600;"
    if val >= low:
        return "color: #f57c00; font-weight: 600;"
    return "color: #c62828; font-weight: 600;"


def _daily_signal(score: float) -> str:
    if score >= 80:
        return "🟢 진입"
    if score >= 60:
        return "🟡 관찰"
    if score >= 40:
        return "🟠 주의"
    return "🔴 회피"


def _validate_ticker_before_adding(ticker: str, ticker_meta: dict[str, dict[str, str]]) -> dict:
    ticker_upper = ticker.strip().upper()
    if not ticker_upper:
        return {"valid": False, "message": "❌ 빈 티커는 추가할 수 없습니다.", "missing_fields": ["ticker"]}

    try:
        base_sector = get_sector(ticker_upper)
        metrics = stock_metrics(ticker_upper, base_sector)
        sector, sub_sector, industry = get_sector_hierarchy(ticker_upper)
        price = metrics.get("current_price")
    except Exception:
        return {"valid": False, "message": f"❌ {ticker_upper}: 존재하지 않는 종목 또는 API 오류", "missing_fields": ["all"]}

    meta = ticker_meta.get(ticker_upper, {})
    required_fields = {
        "currentPrice": price,
        "sector": meta.get("sector") or sector,
        "industry": meta.get("industry") or industry or sub_sector,
    }
    missing_fields = [k for k, v in required_fields.items() if v in (None, "", "Unknown")]
    if missing_fields:
        return {
            "valid": False,
            "message": f"❌ {ticker_upper}: 데이터 불완전 (섹터/가격 정보 없음)",
            "missing_fields": missing_fields,
        }

    return {"valid": True, "data": required_fields, "message": f"✅ {ticker_upper}: 검증 완료", "meta": meta}


def _get_hold_action(profit_rate: float | None, metrics: dict) -> dict[str, str]:
    if profit_rate is None or pd.isna(profit_rate):
        return {"action": "🟢 홀딩", "reason": "보유 정보 부족"}
    pe = metrics.get("P/E")
    peg = pd.to_numeric(metrics.get("PEG"), errors="coerce")
    if profit_rate < -5:
        return {"action": "🔴 손절 검토", "reason": f"손실률 {profit_rate:.1f}% | PE {pe if pd.notna(pe) else '-'} 점검"}
    if profit_rate > 15:
        return {"action": "🟡 부분 매도 검토", "reason": f"수익률 {profit_rate:.1f}% - 일부 수익 실현"}
    if profit_rate < 0 and pd.notna(peg) and peg < 1.0:
        return {"action": "🟢 평균매수 기회", "reason": f"PEG {peg:.2f} 저평가 구간"}
    return {"action": "🟢 홀딩", "reason": f"수익률 {profit_rate:.1f}% - 목표가까지 보유"}


def _get_entry_signal(row: pd.Series) -> dict[str, str]:
    score = float(row.get("스코어") or 0)
    peg = pd.to_numeric(row.get("PEG"), errors="coerce")
    pe = pd.to_numeric(row.get("P/E"), errors="coerce")
    growth = pd.to_numeric(row.get("Revenue Growth(%)"), errors="coerce")
    if pd.notna(peg) and peg < 1.0:
        return {"action": "🟢 강 진입", "reason": f"PEG {peg:.2f} < 1.0"}
    if pd.notna(pe) and pd.notna(growth) and pe < 20 and growth > 12:
        return {"action": "🟡 중 진입", "reason": f"PE {pe:.1f} / 성장률 {growth:.1f}%"}
    if score >= 70:
        return {"action": "🔵 관찰", "reason": f"스코어 {score:.0f}/100"}
    return {"action": "🟠 회피", "reason": "매수 신호 부족"}


def _generate_portfolio_insights(df: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    if df.empty:
        return insights
    held = df[df["보유여부"] == "✅"]
    if not held.empty:
        concentration = held["섹터"].value_counts(normalize=True)
        if not concentration.empty and concentration.iloc[0] > 0.5:
            insights.append(f"⚠️ {concentration.index[0]} 비중 {concentration.iloc[0]*100:.0f}% → 분산 추천")
        losing = held[pd.to_numeric(held["수익률(%)"], errors="coerce") < 0]
        if not losing.empty:
            insights.append(f"⚠️ 손실 종목 {len(losing)}개: {', '.join(losing['TICKER'].head(3).tolist())}")
    sector_returns = (
        df[pd.to_numeric(df["수익률(%)"], errors="coerce").notna()].groupby("섹터")["수익률(%)"].mean().sort_values()
    )
    if len(sector_returns) >= 2:
        insights.append(f"📈 {sector_returns.index[-1]}: {sector_returns.iloc[-1]:+.1f}% | 📉 {sector_returns.index[0]}: {sector_returns.iloc[0]:+.1f}%")
    insights.append("📊 3개월 단위 리밸런싱 점검 권장")
    return insights


def _resolve_history_symbol(ticker: str, exchange: str) -> str:
    if _is_kr_exchange(exchange):
        if ticker.endswith(".KS") or ticker.endswith(".KQ"):
            return ticker
        return f"{ticker}.KS" if str(exchange).upper() == "KOSPI" else f"{ticker}.KQ"
    return ticker


@st.cache_data(show_spinner=False, ttl=3600)
def _portfolio_flow_heatmap_data(pairs: tuple[tuple[str, str], ...], months: int = 12) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}
    for ticker, exchange in pairs:
        try:
            hist = yf.Ticker(_resolve_history_symbol(ticker, exchange)).history(period="2y", interval="1mo")
        except Exception:
            continue
        if hist is None or hist.empty or "Close" not in hist.columns:
            continue
        returns = pd.to_numeric(hist["Close"], errors="coerce").pct_change() * 100
        returns = returns.dropna().tail(months)
        if returns.empty:
            continue
        returns.index = returns.index.strftime("%Y-%m")
        rows[ticker] = returns
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T.sort_index(axis=1)


def _format_news_datetime(value: object) -> str:
    try:
        ts = int(value or 0)
    except (TypeError, ValueError):
        return ""
    if ts <= 0:
        return ""
    return pd.to_datetime(ts, unit="s", utc=True).strftime("%Y-%m-%d")


METRIC_HELP = {
    "P/E Ratio": "주가를 주당순이익으로 나눈 값. 낮을수록 밸류에이션 부담이 낮습니다.",
    "PEG Ratio": "P/E를 이익 성장률로 나눈 값. 1에 가까울수록 성장 대비 가격이 균형적입니다.",
    "YoY Revenue Growth": "전년 대비 매출 성장률입니다.",
    "R&D / Revenue": "매출 대비 연구개발비 비중입니다.",
    "Asset Turnover": "총자산 대비 매출 효율입니다.",
    "Tech Cycle Position": "52주 수익률 기반의 기술 사이클 위치 지표입니다.",
    "P/B Ratio": "주가를 주당순자산으로 나눈 값입니다.",
    "Operating Margin": "매출 대비 영업이익 비율입니다.",
    "Debt Ratio": "자기자본 대비 부채 수준입니다.",
    "Dividend Yield": "주가 대비 배당금 수익률입니다.",
    "Contract Pipeline (Proxy)": "분기 매출 흐름을 기반으로 한 수주 모멘텀 대체 지표입니다.",
    "Gov Spending Cycle (Proxy)": "정책/재정 사이클 민감도를 반영한 대체 지표입니다.",
}
# Cap API calls per refresh while still surfacing a representative notable-news sample.
NEWS_SCAN_LIMIT = 20
# Conservative placeholder for daily drift when no intraday PnL feed is available (~0.2%).
ESTIMATED_DAILY_CHANGE_RATE = 0.002
news_agg = NewsAggregator(os.getenv("FINNHUB_API_KEY", ""))


def _average_return_for_held(frame: pd.DataFrame, holdings_payload: dict) -> str:
    held = frame[frame["ticker"].isin(set(holdings_payload.keys()))].copy()
    if held.empty:
        return "N/A"
    held["purchase_price"] = held["ticker"].apply(lambda t: float(holdings_payload.get(t, {}).get("purchase_price", 0) or 0))
    held["quantity"] = held["ticker"].apply(lambda t: float(holdings_payload.get(t, {}).get("quantity", 0) or 0))
    held["current_value"] = held["current_price"].astype(float) * held["quantity"]
    held["cost"] = held["purchase_price"] * held["quantity"]
    held = held[held["cost"] > 0]
    if held.empty:
        return "N/A"
    ret = ((held["current_value"] - held["cost"]) / held["cost"]) * 100
    return _format_number(ret.mean())

watchlist_defaults = load_watchlist_tickers()
market_rows = load_all_tickers()
ticker_meta_map = {row["ticker"]: row for row in market_rows}
ticker_label_map = {
    row["ticker"]: f"{(row.get('company_name_ko') or row.get('company_name_en') or row.get('company_name') or row['ticker'])} ({row['ticker']}) [{row.get('exchange') or row.get('market', 'UNKNOWN')}]"
    for row in market_rows
}

favorite_tickers = load_favorites()
holdings_data = load_holdings()
notes_data = load_notes()

with st.sidebar:
    st.header("⚙️ 커스터마이징")

    search_text = st.text_input("종목 검색 (Ticker/회사명)", value="")
    normalized_search = search_text.strip().lower()
    filtered_tickers = sorted(
        [
        t for t, label in ticker_label_map.items() if not normalized_search or normalized_search in label.lower() or normalized_search in t.lower()
        ]
    )

    sidebar_options = filtered_tickers if normalized_search else sorted(ticker_label_map.keys())
    default_selected = [t for t in watchlist_defaults if t in sidebar_options][:30]
    selected = st.multiselect(
        "모니터링 종목 (NASDAQ/NYSE/KOSPI/KOSDAQ)",
        options=sidebar_options,
        default=default_selected,
        format_func=lambda t: ticker_label_map.get(t, t),
    )

    favorite_selected = st.multiselect(
        "즐겨찾기 종목",
        options=sorted(ticker_label_map.keys()),
        default=[t for t in favorite_tickers if t in ticker_label_map],
        format_func=lambda t: ticker_label_map.get(t, t),
    )
    if st.button("즐겨찾기 저장"):
        save_favorites(favorite_selected)
        st.success("즐겨찾기를 저장했습니다.")

    custom = st.text_input("직접 추가 종목(쉼표 구분)", value="")
    add_custom_clicked = st.button("종목 추가")

    ai_weights = {}
    with st.expander("AI 가중치 조정", expanded=False):
        for key, default in DEFAULT_WEIGHTS["AI"].items():
            ai_weights[key] = st.slider(f"AI:{key}", 0.0, 1.0, float(default), 0.01)

    space_weights = {}
    with st.expander("우주 가중치 조정", expanded=False):
        for key, default in DEFAULT_WEIGHTS["SPACE"].items():
            space_weights[key] = st.slider(f"SPACE:{key}", 0.0, 1.0, float(default), 0.01)

    market_weights = {}
    with st.expander("시장 공통 가중치 조정", expanded=False):
        for key, default in DEFAULT_WEIGHTS["MARKET"].items():
            market_weights[key] = st.slider(f"MARKET:{key}", 0.0, 1.0, float(default), 0.01)

    score_alert = st.slider("이상 알림 점수 기준", 0, 100, 45)

custom_symbols = [s.strip().upper() for s in custom.split(",") if s.strip()]
symbols = selected + custom_symbols

if add_custom_clicked and custom_symbols:
    existing_watchlist = load_watchlist()
    existing = {str(t).upper() for t in existing_watchlist["ticker"].tolist()}
    additions = []
    validation_errors = []
    for ticker in sorted(set(custom_symbols)):
        if ticker in existing:
            continue
        validation = _validate_ticker_before_adding(ticker, ticker_meta_map)
        if not validation["valid"]:
            validation_errors.append(validation)
            continue
        meta = ticker_meta_map.get(ticker, {})
        sector, sub_sector, industry = get_sector_hierarchy(ticker)
        additions.append(
            {
                "ticker": ticker,
                "company_name_ko": str(meta.get("company_name_ko", "") or ""),
                "company_name_en": str(meta.get("company_name_en", ticker) or ticker),
                "exchange": str(meta.get("exchange", meta.get("market", "UNKNOWN")) or "UNKNOWN"),
                "sector": str(meta.get("sector", sector) or "MARKET"),
                "sub_sector": str(meta.get("sub_sector", sub_sector) or "General"),
                "industry": str(meta.get("industry", industry) or sub_sector or "General"),
            }
        )
    if additions:
        updated_watchlist = pd.concat([existing_watchlist, pd.DataFrame(additions)], ignore_index=True)
        save_watchlist(updated_watchlist)
        st.info(f"직접 추가한 {len(additions)}개 종목을 watchlist에 저장했습니다.")
    for err in validation_errors:
        st.error(err["message"])
        st.warning(f"누락된 정보: {', '.join(err.get('missing_fields', []))}")

watchlist_df = load_watchlist()
st.subheader("📋 관리 중인 종목")
for idx, ticker in enumerate(watchlist_df["ticker"].tolist()):
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write(ticker_label_map.get(ticker, ticker))
    with col2:
        if st.button("🗑️ 제거", key=f"remove_{idx}_{ticker}"):
            updated = watchlist_df[watchlist_df["ticker"] != ticker].copy()
            save_watchlist(updated)
            st.rerun()
    with col3:
        st.write("✅" if ticker in holdings_data else "")

portfolio = build_portfolio_dataframe(symbols, {"AI": ai_weights, "SPACE": space_weights, "MARKET": market_weights})

if portfolio.empty:
    st.warning("표시 가능한 종목이 없습니다. 시장 검색 또는 직접 티커를 입력해 주세요.")
    st.stop()

alert_df = portfolio[portfolio["score"] <= score_alert]
if not alert_df.empty:
    names = alert_df.apply(_display_name, axis=1).tolist()
    st.error(f"⚠️ 이상알림: {', '.join(names)} 점수 {score_alert} 이하")

summary_df = portfolio[portfolio["ticker"].isin(set(holdings_data.keys()))].copy()
if not summary_df.empty:
    summary_df["purchase_price"] = summary_df["ticker"].apply(lambda t: float(holdings_data.get(t, {}).get("purchase_price", 0) or 0))
    summary_df["quantity"] = summary_df["ticker"].apply(lambda t: float(holdings_data.get(t, {}).get("quantity", 0) or 0))
    summary_df["cost"] = summary_df["purchase_price"] * summary_df["quantity"]
    summary_df["current_value"] = pd.to_numeric(summary_df["current_price"], errors="coerce") * summary_df["quantity"]
    summary_df["pnl"] = summary_df["current_value"] - summary_df["cost"]
    total_cost = float(summary_df["cost"].sum())
    total_value = float(summary_df["current_value"].sum())
    total_pnl = float(summary_df["pnl"].sum())
else:
    total_cost = total_value = total_pnl = 0.0
total_return = ((total_pnl / total_cost) * 100) if total_cost > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("총자산", _format_kr_price(total_value))
with col2:
    st.metric("평가손익", _format_kr_price(total_pnl))
with col3:
    st.metric("수익률", f"{total_return:+.1f}%")
with col4:
    estimated_daily_change_pct = ESTIMATED_DAILY_CHANGE_RATE * 100
    st.metric(
        "일일 변화(추정)",
        _format_kr_price(total_value * ESTIMATED_DAILY_CHANGE_RATE),
        delta=f"{estimated_daily_change_pct:+.1f}%",
    )

st.subheader("보유 정보 입력")
held_selection = st.multiselect(
    "보유 중인 종목 선택",
    options=portfolio["ticker"].tolist(),
    default=[t for t in holdings_data.keys() if t in set(portfolio["ticker"])],
    format_func=lambda t: ticker_label_map.get(t, t),
)

edited_holdings = {}
for ticker in held_selection:
    stored = holdings_data.get(ticker, {"purchase_price": 0.0, "quantity": 0.0, "currency": "USD", "target_price": 0.0})
    with st.container(border=True):
        st.markdown(f"**{ticker_label_map.get(ticker, ticker)}**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            purchase_price = st.number_input(f"{ticker} 매입가", min_value=0.0, value=float(stored.get("purchase_price", 0.0)), step=0.01)
        with c2:
            quantity = st.number_input(f"{ticker} 보유수량", min_value=0.0, value=float(stored.get("quantity", 0.0)), step=1.0)
        with c3:
            currency = st.selectbox(f"{ticker} 통화", ["USD", "KRW"], index=0 if stored.get("currency", "USD") == "USD" else 1)
        with c4:
            target_price = st.number_input(f"{ticker} 목표가", min_value=0.0, value=float(stored.get("target_price", 0.0)), step=0.01)
        edited_holdings[ticker] = {
            "purchase_price": purchase_price,
            "quantity": quantity,
            "currency": currency,
            "target_price": target_price,
        }

if st.button("보유정보 저장"):
    save_holdings(edited_holdings)
    holdings_data = edited_holdings
    st.success("보유 정보를 저장했습니다.")

st.subheader("보유/미보유 통합 테이블")
table_df = portfolio.copy()
held_set = set(holdings_data.keys())
table_df["보유여부"] = table_df["ticker"].apply(lambda t: "✅" if t in held_set else "❌")
table_df["종목명(한글)"] = table_df.apply(_display_name_ko, axis=1)
table_df["매입가_raw"] = table_df["ticker"].apply(lambda t: float(holdings_data.get(t, {}).get("purchase_price", 0) or 0) if t in held_set else pd.NA)
table_df["보유수량"] = table_df["ticker"].apply(lambda t: float(holdings_data.get(t, {}).get("quantity", 0) or 0) if t in held_set else pd.NA)
table_df["현재가_raw"] = pd.to_numeric(table_df["current_price"], errors="coerce")
table_df["평가손익_raw"] = (table_df["현재가_raw"] - table_df["매입가_raw"]) * table_df["보유수량"]
table_df["수익률(%)"] = ((table_df["현재가_raw"] - table_df["매입가_raw"]) / table_df["매입가_raw"].replace(0, pd.NA)) * 100
table_df["메모"] = table_df["ticker"].apply(lambda t: str(notes_data.get(t, "")))
table_df["신호"] = table_df["score"].apply(lambda s: _daily_signal(float(s)))
table_df.rename(columns={"ticker": "TICKER", "score": "스코어", "sector": "섹터", "sub_sector": "소섹터", "industry": "산업", "market": "거래소"}, inplace=True)
table_df["섹터 계층"] = table_df["섹터"].fillna("Unknown") + " > " + table_df["소섹터"].fillna("Unknown") + " > " + table_df["산업"].fillna(table_df["소섹터"])
table_df["통화코드"] = table_df.apply(
    lambda row: holdings_data.get(row["TICKER"], {}).get("currency", "KRW" if _is_kr_exchange(row.get("거래소")) else "USD"),
    axis=1,
)
table_df["매입가"] = table_df.apply(lambda row: _format_kr_price(row["매입가_raw"]) if row["통화코드"] == "KRW" else _format_us_price(row["매입가_raw"]), axis=1)
table_df["현재가"] = table_df.apply(lambda row: _format_price_by_exchange(row["현재가_raw"], row["거래소"]), axis=1)
table_df["평가손익"] = table_df.apply(
    lambda row: _format_kr_price(row["평가손익_raw"]) if row["통화코드"] == "KRW" else _format_us_price(row["평가손익_raw"]),
    axis=1,
)

metric_columns = {
    "pe_ratio": "P/E",
    "peg_ratio": "PEG",
    "revenue_growth_yoy": "Revenue Growth(%)",
    "rd_ratio": "R&D Ratio(%)",
    "asset_turnover": "Asset Turnover",
    "tech_cycle": "Tech Cycle",
    "pb_ratio": "P/B",
    "operating_margin": "Operating Margin(%)",
    "debt_ratio": "Debt Ratio(%)",
    "dividend_yield": "Dividend Yield(%)",
    "backlog_proxy": "Backlog Proxy",
    "gov_cycle": "Gov Cycle",
}
table_df.rename(columns=metric_columns, inplace=True)

for percent_col in ["Revenue Growth(%)", "R&D Ratio(%)", "Operating Margin(%)", "Debt Ratio(%)", "Dividend Yield(%)"]:
    if percent_col in table_df.columns:
        table_df[percent_col] = pd.to_numeric(table_df[percent_col], errors="coerce") * 100

sort_col, filter_col = st.columns(2)
with sort_col:
    sort_key = st.selectbox("정렬", ["수익률(%)", "스코어", "신호", "섹터"], index=0)
with filter_col:
    filter_key = st.selectbox("필터", ["전체", "보유만", "미보유만", "진입 신호만"], index=0)

if filter_key == "보유만":
    table_df = table_df[table_df["보유여부"] == "✅"]
elif filter_key == "미보유만":
    table_df = table_df[table_df["보유여부"] == "❌"]
elif filter_key == "진입 신호만":
    table_df = table_df[table_df["신호"].str.contains("진입", na=False)]

if sort_key == "신호":
    signal_rank = {"🟢 진입": 0, "🟡 관찰": 1, "🟠 주의": 2, "🔴 회피": 3}
    table_df["__signal_rank"] = table_df["신호"].map(signal_rank).fillna(99)
    table_df = table_df.sort_values(["__signal_rank", "스코어"], ascending=[True, False]).drop(columns=["__signal_rank"])
elif sort_key in table_df.columns:
    table_df = table_df.sort_values(sort_key, ascending=False, na_position="last")

table_df["액션"] = table_df.apply(
    lambda row: _get_hold_action(row.get("수익률(%)"), row).get("action")
    if row.get("보유여부") == "✅"
    else _get_entry_signal(row).get("action"),
    axis=1,
)
table_df["액션 근거"] = table_df.apply(
    lambda row: _get_hold_action(row.get("수익률(%)"), row).get("reason")
    if row.get("보유여부") == "✅"
    else _get_entry_signal(row).get("reason"),
    axis=1,
)

unified_columns = [
    "보유여부",
    "종목명(한글)",
    "TICKER",
    "거래소",
    "섹터 계층",
    "매입가",
    "보유수량",
    "현재가",
    "평가손익",
    "수익률(%)",
    "메모",
    "스코어",
    "P/E",
    "PEG",
    "Revenue Growth(%)",
    "R&D Ratio(%)",
    "Asset Turnover",
    "Tech Cycle",
    "P/B",
    "Operating Margin(%)",
    "Debt Ratio(%)",
    "Dividend Yield(%)",
    "Backlog Proxy",
    "Gov Cycle",
    "신호",
    "액션",
    "액션 근거",
]
for col in unified_columns:
    if col not in table_df.columns:
        table_df[col] = pd.NA

non_numeric_unified_columns = {"보유여부", "종목명(한글)", "TICKER", "거래소", "섹터 계층", "매입가", "현재가", "평가손익", "메모", "신호", "액션", "액션 근거"}
column_config = {
    "P/E": st.column_config.NumberColumn(help="주가수익비율"),
    "PEG": st.column_config.NumberColumn(help="P/E 대비 성장률 보정 지표"),
    "Revenue Growth(%)": st.column_config.NumberColumn(help="전년 동기 대비 매출 성장률"),
    "R&D Ratio(%)": st.column_config.NumberColumn(help="매출 대비 연구개발비 비율"),
    "Asset Turnover": st.column_config.NumberColumn(help="총자산 대비 매출 효율"),
    "Tech Cycle": st.column_config.NumberColumn(help="52주 수익률 기반 기술 사이클"),
    "P/B": st.column_config.NumberColumn(help="주가순자산비율"),
    "Operating Margin(%)": st.column_config.NumberColumn(help="영업이익률"),
    "Debt Ratio(%)": st.column_config.NumberColumn(help="부채비율"),
    "Dividend Yield(%)": st.column_config.NumberColumn(help="배당수익률"),
    "Backlog Proxy": st.column_config.NumberColumn(help="수주/파이프라인 대체지표"),
    "Gov Cycle": st.column_config.NumberColumn(help="정부 지출 사이클 민감도"),
}
try:
    styled_unified = (
        table_df[unified_columns]
        .style.apply(_row_background_style, axis=1)
        .applymap(_return_style, subset=["수익률(%)"])
        .applymap(lambda v: _metric_value_style(v, "P/E"), subset=["P/E"])
        .applymap(lambda v: _metric_value_style(v, "PEG"), subset=["PEG"])
        .applymap(lambda v: _metric_value_style(v, "Revenue Growth(%)"), subset=["Revenue Growth(%)"])
        .applymap(lambda v: _metric_value_style(v, "R&D Ratio(%)"), subset=["R&D Ratio(%)"])
        .applymap(lambda v: _metric_value_style(v, "Operating Margin(%)"), subset=["Operating Margin(%)"])
        .applymap(lambda v: _metric_value_style(v, "Debt Ratio(%)"), subset=["Debt Ratio(%)"])
        .applymap(lambda v: _metric_value_style(v, "Dividend Yield(%)"), subset=["Dividend Yield(%)"])
        .format(
            {
                "보유수량": "{:.2f}",
                "수익률(%)": lambda v: "-" if pd.isna(v) else f"{float(v):+.1f}%",
                "스코어": "{:.2f}",
                "P/E": "{:.2f}",
                "PEG": "{:.2f}",
                "Revenue Growth(%)": lambda v: "-" if pd.isna(v) else f"{float(v):.1f}%",
                "R&D Ratio(%)": lambda v: "-" if pd.isna(v) else f"{float(v):.1f}%",
                "Asset Turnover": "{:.2f}",
                "Tech Cycle": "{:.2f}",
                "P/B": "{:.2f}",
                "Operating Margin(%)": lambda v: "-" if pd.isna(v) else f"{float(v):.1f}%",
                "Debt Ratio(%)": lambda v: "-" if pd.isna(v) else f"{float(v):.1f}%",
                "Dividend Yield(%)": lambda v: "-" if pd.isna(v) else f"{float(v):.1f}%",
                "Backlog Proxy": "{:.2f}",
                "Gov Cycle": "{:.2f}",
            },
            na_rep="-",
        )
    )
    st.dataframe(styled_unified, use_container_width=True, hide_index=True, column_config=column_config)
except AttributeError:
    fallback_df = table_df[unified_columns].copy()
    for c in fallback_df.columns:
        if c not in non_numeric_unified_columns:
            if c in {"수익률(%)"}:
                fallback_df[c] = fallback_df[c].apply(lambda v: "-" if pd.isna(v) else f"{float(v):+.1f}%")
            elif c in {"Revenue Growth(%)", "R&D Ratio(%)", "Operating Margin(%)", "Debt Ratio(%)", "Dividend Yield(%)"}:
                fallback_df[c] = fallback_df[c].apply(lambda v: "-" if pd.isna(v) else f"{float(v):.1f}%")
            else:
                fallback_df[c] = fallback_df[c].apply(_format_number)
    st.dataframe(fallback_df.fillna("-"), use_container_width=True, hide_index=True)

selected_ticker_for_note = st.selectbox("메모 수정 종목", portfolio["ticker"].tolist(), format_func=lambda t: ticker_label_map.get(t, t))
note_value = st.text_input("메모", value=str(notes_data.get(selected_ticker_for_note, "")))
if st.button("메모 저장"):
    notes_data[selected_ticker_for_note] = note_value
    save_notes(notes_data)
    st.success("메모를 저장했습니다.")

st.subheader("📊 포트폴리오 일일 체크")
daily_checks = [
    ("신규 매수 기회", len(table_df[table_df["신호"].str.contains("진입", na=False)]) > 0),
    ("손절 필요 (손실률 -5% 이하)", len(table_df[pd.to_numeric(table_df["수익률(%)"], errors="coerce") <= -5]) > 0),
    ("수익실현 후보 (수익률 +10% 이상)", len(table_df[pd.to_numeric(table_df["수익률(%)"], errors="coerce") >= 10]) > 0),
    ("섹터별 리밸런싱 점검", True),
    ("주요 뉴스 확인", True),
]
for label, checked in daily_checks:
    st.markdown(f"{'✅' if checked else '☑️'} {label}")

kr_missing = table_df[
    table_df["거래소"].apply(_is_kr_exchange)
    & (table_df["P/E"].isna() | table_df["PEG"].isna())
]["TICKER"].tolist()
if kr_missing:
    st.info(f"⚠️ {', '.join(kr_missing[:5])}: 일부 국내 종목 재무지표가 부족해 '-'로 표시됩니다.")

st.subheader("💡 INSIGHTS")
for idx, insight in enumerate(_generate_portfolio_insights(table_df), start=1):
    st.markdown(f"{idx}️⃣ {insight}")

with st.expander("Metric Definitions", expanded=False):
    for metric_name, metric_desc in METRIC_HELP.items():
        st.markdown(f"- **{metric_name}**: {metric_desc}")
    for sector in ["AI", "SPACE", "MARKET"]:
        st.markdown(f"#### {sector}")
        st.dataframe(metric_basis_table(sector), use_container_width=True, hide_index=True)

heatmap_cols = [c for c in portfolio.columns if c.endswith("_normalized")]
st.subheader("📈 포트폴리오 전체 흐름 히트맵")
flow_pairs = tuple((str(row["ticker"]), str(row.get("market", "UNKNOWN"))) for _, row in portfolio[["ticker", "market"]].iterrows())
flow_heatmap = _portfolio_flow_heatmap_data(flow_pairs, months=12)
if flow_heatmap.empty:
    st.info("포트폴리오 흐름 히트맵 데이터가 부족합니다.")
else:
    fig_flow = px.imshow(
        flow_heatmap,
        labels=dict(x="월", y="종목", color="수익률(%)"),
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Portfolio Monthly Return Heatmap",
    )
    st.plotly_chart(fig_flow, use_container_width=True)

if heatmap_cols:
    heatmap_input = portfolio.set_index("ticker")[heatmap_cols].apply(pd.to_numeric, errors="coerce")
    heatmap_input = heatmap_input.dropna(axis=1, how="all").dropna(axis=0, how="all")
    metric_name_map = {}
    for col in heatmap_input.columns:
        metric = col.replace("_normalized", "")
        metric_name_map[col] = next((spec.label for specs in METRIC_SPECS.values() for k, spec in specs.items() if k == metric), metric)
    if heatmap_input.empty:
        st.info("히트맵 데이터가 부족합니다.")
    else:
        fig_heat = px.imshow(
            heatmap_input.rename(columns=metric_name_map),
            color_continuous_scale="RdYlGn",
            title="Stock Metric Normalization Heatmap (0~1)",
            aspect="auto",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("히트맵 데이터가 부족합니다.")

fig_score = px.bar(
    portfolio,
    x="ticker",
    y="score",
    color="sector",
    text="signal",
    title="Score and Signal by Sector",
)
fig_score.update_traces(textposition="outside")
st.plotly_chart(fig_score, use_container_width=True)

st.subheader("Notable News")
news_ticker = st.selectbox("뉴스 조회 종목", portfolio["ticker"].tolist(), format_func=lambda t: ticker_label_map.get(t, t))
keyword_filter = st.text_input("뉴스 키워드 필터")

notable_rows = []
news_candidates = portfolio.sort_values(["score", "ticker"], ascending=[False, True])["ticker"].tolist()[:NEWS_SCAN_LIMIT]
for ticker in news_candidates:
    rows = news_agg.get_stock_news(ticker, days=60)
    for item in rows[:2]:
        notable_rows.append(
            {
                "ticker": ticker,
                "company": _display_name(portfolio[portfolio["ticker"] == ticker].iloc[0]),
                "datetime": _format_news_datetime(item.get("datetime")),
                "source": item.get("source", ""),
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
            }
        )

if notable_rows:
    notable_df = pd.DataFrame(notable_rows).sort_values("datetime", ascending=False).head(5)
    st.dataframe(
        notable_df[["datetime", "company", "source", "headline", "url"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Notable News를 불러오지 못했습니다. API 키를 확인해 주세요.")

news_rows = recent_news(news_ticker, keyword=keyword_filter, months=3)
if news_rows:
    news_df = pd.DataFrame(news_rows).head(5)
    st.dataframe(news_df[["datetime", "source", "headline", "summary", "url"]], use_container_width=True, hide_index=True)

    keywords = extract_keywords(news_rows)
    if keywords:
        keyword_df = pd.DataFrame(keywords)
        fig_kw = px.bar(keyword_df, x="keyword", y="count", title="뉴스 키워드 빈도")
        st.plotly_chart(fig_kw, use_container_width=True)

    score_series = portfolio.loc[portfolio["ticker"] == news_ticker, "score"]
    if not score_series.empty:
        base_score = float(score_series.iloc[0])
        blended = blended_signal_score(base_score, len(news_rows))
        signal_df = pd.DataFrame(
            [
                {"Metric": "Financial Score", "Score": round(base_score, 2)},
                {"Metric": "Blended AI Score", "Score": round(blended, 2)},
            ]
        )
        fig_signal = px.bar(signal_df, x="Metric", y="Score", range_y=[0, 100], title="AI 판단 점수")
        st.plotly_chart(fig_signal, use_container_width=True)
else:
    st.info("최근 3개월 뉴스가 없거나 API 키가 설정되지 않았습니다.")

st.subheader("섹터별 분석")
sector_overview = {
    "Technology": "AI, 클라우드, 반도체 사이클과 소프트웨어 생산성 개선이 핵심 트렌드입니다.",
    "Healthcare": "신약 파이프라인, 규제 변화, 보험수가 환경이 핵심 변수입니다.",
    "Financial": "금리, 대손비용, 자본건전성 변화가 실적에 직접 반영됩니다.",
    "Industrials": "수주·설비투자·방산/항공 수요 흐름이 핵심입니다.",
    "Communication Services": "플랫폼 트래픽, 광고 경기, 구독 성장률이 중요합니다.",
    "Materials": "원자재 가격과 공급 사이클이 수익성에 직접 영향을 줍니다.",
    "Energy": "유가·가스 가격과 정책 수혜(원전/재생)가 성과를 좌우합니다.",
}
sector_tabs = st.tabs(sorted(portfolio["sector"].dropna().unique().tolist()))
for idx, sector_name in enumerate(sorted(portfolio["sector"].dropna().unique().tolist())):
    with sector_tabs[idx]:
        sec_df = portfolio[portfolio["sector"] == sector_name].copy()
        st.markdown(f"### {sector_name}")
        st.write(sector_overview.get(sector_name, "선택 종목 기반 섹터로, 실적·밸류에이션·뉴스 흐름을 함께 모니터링하세요."))

        st.markdown("#### Sector News")
        sector_news = news_agg.get_sector_news(sector_name, days=7)
        if sector_news:
            for article in sector_news[:5]:
                dt = _format_news_datetime(article.get("datetime"))
                st.markdown(f"- **{article.get('headline', '')}** ({article.get('source', '')}, {dt})  \n  [Read more]({article.get('url', '')})")
        else:
            st.info("섹터 뉴스를 불러오지 못했습니다.")

        st.markdown("#### Sector Financial Snapshot")
        numeric_cols = [c for c in ["score", "pe_ratio", "peg_ratio", "asset_turnover", "operating_margin", "debt_ratio"] if c in sec_df.columns]
        if numeric_cols:
            snapshot = pd.DataFrame(
                [{"Metric": c, "Average": _format_number(pd.to_numeric(sec_df[c], errors="coerce").mean())} for c in numeric_cols]
            )
            st.dataframe(snapshot, use_container_width=True, hide_index=True)

        st.markdown("#### Sector Ranking")
        ranking = sec_df.copy()
        ranking["종목명"] = ranking.apply(_display_name, axis=1)
        ranking["score"] = ranking["score"].apply(_format_number)
        st.dataframe(ranking[["종목명", "ticker", "score", "signal"]], use_container_width=True, hide_index=True)

        st.markdown("#### 산업 동향 차트")
        trend_metrics = [c for c in ["pe_ratio", "peg_ratio", "score"] if c in sec_df.columns]
        if trend_metrics:
            trend_data = pd.DataFrame(
                [{"Metric": m, "Average": pd.to_numeric(sec_df[m], errors="coerce").mean()} for m in trend_metrics]
            )
            st.plotly_chart(px.bar(trend_data, x="Metric", y="Average", title=f"{sector_name} 평균 지표"), use_container_width=True)

        if "score" in sec_df.columns:
            st.plotly_chart(
                px.bar(sec_df, x="ticker", y="score", color="ticker", title=f"{sector_name} 종목별 Score 비교"),
                use_container_width=True,
            )
