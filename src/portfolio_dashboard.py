from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ai_analysis import blended_signal_score, recommendation_from_score
from market_tickers import load_market_tickers
from news_aggregator import extract_keywords, recent_news
from portfolio_data import DEFAULT_WEIGHTS, METRIC_SPECS, build_portfolio_dataframe, metric_basis_table
from watchlist_manager import load_favorites, load_holdings, load_notes, load_watchlist_tickers, save_favorites, save_holdings, save_notes

st.set_page_config(page_title="Portfolio Dashboard", page_icon="📊", layout="wide")
st.title("📊 포트폴리오 대시보드")
st.caption("시장 전체 티커 검색 + 보유/미보유 통합 분석 + 재무/뉴스 통합 모니터링")


def _format_number(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _display_name(row: pd.Series) -> str:
    ko = str(row.get("company_name_ko") or "").strip()
    en = str(row.get("company_name_en") or row.get("company_name") or row.get("ticker") or "").strip()
    return ko or en


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

watchlist_defaults = load_watchlist_tickers()
market_rows = load_market_tickers()
ticker_meta_map = {row["ticker"]: row for row in market_rows}
ticker_label_map = {
    row["ticker"]: f"{(row.get('company_name_ko') or row.get('company_name_en') or row.get('company_name') or row['ticker'])} ({row['ticker']}) [{row.get('market', 'UNKNOWN')}]"
    for row in market_rows
}

favorite_tickers = load_favorites()
holdings_data = load_holdings()
notes_data = load_notes()

with st.sidebar:
    st.header("⚙️ 커스터마이징")

    search_text = st.text_input("종목 검색 (Ticker/회사명)", value="")
    normalized_search = search_text.strip().lower()
    filtered_tickers = [
        t for t, label in ticker_label_map.items() if not normalized_search or normalized_search in label.lower() or normalized_search in t.lower()
    ]

    selected = st.multiselect(
        "모니터링 종목 (NASDAQ/NYSE/KOSPI/KOSDAQ)",
        options=filtered_tickers if normalized_search else sorted(ticker_label_map.keys()),
        default=[t for t in watchlist_defaults if t in ticker_label_map][:30],
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

symbols = selected + [s.strip().upper() for s in custom.split(",") if s.strip()]
portfolio = build_portfolio_dataframe(symbols, {"AI": ai_weights, "SPACE": space_weights, "MARKET": market_weights})

if portfolio.empty:
    st.warning("표시 가능한 종목이 없습니다. 시장 검색 또는 직접 티커를 입력해 주세요.")
    st.stop()

alert_df = portfolio[portfolio["score"] <= score_alert]
if not alert_df.empty:
    names = alert_df.apply(_display_name, axis=1).tolist()
    st.error(f"⚠️ 이상알림: {', '.join(names)} 점수 {score_alert} 이하")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Symbols", len(portfolio))
with col2:
    st.metric("Average Score", _format_number(portfolio["score"].mean()))
with col3:
    held_set = set(holdings_data.keys())
    held_df = portfolio[portfolio["ticker"].isin(held_set)].copy()
    if not held_df.empty:
        held_df["purchase_price"] = held_df["ticker"].apply(lambda t: float(holdings_data.get(t, {}).get("purchase_price", 0) or 0))
        held_df["quantity"] = held_df["ticker"].apply(lambda t: float(holdings_data.get(t, {}).get("quantity", 0) or 0))
        held_df["current_value"] = held_df["current_price"].astype(float) * held_df["quantity"]
        held_df["cost"] = held_df["purchase_price"] * held_df["quantity"]
        ret = ((held_df["current_value"] - held_df["cost"]) / held_df["cost"].replace(0, pd.NA)) * 100
        st.metric("Average Return (%)", _format_number(ret.mean()))
    else:
        st.metric("Average Return (%)", "N/A")

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
table_df["보유여부"] = table_df["ticker"].apply(lambda t: "보유" if t in held_set else "미보유")
table_df["종목명"] = table_df.apply(_display_name, axis=1)
table_df["매입가"] = table_df["ticker"].apply(lambda t: float(holdings_data.get(t, {}).get("purchase_price", 0) or 0) if t in held_set else pd.NA)
table_df["보유수량"] = table_df["ticker"].apply(lambda t: float(holdings_data.get(t, {}).get("quantity", 0) or 0) if t in held_set else pd.NA)
table_df["통화"] = table_df["ticker"].apply(lambda t: holdings_data.get(t, {}).get("currency", "") if t in held_set else "")
table_df["목표가"] = table_df["ticker"].apply(lambda t: float(holdings_data.get(t, {}).get("target_price", 0) or 0) if t in held_set else pd.NA)
table_df["현재가"] = pd.to_numeric(table_df["current_price"], errors="coerce")
table_df["수익률(%)"] = ((table_df["현재가"] - table_df["매입가"]) / table_df["매입가"].replace(0, pd.NA)) * 100
table_df["메모"] = table_df["ticker"].apply(lambda t: str(notes_data.get(t, "")))
table_df["신호"] = table_df.apply(lambda r: recommendation_from_score(float(r["score"]), held=r["ticker"] in held_set), axis=1)

editor_columns = [
    "보유여부",
    "종목명",
    "ticker",
    "매입가",
    "보유수량",
    "현재가",
    "수익률(%)",
    "통화",
    "목표가",
    "메모",
    "score",
    "신호",
]
editor_df = table_df[editor_columns].rename(columns={"ticker": "TICKER", "score": "스코어"}).copy()
for col in ["매입가", "보유수량", "현재가", "수익률(%)", "목표가", "스코어"]:
    editor_df[col] = editor_df[col].apply(_format_number)

edited_table = st.data_editor(
    editor_df,
    width="stretch",
    hide_index=True,
    disabled=["보유여부", "종목명", "TICKER", "매입가", "보유수량", "현재가", "수익률(%)", "통화", "목표가", "스코어", "신호"],
)
if st.button("메모 저장"):
    updated_notes = {str(row["TICKER"]).upper(): str(row["메모"]) for _, row in edited_table.iterrows()}
    save_notes(updated_notes)
    st.success("메모를 저장했습니다.")

st.subheader("재무 지표 테이블")
display_columns = [
    "company_name_en",
    "ticker",
    "market",
    "sector",
    "sub_sector",
    "score",
    "signal",
    "sector_rank",
    "pe_ratio",
    "peg_ratio",
    "revenue_growth_yoy",
    "rd_ratio",
    "asset_turnover",
    "tech_cycle",
    "pb_ratio",
    "operating_margin",
    "debt_ratio",
    "dividend_yield",
    "backlog_proxy",
    "gov_cycle",
]
column_labels = {
    "company_name_en": "Company Name",
    "ticker": "TICKER",
    "market": "Exchange",
    "sector": "Sector",
    "sub_sector": "Sub Sector",
    "score": "Score",
    "signal": "Signal",
    "sector_rank": "Sector Rank",
    "pe_ratio": "P/E Ratio",
    "peg_ratio": "PEG Ratio",
    "revenue_growth_yoy": "YoY Revenue Growth",
    "rd_ratio": "R&D / Revenue",
    "asset_turnover": "Asset Turnover",
    "tech_cycle": "Tech Cycle Position",
    "pb_ratio": "P/B Ratio",
    "operating_margin": "Operating Margin",
    "debt_ratio": "Debt Ratio",
    "dividend_yield": "Dividend Yield",
    "backlog_proxy": "Contract Pipeline (Proxy)",
    "gov_cycle": "Gov Spending Cycle (Proxy)",
}
display_df = portfolio[[c for c in display_columns if c in portfolio.columns]].copy()
non_numeric_columns = {"company_name_en", "ticker", "market", "sector", "sub_sector", "signal"}
for c in display_df.columns:
    if c not in non_numeric_columns:
        display_df[c] = display_df[c].apply(_format_number)
st.dataframe(display_df.rename(columns=column_labels), width="stretch", hide_index=True)

with st.expander("Metric Definitions", expanded=False):
    for metric_name, metric_desc in METRIC_HELP.items():
        st.markdown(f"- **{metric_name}**: {metric_desc}")
    for sector in ["AI", "SPACE", "MARKET"]:
        st.markdown(f"#### {sector}")
        st.dataframe(metric_basis_table(sector), width="stretch", hide_index=True)

heatmap_cols = [c for c in portfolio.columns if c.endswith("_normalized")]
if heatmap_cols:
    heatmap_input = portfolio.set_index("ticker")[heatmap_cols]
    metric_name_map = {}
    for col in heatmap_input.columns:
        metric = col.replace("_normalized", "")
        metric_name_map[col] = next((spec.label for specs in METRIC_SPECS.values() for k, spec in specs.items() if k == metric), metric)
    fig_heat = px.imshow(
        heatmap_input.rename(columns=metric_name_map),
        color_continuous_scale="RdYlGn",
        title="Stock Metric Normalization Heatmap (0~1)",
        aspect="auto",
    )
    st.plotly_chart(fig_heat, width="stretch")

fig_score = px.bar(
    portfolio,
    x="ticker",
    y="score",
    color="sector",
    text="signal",
    title="Score and Signal by Sector",
)
fig_score.update_traces(textposition="outside")
st.plotly_chart(fig_score, width="stretch")

st.subheader("Notable News")
news_ticker = st.selectbox("뉴스 조회 종목", portfolio["ticker"].tolist(), format_func=lambda t: ticker_label_map.get(t, t))
keyword_filter = st.text_input("뉴스 키워드 필터")

notable_rows = []
for ticker in portfolio["ticker"].tolist()[:20]:
    rows = recent_news(ticker, months=2)
    for item in rows[:2]:
        notable_rows.append(
            {
                "ticker": ticker,
                "company": _display_name(portfolio[portfolio["ticker"] == ticker].iloc[0]),
                "datetime": item.get("datetime", ""),
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
        width="stretch",
        hide_index=True,
    )
else:
    st.info("Notable News를 불러오지 못했습니다. API 키를 확인해 주세요.")

news_rows = recent_news(news_ticker, keyword=keyword_filter, months=3)
if news_rows:
    news_df = pd.DataFrame(news_rows).head(5)
    st.dataframe(news_df[["datetime", "source", "headline", "summary", "url"]], width="stretch", hide_index=True)

    keywords = extract_keywords(news_rows)
    if keywords:
        keyword_df = pd.DataFrame(keywords)
        fig_kw = px.bar(keyword_df, x="keyword", y="count", title="뉴스 키워드 빈도")
        st.plotly_chart(fig_kw, width="stretch")

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
        st.plotly_chart(fig_signal, width="stretch")
else:
    st.info("최근 3개월 뉴스가 없거나 API 키가 설정되지 않았습니다.")

st.subheader("섹터별 분석")
sector_overview = {
    "Technology": "AI, 클라우드, 반도체 사이클과 소프트웨어 생산성 개선이 핵심 트렌드입니다.",
    "Healthcare": "신약 파이프라인, 규제 변화, 보험수가 환경이 핵심 변수입니다.",
    "Financial": "금리, 대손비용, 자본건전성 변화가 실적에 직접 반영됩니다.",
    "Manufacturing": "원자재 가격, 공급망 안정성, 설비투자 회복이 중요합니다.",
}
sector_tabs = st.tabs(sorted(portfolio["sector"].dropna().unique().tolist()))
for idx, sector_name in enumerate(sorted(portfolio["sector"].dropna().unique().tolist())):
    with sector_tabs[idx]:
        sec_df = portfolio[portfolio["sector"] == sector_name].copy()
        st.markdown(f"### {sector_name}")
        st.write(sector_overview.get(sector_name, "선택 종목 기반 섹터로, 실적·밸류에이션·뉴스 흐름을 함께 모니터링하세요."))

        st.markdown("#### Sector Financial Snapshot")
        numeric_cols = [c for c in ["score", "pe_ratio", "asset_turnover", "operating_margin", "debt_ratio"] if c in sec_df.columns]
        if numeric_cols:
            snapshot = pd.DataFrame(
                [{"Metric": c, "Average": _format_number(pd.to_numeric(sec_df[c], errors="coerce").mean())} for c in numeric_cols]
            )
            st.dataframe(snapshot, width="stretch", hide_index=True)

        st.markdown("#### Sector Ranking")
        ranking = sec_df.copy()
        ranking["종목명"] = ranking.apply(_display_name, axis=1)
        ranking["score"] = ranking["score"].apply(_format_number)
        st.dataframe(ranking[["종목명", "ticker", "score", "signal"]], width="stretch", hide_index=True)
