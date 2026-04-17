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
st.caption("시장 전체 티커 검색 + 보유/미보유 분리 + 재무/뉴스 통합 모니터링")

watchlist_defaults = load_watchlist_tickers()
market_rows = load_market_tickers()
ticker_label_map = {
    row["ticker"]: f"{row['ticker']} - {row.get('company_name', row['ticker'])} [{row.get('market', 'UNKNOWN')}]" for row in market_rows
}

favorite_tickers = load_favorites()

with st.sidebar:
    st.header("⚙️ 커스터마이징")

    search_text = st.text_input("종목 검색 (Ticker/회사명)", value="")
    normalized_search = search_text.strip().lower()
    filtered_tickers = [
        t for t, label in ticker_label_map.items() if not normalized_search or normalized_search in label.lower() or normalized_search in t.lower()
    ]

    selected = st.multiselect(
        "모니터링 종목 (NASDAQ/NYSE/KOSPI/KOSDAQ)",
        options=filtered_tickers,
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
    st.error(f"⚠️ 이상 알림: 점수 {score_alert} 이하 종목 {', '.join(alert_df['ticker'].tolist())}")


def sector_mean_score(df: pd.DataFrame, sector: str) -> float:
    series = df[df["sector"] == sector]["score"]
    if series.empty:
        return 0.0
    return round(float(series.mean()), 2)


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("총 모니터링 종목", len(portfolio))
with col2:
    st.metric("AI 평균 점수", sector_mean_score(portfolio, "AI"))
with col3:
    st.metric("우주 평균 점수", sector_mean_score(portfolio, "SPACE"))
with col4:
    st.metric("시장 평균 점수", sector_mean_score(portfolio, "MARKET"))

st.subheader("섹터별 종목 점수카드")
display_columns = [
    "ticker_display",
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
    "ticker_display": "종목 (Ticker - 자산명)",
    "market": "시장",
    "sector": "섹터",
    "sub_sector": "소분류",
    "score": "종합점수",
    "signal": "신호",
    "sector_rank": "섹터순위",
    "pe_ratio": "P/E Ratio / 주가수익비율",
    "peg_ratio": "PEG Ratio / PEG 비율",
    "revenue_growth_yoy": "YoY Revenue Growth / 매출성장률",
    "rd_ratio": "R&D / Revenue / 연구개발비율",
    "asset_turnover": "Asset Turnover / 자산회전율",
    "tech_cycle": "Tech Cycle Position / 기술사이클",
    "pb_ratio": "P/B Ratio / 주가순자산비율",
    "operating_margin": "Operating Margin / 영업이익률",
    "debt_ratio": "Debt Ratio / 부채비율",
    "dividend_yield": "Dividend Yield / 배당수익률",
    "backlog_proxy": "Contract Pipeline (Proxy) / 수주모멘텀",
    "gov_cycle": "Gov Spending Cycle (Proxy) / 정책사이클",
}

display_df = portfolio[[c for c in display_columns if c in portfolio.columns]].copy()
non_numeric_columns = {"ticker_display", "market", "sector", "sub_sector", "signal"}
for c in display_df.columns:
    if c not in non_numeric_columns and pd.api.types.is_numeric_dtype(display_df[c]):
        display_df[c] = display_df[c].where(pd.notna(display_df[c]), "-")

st.dataframe(display_df.rename(columns=column_labels), width="stretch", hide_index=True)

heatmap_cols = [c for c in portfolio.columns if c.endswith("_normalized")]
if heatmap_cols:
    heatmap_input = portfolio.set_index("ticker_display")[heatmap_cols]
    metric_name_map = {}
    for col in heatmap_input.columns:
        metric = col.replace("_normalized", "")
        metric_name_map[col] = next((spec.label for specs in METRIC_SPECS.values() for k, spec in specs.items() if k == metric), metric)
    fig_heat = px.imshow(
        heatmap_input.rename(columns=metric_name_map),
        color_continuous_scale="RdYlGn",
        title="종목별 지표 정규화 히트맵 (0~1)",
        aspect="auto",
    )
    st.plotly_chart(fig_heat, width="stretch")

fig_score = px.bar(
    portfolio,
    x="ticker_display",
    y="score",
    color="sector",
    text="signal",
    title="섹터별 점수 및 진입/보유/매도 신호",
)
fig_score.update_traces(textposition="outside")
st.plotly_chart(fig_score, width="stretch")

holdings_data = load_holdings()
notes_data = load_notes()

tab_hold, tab_watch = st.tabs(["보유 주식", "미보유 주식"])

with tab_hold:
    st.subheader("보유 주식 입력")
    held_selection = st.multiselect(
        "보유 중인 종목 선택",
        options=portfolio["ticker"].tolist(),
        default=[t for t in holdings_data.keys() if t in set(portfolio["ticker"])],
        format_func=lambda t: ticker_label_map.get(t, t),
    )

    edited_holdings = {}
    for ticker in held_selection:
        stored = holdings_data.get(ticker, {"purchase_price": 0.0, "quantity": 0.0, "currency": "USD"})
        with st.container(border=True):
            st.markdown(f"**{ticker_label_map.get(ticker, ticker)}**")
            c1, c2, c3 = st.columns(3)
            with c1:
                purchase_price = st.number_input(f"{ticker} 매입가", min_value=0.0, value=float(stored.get("purchase_price", 0.0)), step=0.01)
            with c2:
                quantity = st.number_input(f"{ticker} 보유수량", min_value=0.0, value=float(stored.get("quantity", 0.0)), step=1.0)
            with c3:
                currency = st.selectbox(f"{ticker} 통화", ["USD", "KRW"], index=0 if stored.get("currency", "USD") == "USD" else 1)
            edited_holdings[ticker] = {"purchase_price": purchase_price, "quantity": quantity, "currency": currency}

    if st.button("보유정보 저장"):
        save_holdings(edited_holdings)
        st.success("보유 정보를 저장했습니다.")
        holdings_data = edited_holdings

    held_df = portfolio[portfolio["ticker"].isin(set(holdings_data.keys()))].copy()
    if not held_df.empty:
        held_df["판단"] = held_df["score"].apply(lambda s: recommendation_from_score(float(s), held=True))
        st.dataframe(held_df[["ticker_display", "score", "signal", "판단"]], width="stretch", hide_index=True)

with tab_watch:
    st.subheader("미보유 종목 모니터링")
    held_set = set(holdings_data.keys())
    unheld_df = portfolio[~portfolio["ticker"].isin(held_set)].copy()
    if unheld_df.empty:
        st.info("미보유 종목이 없습니다.")
    else:
        unheld_df["판단"] = unheld_df["score"].apply(lambda s: recommendation_from_score(float(s), held=False))
        st.dataframe(unheld_df[["ticker_display", "score", "signal", "판단"]], width="stretch", hide_index=True)

st.subheader("종목 메모")
editable_notes = dict(notes_data)
for ticker in portfolio["ticker"].tolist():
    key = f"memo_{ticker}"
    editable_notes[ticker] = st.text_area(
        f"{ticker_label_map.get(ticker, ticker)} 메모",
        value=str(editable_notes.get(ticker, "")),
        key=key,
        height=70,
    )
if st.button("메모 저장"):
    save_notes(editable_notes)
    st.success("메모를 저장했습니다.")

st.subheader("정성 정보 (뉴스/키워드/AI 신호)")
news_ticker = st.selectbox("뉴스 조회 종목", portfolio["ticker"].tolist(), format_func=lambda t: ticker_label_map.get(t, t))
keyword_filter = st.text_input("뉴스 키워드 필터")
news_rows = recent_news(news_ticker, keyword=keyword_filter, months=3)

if news_rows:
    news_df = pd.DataFrame(news_rows)
    st.dataframe(news_df[["datetime", "source", "headline", "url"]], width="stretch", hide_index=True)

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
                {"지표": "재무 기반 점수", "점수": base_score},
                {"지표": "뉴스 반영 AI 점수", "점수": blended},
            ]
        )
        fig_signal = px.bar(signal_df, x="지표", y="점수", range_y=[0, 100], title="AI 판단 점수")
        st.plotly_chart(fig_signal, width="stretch")
else:
    st.info("최근 3개월 뉴스가 없거나 API 키가 설정되지 않았습니다.")

with st.expander("지표 계산 근거 및 데이터 출처", expanded=False):
    for sector in ["AI", "SPACE", "MARKET"]:
        st.markdown(f"#### {sector} 섹터")
        st.dataframe(metric_basis_table(sector), width="stretch", hide_index=True)
    st.caption("데이터 연동: yfinance(가격/재무), Finnhub(세부 metric/뉴스) · API 호출 최소화를 위해 캐싱 사용")
