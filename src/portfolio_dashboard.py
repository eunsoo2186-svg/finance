from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from portfolio_data import (
    AI_TICKERS,
    DEFAULT_WEIGHTS,
    SPACE_TICKERS,
    build_portfolio_dataframe,
    metric_basis_table,
)

st.set_page_config(page_title="Portfolio Dashboard", page_icon="📊", layout="wide")
st.title("📊 포트폴리오 대시보드")
st.caption("AI/우주 섹터 종목의 객관적 재무지표 기반 비교 모니터링")

all_default = AI_TICKERS + SPACE_TICKERS

with st.sidebar:
    st.header("⚙️ 커스터마이징")
    selected = st.multiselect("모니터링 종목", all_default, default=all_default)
    custom = st.text_input("추가 종목(쉼표 구분)", value="")

    ai_weights = {}
    with st.expander("AI 가중치 조정", expanded=False):
        for key, default in DEFAULT_WEIGHTS["AI"].items():
            ai_weights[key] = st.slider(f"AI:{key}", 0.0, 1.0, float(default), 0.01)

    space_weights = {}
    with st.expander("우주 가중치 조정", expanded=False):
        for key, default in DEFAULT_WEIGHTS["SPACE"].items():
            space_weights[key] = st.slider(f"SPACE:{key}", 0.0, 1.0, float(default), 0.01)

    score_alert = st.slider("이상 알림 점수 기준", 0, 100, 45)

symbols = selected + [s.strip().upper() for s in custom.split(",") if s.strip()]
portfolio = build_portfolio_dataframe(symbols, {"AI": ai_weights, "SPACE": space_weights})

if portfolio.empty:
    st.warning("표시 가능한 AI/우주 섹터 종목이 없습니다.")
    st.stop()

alert_df = portfolio[portfolio["score"] <= score_alert]
if not alert_df.empty:
    st.error(f"⚠️ 이상 알림: 점수 {score_alert} 이하 종목 {', '.join(alert_df['ticker'].tolist())}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("총 모니터링 종목", len(portfolio))
with col2:
    st.metric("AI 평균 점수", round(portfolio[portfolio["sector"] == "AI"]["score"].mean(), 2))
with col3:
    st.metric("우주 평균 점수", round(portfolio[portfolio["sector"] == "SPACE"]["score"].mean(), 2))

st.subheader("섹터별 종목 점수카드")

display_columns = [
    "ticker",
    "sector",
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

st.dataframe(
    portfolio[[c for c in display_columns if c in portfolio.columns]],
    width="stretch",
    hide_index=True,
)

heatmap_input = portfolio.set_index("ticker")[[c for c in portfolio.columns if c.endswith("_normalized")]]
heatmap_renamed = heatmap_input.rename(columns=lambda x: x.replace("_normalized", ""))
fig_heat = px.imshow(
    heatmap_renamed,
    color_continuous_scale="RdYlGn",
    title="종목별 지표 정규화 히트맵 (0~1)",
    aspect="auto",
)
st.plotly_chart(fig_heat, width="stretch")

fig_score = px.bar(
    portfolio,
    x="ticker",
    y="score",
    color="sector",
    text="signal",
    title="섹터별 점수 및 진입/보유/매도 신호",
)
fig_score.update_traces(textposition="outside")
st.plotly_chart(fig_score, width="stretch")

with st.expander("지표 계산 근거 및 데이터 출처", expanded=False):
    st.markdown("#### AI 섹터")
    st.dataframe(metric_basis_table("AI"), width="stretch", hide_index=True)
    st.markdown("#### 우주 섹터")
    st.dataframe(metric_basis_table("SPACE"), width="stretch", hide_index=True)
    st.caption("데이터 연동: yfinance(가격/기본 재무), Finnhub(세부 metric) · API 호출 최소화를 위해 내부 캐싱 사용")
