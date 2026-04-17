from __future__ import annotations

import plotly.express as px
import streamlit as st

from portfolio_data import AI_TICKERS, build_portfolio_dataframe, metric_basis_table

st.set_page_config(page_title="AI 섹터 분석", layout="wide")
st.title("🤖 AI 섹터 분석")

portfolio = build_portfolio_dataframe(AI_TICKERS)

if portfolio.empty:
    st.warning("AI 섹터 데이터를 불러오지 못했습니다.")
    st.stop()

st.dataframe(
    portfolio[["ticker", "score", "signal", "sector_rank", "pe_ratio", "peg_ratio", "revenue_growth_yoy", "rd_ratio", "asset_turnover", "tech_cycle"]],
    width="stretch",
    hide_index=True,
)

fig = px.scatter(
    portfolio,
    x="peg_ratio",
    y="revenue_growth_yoy",
    size="score",
    color="signal",
    hover_name="ticker",
    title="PEG vs 매출성장률 (점크기=종합점수)",
)
st.plotly_chart(fig, width="stretch")

st.subheader("지표 산식/출처")
st.dataframe(metric_basis_table("AI"), width="stretch", hide_index=True)
