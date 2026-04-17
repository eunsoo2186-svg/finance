from __future__ import annotations

import plotly.express as px
import streamlit as st

from portfolio_data import SPACE_TICKERS, build_portfolio_dataframe, metric_basis_table

st.set_page_config(page_title="우주 섹터 분석", layout="wide")
st.title("🚀 우주 섹터 분석")

portfolio = build_portfolio_dataframe(SPACE_TICKERS)

if portfolio.empty:
    st.warning("우주 섹터 데이터를 불러오지 못했습니다.")
    st.stop()

st.dataframe(
    portfolio[["ticker", "score", "signal", "sector_rank", "pb_ratio", "operating_margin", "debt_ratio", "dividend_yield", "backlog_proxy", "gov_cycle"]],
    width="stretch",
    hide_index=True,
)

fig = px.scatter(
    portfolio,
    x="pb_ratio",
    y="operating_margin",
    size="score",
    color="signal",
    hover_name="ticker",
    title="P/B vs 영업마진 (점크기=종합점수)",
)
st.plotly_chart(fig, width="stretch")

st.subheader("지표 산식/출처")
st.dataframe(metric_basis_table("SPACE"), width="stretch", hide_index=True)
