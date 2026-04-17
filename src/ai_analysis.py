from __future__ import annotations

from portfolio_data import build_entry_analysis, build_exit_analysis


def recommendation_from_score(score: float, held: bool) -> str:
    score = float(score)
    if held:
        if score < 35:
            return "손절"
        if score < 50:
            return "비중축소"
        if score < 70:
            return "보유"
        return "부분익절"

    if score >= 75:
        return "매수"
    if score >= 55:
        return "관심"
    if score >= 45:
        return "관찰"
    return "회피"


def blended_signal_score(score: float, news_count: int, sentiment_ratio: float = 0.5) -> float:
    bonus = min(10.0, float(news_count) * 0.5 * max(0.0, float(sentiment_ratio)))
    return round(min(100.0, max(0.0, float(score) + bonus)), 2)


def entry_signal_detail(symbol: str, metrics: dict, score: float, news_count: int = 0, sentiment_ratio: float = 0.5) -> dict:
    return build_entry_analysis(symbol, metrics, score, news_count=news_count, sentiment_ratio=sentiment_ratio)


def exit_signal_detail(symbol: str, metrics: dict, profit_rate: float | None) -> dict:
    return build_exit_analysis(symbol, metrics, profit_rate)
