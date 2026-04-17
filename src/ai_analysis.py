from __future__ import annotations


def recommendation_from_score(score: float, held: bool) -> str:
    if held:
        if score >= 75:
            return "보유/추매 고려"
        if score >= 50:
            return "보유"
        return "비중 축소/매도 검토"

    if score >= 75:
        return "분할 매수 시점"
    if score >= 55:
        return "관찰 후 눌림목 매수"
    return "매수 대기"


def blended_signal_score(score: float, news_count: int) -> float:
    bonus = min(10.0, float(news_count) * 0.5)
    return round(min(100.0, max(0.0, float(score) + bonus)), 2)
