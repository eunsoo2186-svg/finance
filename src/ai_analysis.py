from __future__ import annotations


def recommendation_from_score(score: float, held: bool) -> str:
    if held:
        if score >= 70:
            return "매도"
        if score >= 55:
            return "보유"
        if score >= 45:
            return "추매"
        return "손절"

    if score >= 70:
        return "매수"
    if score >= 45:
        return "관심"
    return "회피"


def blended_signal_score(score: float, news_count: int) -> float:
    bonus = min(10.0, float(news_count) * 0.5)
    return round(min(100.0, max(0.0, float(score) + bonus)), 2)
