"""
관련성 필터 (Relevance Filtering)

검색 단계는 재현율(recall)을 높이기 위해 영문 약어 등 동음이의어 위험이 있는
키워드까지 포함해 넓게 검색한다. 그 결과 "SWU"처럼 짧은 약어가 전혀 다른 문맥의
기사와 우연히 매칭되는 경우가 생긴다.

이 모듈은 크롤링된 기사 본문에 SearchQuery.filter_terms(정식 명칭)가 실제로
등장하는지 검증하여, 약어만으로 잘못 매칭된 무관한 기사를 걸러낸다.
"""
from __future__ import annotations

from app.schemas import Article


def filter_by_relevance(articles: list[Article], filter_terms: list[str]) -> list[Article]:
    """
    filter_terms 중 하나라도 제목 또는 본문에 등장하는 기사만 남긴다.

    filter_terms가 비어 있으면(예: LLM이 채우지 못한 경우) 안전하게 필터링을 건너뛴다.
    """
    if not filter_terms:
        return articles

    normalized_terms = [term.strip().lower() for term in filter_terms if term.strip()]
    if not normalized_terms:
        return articles

    relevant: list[Article] = []
    for article in articles:
        haystack = f"{article.title} {article.content}".lower()
        if any(term in haystack for term in normalized_terms):
            relevant.append(article)

    return relevant
