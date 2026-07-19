"""
관련성 필터 (Relevance Filtering) — 임베딩 + LLM 하이브리드

기존 방식(문자열 포함 검사)의 문제: 검증어와 동의어/우회 표현을 쓰면 다 놓친다.
LLM 전수 판정의 문제: 기사 수만큼 LLM 호출이 필요해 느리고 비용이 든다.

이 모듈은 2단계로 판정한다:
  1단계 (임베딩 유사도, 빠름/저비용):
    - Article.embedding은 embed_articles()에서 이미 채워졌다고 가정 (재계산 없음)
    - "topic + validation_terms"를 하나의 쿼리 텍스트로 임베딩
    - 코사인 유사도가 high_threshold 이상이면 명확히 관련 있음 -> 통과
    - low_threshold 미만이면 명확히 무관함 -> 제거
  2단계 (LLM 정밀 판정, 느림/고비용이지만 정확):
    - low_threshold ~ high_threshold 사이(애매한 경계)의 기사만 LLM에게 재질문
    - 동음이의어처럼 표면 유사도로는 구분 안 되는 케이스를 여기서 걸러낸다

주의: high/low threshold는 임베딩 모델·도메인마다 다르므로, eval_filter.py로
라벨링된 데이터에 대해 실제로 sweep 해보고 튜닝해야 한다. 아래 기본값은 잠정치다.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from app.embeddings import get_embeddings
from app.schemas import Article
from config import settings

# NOTE: config.settings에 아래 두 값이 없다면 추가해야 한다.
#   relevance_low_threshold: float = 0.15   # 이 미만이면 명확히 무관
#   relevance_high_threshold: float = 0.35  # 이 이상이면 명확히 관련
_DEFAULT_LOW = 0.15
_DEFAULT_HIGH = 0.35

_LLM_BATCH_SIZE = 10


class _ArticleJudgment(BaseModel):
    index: int = Field(description="입력에서 주어진 기사 번호")
    is_relevant: bool = Field(description="topic과 validation_terms 기준 실제 관련 여부")


class _JudgmentBatch(BaseModel):
    judgments: list[_ArticleJudgment]


_SYSTEM_PROMPT = """너는 뉴스 기사의 주제 관련성을 판정하는 필터다.

각 기사가 아래 기준을 모두 만족할 때만 관련 있다(is_relevant=true)고 판단하라:
1. 기사가 실제로 '{topic}'에 대한 내용이다 (동음이의어나 무관한 다른 대상이 아님)
2. 기사 내용이 다음 검증 기준과 실질적으로 관련이 있다: {validation_terms}
   (검증 기준 단어가 본문에 그대로 등장하지 않아도 동의어/유사 표현/문맥으로
   관련이 있으면 relevant로 판단한다.)

이 기사들은 이미 1차 임베딩 필터에서 '애매하다'고 판정된 경계 사례들이다.
신중하게 판단하라.
"""

_USER_PROMPT_TEMPLATE = """아래 기사들을 판정하라.

{articles_block}
"""


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) or 1e-10)
    b_norm = b / (np.linalg.norm(b) or 1e-10)
    return float(np.dot(a_norm, b_norm))


def _build_articles_block(articles: list[Article], indices: list[int]) -> str:
    lines = []
    for idx, article in zip(indices, articles):
        snippet = article.content[:500]
        lines.append(f"[{idx}] 제목: {article.title}\n본문: {snippet}\n")
    return "\n".join(lines)


async def _llm_judge_borderline(
    topic: str,
    validation_terms: list[str],
    articles: list[Article],
    indices: list[int],
) -> dict[int, bool]:
    """경계 구간 기사들만 LLM에게 배치로 재질문한다."""
    from langchain_openai import ChatOpenAI  # NOTE: config에 이미 있는 클라이언트/모델명으로 교체 권장

    llm = ChatOpenAI(
        model=getattr(settings, "relevance_llm_model", "gpt-4o-mini"),
        api_key=settings.openai_api_key,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(_JudgmentBatch)

    results: dict[int, bool] = {}
    for start in range(0, len(articles), _LLM_BATCH_SIZE):
        batch_articles = articles[start : start + _LLM_BATCH_SIZE]
        batch_indices = indices[start : start + _LLM_BATCH_SIZE]

        system = _SYSTEM_PROMPT.format(
            topic=topic, validation_terms=", ".join(validation_terms)
        )
        user = _USER_PROMPT_TEMPLATE.format(
            articles_block=_build_articles_block(batch_articles, batch_indices)
        )
        try:
            result: _JudgmentBatch = await structured_llm.ainvoke(
                [("system", system), ("human", user)]
            )
            for j in result.judgments:
                results[j.index] = j.is_relevant
        except Exception:
            # 판정 실패 시 recall을 지키기 위해 통과시킨다.
            for idx in batch_indices:
                results[idx] = True

    return results


async def filter_by_relevance(
    articles: list[Article],
    topic: str,
    validation_terms: list[str],
    low_threshold: float | None = None,
    high_threshold: float | None = None,
) -> list[Article]:
    """
    1단계(임베딩) + 2단계(LLM, 경계 구간만) 하이브리드 필터.

    articles는 embed_articles()가 먼저 실행되어 embedding이 채워져 있어야 한다.
    (채워져 있지 않으면 이 함수 내부에서 관련성 판단이 불가능하므로 전체 통과시킨다.)
    """
    if not articles or not validation_terms:
        return articles

    normalized_terms = [t.strip() for t in validation_terms if t.strip()]
    if not normalized_terms:
        return articles

    low = low_threshold if low_threshold is not None else getattr(
        settings, "relevance_low_threshold", _DEFAULT_LOW
    )
    high = high_threshold if high_threshold is not None else getattr(
        settings, "relevance_high_threshold", _DEFAULT_HIGH
    )

    embedded = [a for a in articles if a.embedding is not None]
    if len(embedded) != len(articles):
        # embed_articles()가 먼저 실행되지 않은 경우: 안전하게 필터를 건너뛴다.
        return articles

    query_text = f"{topic} {' '.join(normalized_terms)}"
    query_vec = np.array(await get_embeddings().aembed_query(query_text))

    clearly_relevant: list[Article] = []
    borderline: list[Article] = []
    borderline_indices: list[int] = []

    for i, article in enumerate(embedded):
        sim = _cosine_sim(query_vec, np.array(article.embedding))
        if sim >= high:
            clearly_relevant.append(article)
        elif sim < low:
            continue  # 명확히 무관 -> 제거
        else:
            borderline.append(article)
            borderline_indices.append(i)

    if borderline:
        judgments = await _llm_judge_borderline(
            topic, normalized_terms, borderline, borderline_indices
        )
        for article, idx in zip(borderline, borderline_indices):
            if judgments.get(idx, True):
                clearly_relevant.append(article)

    return clearly_relevant