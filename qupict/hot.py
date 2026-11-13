"""
많이 보도된 기사 — 수집한 기사 중 중복(비슷한 기사)이 가장 많이 묶인 대표 기사 Top N + 한 줄 요약

2단계 중복 제거에서 "비슷한 기사"는 대표 기사 하나에 묶인다 (article_topics.duplicate_of / FilterOutcome.duplicates).
묶인 기사가 많다 = 여러 언론사가 같은 내용을 보도했다 = 그 기간의 주요 이슈로 본다.

    top_duplicated_db(topic_id, period_days)      DB 버전: article_topics에서 센다
    top_duplicated(articles, duplicates)          메모리 버전: {중복 link: 대표 link}에서 센다
    summarize(hot)                                LLM 한 번으로 기사마다 한 줄 요약을 채운다

웹 화면(web.py)의 "3 수집·필터링"과 "4 질문하기" 사이에 보여 준다.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from pydantic import BaseModel, Field

from qupict.schemas import Article

TOP_N = 5
ANSWER_STATUSES = ("관련", "보류")     # 질문 답변에 쓰는 기사와 같은 판정만 (다른 의미의 기사는 보여 주지 않는다)
SUMMARY_BODY_CHARS = 1500              # 요약할 때 LLM에 보내는 본문 길이 (기사당)
SUMMARY_MAX_CHARS = 120                # 한 줄 요약 길이 상한 (화면에서 한두 줄)


@dataclass
class HotArticle:
    title: str
    link: str
    press: Optional[str]
    pub_date: Optional[datetime]
    description: str        # 네이버 검색 API가 준 요약문 (LLM 요약이 실패했을 때 대신 쓴다)
    content: str
    duplicates: int         # 이 기사에 묶인 비슷한 기사 수 (대표 기사 자신은 빼고)
    summary: str = ""       # 한 줄 요약 (summarize가 채운다)


# ---------------------------------------------------------------------------
# 중복이 많이 묶인 대표 기사 찾기
# ---------------------------------------------------------------------------

def top_duplicated_db(topic_id: int, period_days: Optional[int] = None, limit: int = TOP_N,
                      statuses: Iterable[str] = ANSWER_STATUSES) -> list[HotArticle]:
    """[DB 버전] topic의 대표 기사 중 중복이 많이 묶인 순으로 limit건. 중복이 하나도 없는 기사는 빠진다.
    period_days: 대표 기사 발행일이 최근 며칠 이내인 것만 (질문 검색과 같은 기간)."""
    from qupict import db

    params: dict = {"topic_id": topic_id, "statuses": list(statuses), "limit": limit}
    period = ""
    if period_days is not None:
        now = datetime.now(timezone.utc)
        period = "AND a.pub_date BETWEEN %(since)s AND %(until)s"
        params.update(since=now - timedelta(days=period_days), until=now)
    with db.connect() as conn:
        rows = conn.execute(
            f"""SELECT a.title, a.link, a.press, a.pub_date, a.description, a.content,
                       count(*) AS duplicates
                FROM article_topics at
                JOIN articles a ON a.id = at.article_id
                JOIN article_topics d ON d.topic_id = at.topic_id AND d.duplicate_of = at.article_id
                WHERE at.topic_id = %(topic_id)s AND at.duplicate_of IS NULL
                  AND at.status = ANY(%(statuses)s) {period}
                GROUP BY a.id
                ORDER BY duplicates DESC, a.pub_date DESC NULLS LAST, a.id
                LIMIT %(limit)s""",
            params,
        ).fetchall()
    return [HotArticle(title=r["title"], link=r["link"], press=r["press"], pub_date=r["pub_date"],
                       description=r["description"] or "", content=r["content"] or "",
                       duplicates=r["duplicates"]) for r in rows]


def top_duplicated(articles: list[Article], duplicates: dict[str, str], limit: int = TOP_N) -> list[HotArticle]:
    """[메모리 버전] articles(질문에 쓰는 대표 기사) 중 중복이 많이 묶인 순으로 limit건.
    duplicates = {중복 기사 link: 대표 기사 link} (FilterOutcome.duplicates)."""
    counts = Counter(duplicates.values())
    hot = [a for a in articles if counts.get(a.link)]
    hot.sort(key=lambda a: (-counts[a.link], -(a.pub_date.timestamp() if a.pub_date else 0)))
    return [HotArticle(title=a.title, link=a.link, press=a.press, pub_date=a.pub_date,
                       description=a.description, content=a.content, duplicates=counts[a.link])
            for a in hot[:limit]]


# ---------------------------------------------------------------------------
# 한 줄 요약 (LLM 한 번)
# ---------------------------------------------------------------------------

class _Summary(BaseModel):
    index: int = Field(description="기사 번호")
    summary: str = Field(description="기사 핵심 내용 한 문장 (60자 안팎)")


class _SummaryBatch(BaseModel):
    summaries: list[_Summary]


_SUMMARY_SYSTEM_PROMPT = """\
너는 뉴스 기사를 한 줄로 요약한다.

규칙:
- 기사마다 핵심 내용(누가 · 무엇을 · 어떻게 됐는지)을 한국어 한 문장, 60자 안팎으로 적어라.
- 기사에 있는 내용만 써라. 추측하거나 평가를 덧붙이지 마라.
- 제목을 그대로 베끼지 말고 본문 내용을 담아라.
- 입력된 모든 기사 번호에 대해 하나씩 답하라.
"""


def _one_line(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= SUMMARY_MAX_CHARS else text[:SUMMARY_MAX_CHARS - 1].rstrip() + "…"


def _fallback_summary(h: HotArticle) -> str:
    """LLM 요약이 없을 때: 네이버 요약문, 없으면 본문 첫 문장."""
    if h.description.strip():
        return _one_line(h.description)
    first = re.split(r"(?<=[.!?다])\s", h.content.strip(), maxsplit=1)[0]
    return _one_line(first)


def summarize(hot: list[HotArticle]) -> list[HotArticle]:
    """기사마다 한 줄 요약(summary)을 채운다. LLM 요청은 한 번 (기사 전체를 묶어서).
    실패하거나 답이 빠진 기사는 네이버 요약문(없으면 본문 첫 문장)으로 대신한다 → 화면은 항상 채워진다."""
    if not hot:
        return hot
    from qupict.llm import get_structured_model

    block = "\n\n".join(f"[{i}] 제목: {h.title}\n본문: {h.content[:SUMMARY_BODY_CHARS] or h.description}"
                        for i, h in enumerate(hot))
    try:
        result: _SummaryBatch = get_structured_model(_SummaryBatch).invoke(
            [("system", _SUMMARY_SYSTEM_PROMPT), ("human", f"아래 기사들을 요약하라.\n\n{block}")])
        for s in result.summaries:
            if 0 <= s.index < len(hot) and s.summary.strip():
                hot[s.index].summary = _one_line(s.summary)
    except Exception as e:
        print(f"  [경고] 많이 보도된 기사 요약 실패 ({type(e).__name__}) — 기사 요약문으로 대신합니다")
    for h in hot:
        if not h.summary:
            h.summary = _fallback_summary(h)
    return hot
