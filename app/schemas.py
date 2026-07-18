"""
파이프라인 전체에서 공유하는 데이터 스키마.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Query Understanding 단계의 출력."""
    raw_question: str
    entities: list[str] = Field(default_factory=list)
    intent: str = ""
    optimized_keywords: list[str] = Field(default_factory=list)
    filter_terms: list[str] = Field(default_factory=list)
    period_days: int = 7


class NewsMeta(BaseModel):
    """News Search API가 반환하는 메타데이터 (본문 크롤링 전)."""
    title: str
    link: str
    press: Optional[str] = None
    pub_date: Optional[datetime] = None
    description: str = ""


class Article(BaseModel):
    """크롤링까지 완료된 기사 본문."""
    title: str
    link: str
    press: Optional[str] = None
    pub_date: Optional[datetime] = None
    content: str = ""
    embedding: Optional[list[float]] = None

    def to_context_block(self, idx: int) -> str:
        return (
            f"[기사 {idx}]\n"
            f"제목: {self.title}\n"
            f"언론사: {self.press or '알 수 없음'}\n"
            f"URL: {self.link}\n"
            f"본문:\n{self.content}\n"
        )


class Answer(BaseModel):
    """최종 답변 결과."""
    question: str
    answer: str
    sources: list[Article]