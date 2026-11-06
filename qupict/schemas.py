"""
파이프라인 전체에서 공유하는 데이터 스키마.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Sense(BaseModel):
    """[0단계] 검색어가 가리킬 수 있는 의미 하나. (예: "애플" → 기업 Apple)"""
    name: str = Field(description="다른 후보와 구분되는 짧은 이름 (예: '기업 Apple', '과일 사과')")
    definition: str = Field(description="한 줄 정의")
    aliases: list[str] = Field(
        default_factory=list,
        description="같은 대상을 가리키는 다른 표기: 정식 명칭, 약어, 영문명 (검색어 원문 제외)",
    )
    related_terms: list[str] = Field(
        default_factory=list,
        description="이 의미의 기사에 자주 같이 나오는 대표 연관어",
    )
    is_custom: bool = Field(default=False, description="사용자가 '기타'로 직접 입력한 의미인지")


class SenseCandidates(BaseModel):
    """[0단계] 검색어 하나에 대한 의미 후보 목록 (자주 쓰이는 순)."""
    query: str
    senses: list[Sense] = Field(default_factory=list)
    model: str = ""  # 후보를 만든 LLM 모델명 (재현성 기록용)

    @property
    def needs_selection(self) -> bool:
        """후보가 2개 이상일 때만 사용자가 골라야 한다. 1개면 자동 선택, 0개면 직접 입력."""
        return len(self.senses) > 1


class SelectedSense(BaseModel):
    """[0단계 결과] 사용자가 고른 의미. 이후 수집·필터링·검색의 기준이 된다."""
    query: str
    sense: Sense
    # 고르지 않은 후보들의 연관어 = 규칙 필터(방법 1-c)의 배제어
    excluded_terms: list[str] = Field(default_factory=list)
    # 선택하지 않은 후보 (LLM 검증 단계에서 "이 의미와 대조"할 때 사용)
    other_senses: list[Sense] = Field(default_factory=list)

    @property
    def match_terms(self) -> list[str]:
        """기사에서 이 대상을 찾을 때 쓸 표기 목록 (검색어 원문 + 별칭)."""
        return unique_terms([self.query, *self.sense.aliases])


def unique_terms(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


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