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


class NewsMeta(BaseModel):
    """[1단계] 네이버 뉴스 검색 API가 돌려준 기사 정보 (본문 크롤링 전)."""
    title: str
    link: str                          # 언론사 원문 주소 (출처 표시·중복 확인 기준)
    naver_link: Optional[str] = None   # 네이버 뉴스 주소 (있으면 본문을 여기서 가져온다)
    press: Optional[str] = None
    pub_date: Optional[datetime] = None
    description: str = ""              # API가 주는 요약문
    search_term: str = ""              # 이 기사를 찾은 검색어 (예: "눈 안과")


class ArticleImage(BaseModel):
    """[1단계] 기사 본문 속 이미지. 이미지 → 텍스트 설명은 4단계에서 만든다."""
    url: str
    caption: str = ""                  # 기사에 원래 달린 사진 설명


class Article(BaseModel):
    """[1단계 결과] 본문까지 가져온 기사."""
    title: str
    link: str
    press: Optional[str] = None
    pub_date: Optional[datetime] = None
    content: str = ""
    description: str = ""
    search_term: str = ""
    images: list[ArticleImage] = Field(default_factory=list)
    collected_at: Optional[datetime] = None  # 수집 시각 (검색 결과는 날짜마다 달라지므로 기록)
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