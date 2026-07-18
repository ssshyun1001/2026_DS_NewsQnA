"""
1. 주제 분석 (Topic Analysis)

사용자가 입력한 '관심 분야'에서 핵심 개체(Entity)와 의도(Intent)를 파악하고,
News Search API에 최적화된 검색 키워드(optimized_keywords)와
동음이의어 검증용 정식 명칭(filter_terms)을 함께 생성한다.

기간(조회할 최근 일수)은 LLM이 추측하지 않는다. 사용자가 숫자로 직접 입력하며,
main.py에서 이 모듈의 결과(키워드/필터어)와 결합해 최종 SearchQuery를 만든다.
"""
from __future__ import annotations

import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from app.schemas import SearchQuery

_SYSTEM_PROMPT = """\
당신은 뉴스 검색 쿼리 최적화 전문가입니다.
사용자가 입력한 '관심 분야'를 분석하여 아래 JSON 스키마로만 응답하세요. 다른 설명은 절대 추가하지 마세요.

관심 분야는 완결된 질문이 아니라 주제어/키워드 형태로 입력될 수 있습니다.
(예: "AI 기업 제논", "반도체 업계", "서울여대")

스키마:
{{
  "entities": ["핵심 개체명 목록"],
  "intent": "이 관심 분야로 어떤 뉴스를 모으고 싶은지 한 문장 요약",
  "optimized_keywords": ["뉴스 검색 API에 넣기 좋은 키워드 1~2개"],
  "filter_terms": ["크롤링된 기사 본문에 실제로 등장해야 하는 정식 명칭/핵심어 목록 (관련성 검증용)"]
}}

optimized_keywords 작성 규칙 (매우 중요):
1. 입력에 등장하는 고유명사(회사명/인명/제품명 등)는 반드시 원문 그대로, 단독 키워드로 1순위에 포함하세요.
   예: "AI 기업 제논" → 1순위 키워드는 "제논" (o), "AI 기업" (x, 너무 광범위함)
2. 키워드는 최대 2개까지만 생성하세요. 대부분의 입력은 1개 키워드로 충분합니다.
3. "이슈", "동향", "소식", "최근" 같은 단어만으로 이루어진 키워드나, 이런 단어를 광범위한 카테고리
   명사와 조합한 키워드("AI 기업", "최근 이슈" 등)는 생성하지 마세요.
4. 동음이의어가 의심되는 짧은 고유명사(예: 원소명, 흔한 단어와 겹치는 이름)는 2순위 키워드로
   "고유명사 + 업종/맥락어" 조합(예: "제논 AI")을 추가해 보정하되, 1순위 키워드는 원문 그대로 유지하세요.

filter_terms 작성 규칙:
- entities에 등장한 정식 명칭(약어의 풀네임 등)을 그대로 넣으세요.
- 예: "제논" → filter_terms = ["제논"] (기사 본문에 실제로 "제논"이 3번 이상 언급돼야 통과)
- 확신이 없으면 optimized_keywords의 1순위 키워드와 동일하게 채우세요.
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "관심 분야: {topic}"),
    ]
)

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        llm = ChatOpenAI(model=settings.chat_model, api_key=settings.openai_api_key, temperature=0)
        _chain = _prompt | llm | StrOutputParser()
    return _chain


def analyze_topic(topic: str, period_days: int) -> SearchQuery:
    """
    관심 분야(주제어)를 분석해 검색 키워드/관련성 검증어를 뽑고, 사용자가 입력한
    period_days와 결합해 최종 SearchQuery를 생성한다.

    Args:
        topic: 사용자가 입력한 관심 분야 원문 (예: "AI 기업 제논")
        period_days: 사용자가 숫자로 직접 입력한 조회 기간(일)
    """
    raw = _get_chain().invoke({"topic": topic})
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # LLM이 형식을 어겼을 때의 안전한 폴백: 원문 주제를 그대로 키워드/필터어로 사용
        parsed = {
            "entities": [],
            "intent": topic,
            "optimized_keywords": [topic],
            "filter_terms": [topic],
        }

    optimized_keywords = parsed.get("optimized_keywords") or [topic]
    filter_terms = parsed.get("filter_terms") or [optimized_keywords[0]]

    return SearchQuery(
        raw_question=topic,
        entities=parsed.get("entities", []),
        intent=parsed.get("intent", ""),
        optimized_keywords=optimized_keywords,
        filter_terms=filter_terms,
        period_days=period_days,
    )