"""
1. 검색 키워드 생성 (Search Keyword Generation)

이 모듈은 원래 entities(개체 인식)/intent(의도 요약)/filter_terms(관련성 검증어)까지
함께 생성했지만, 실제 파이프라인에서 쓰이는 건 optimized_keywords(검색 키워드)뿐이었다.
- entities, intent: 어디서도 소비되지 않는 값이라 제거했다.
- filter_terms: main.py에서 사용자가 직접 입력한 검증어로 항상 덮어써지므로
  (관련성 검증 기준은 사용자가 통제해야 한다는 설계 결정) 여기서 생성할 이유가 없어 제거했다.

지금 이 모듈이 하는 일은 하나다: 사용자가 입력한 '관심 분야' 원문을 그대로 쓰면
표기 차이(약어/영문/정식명) 때문에 검색이 새는 경우가 있으므로, 뉴스 검색 API에
넣을 키워드 후보(원문 + 알려진 표기 변형 + 동음이의어 보정용 맥락어)를 생성한다.

기간(조회할 최근 일수)은 LLM이 추측하지 않는다. 사용자가 숫자로 직접 입력하며,
main.py에서 이 모듈의 결과(키워드)와 결합해 최종 SearchQuery를 만든다.
"""
from __future__ import annotations

import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from app.schemas import SearchQuery

_SYSTEM_PROMPT = """\
당신은 뉴스 검색 키워드를 만드는 도우미입니다.
사용자가 입력한 '관심 분야'를 보고 아래 JSON 스키마로만 응답하세요. 다른 설명은 절대 추가하지 마세요.

관심 분야는 완결된 질문이 아니라 주제어/키워드 형태로 입력될 수 있습니다.
(예: "AI 기업 제논", "반도체 업계", "서울여대")

스키마:
{{
  "optimized_keywords": ["뉴스 검색 API에 넣기 좋은 키워드 목록"]
}}

작성 규칙 (매우 중요):
1. 입력에 등장하는 고유명사(회사명/인명/제품명 등)는 반드시 원문 그대로, 단독 키워드로 1순위에 포함하세요.
   예: "AI 기업 제논" → 1순위 키워드는 "제논" (o), "AI 기업" (x, 너무 광범위함)
2. 같은 대상을 가리키지만 표기가 다를 수 있는 이름(공식 명칭, 흔히 쓰는 약어, 영문 표기)을
   확실히 알고 있는 경우에만 추가 키워드로 포함하세요. 뉴스 기사는 매체나 시점에 따라
   다른 표기를 쓸 수 있어서, 원문 표기 하나만으로 검색하면 다른 표기로 쓰인 기사를 놓칩니다.
   예: "서울여대" → ["서울여대", "서울여자대학교", "SWU"]
   예: "제논" (AI 기업) → ["제논", "GenOn"] (영문 사명을 알고 있는 경우)
   불확실하면 지어내지 말고 원문 표기만 사용하세요.
3. 동음이의어가 의심되는 짧은 고유명사(예: 원소명, 흔한 단어와 겹치는 이름)는
   "고유명사 + 업종/맥락어" 조합(예: "제논 AI")을 추가 키워드로 넣어 보정하되,
   1순위 키워드는 원문 그대로 유지하세요.
4. 키워드는 최대 4개까지 생성하세요.
5. "이슈", "동향", "소식", "최근" 같은 단어만으로 이루어진 키워드나, 이런 단어를 광범위한 카테고리
   명사와 조합한 키워드("AI 기업", "최근 이슈" 등)는 생성하지 마세요.
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
    관심 분야(주제어)로부터 검색 키워드 후보를 생성하고, 사용자가 입력한
    period_days와 결합해 SearchQuery를 만든다.

    Args:
        topic: 사용자가 입력한 관심 분야 원문 (예: "AI 기업 제논")
        period_days: 사용자가 숫자로 직접 입력한 조회 기간(일)
    """
    raw = _get_chain().invoke({"topic": topic})
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
        optimized_keywords = parsed.get("optimized_keywords") or [topic]
    except json.JSONDecodeError:
        # LLM이 형식을 어겼을 때의 안전한 폴백: 원문 주제를 그대로 키워드로 사용
        optimized_keywords = [topic]

    return SearchQuery(
        raw_question=topic,
        optimized_keywords=optimized_keywords,
        period_days=period_days,
    )