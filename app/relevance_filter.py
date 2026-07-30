"""
관련성 필터 (Relevance Filtering) — 제외 키워드(동음이의어) + 직접언급 보호

역할 분담을 명확히 나눴다:

  [LLM 담당] 동음이의어(homonym)만 생성한다.
    topic이 완전히 다른 의미/개체를 가리키는 경우만 잡는다.
    예: "애플" -> 과일 사과, "강남" -> 지역 강남구, "당근" -> 채소 당근
    같은 카테고리의 다른 기관/경쟁사(예: "서울대"의 제외어로 "서울시립대")는
    LLM이 절대 생성하지 않는다 (recall을 해치는 주범이었기 때문).

  [규칙 담당] "서울여대 검색 시 이화여대/숙명여대 등이 섞여 들어오는" 유형의
    분리 검색 노이즈는 LLM이 예측하지 않고, 대신 "이 기사에 topic 문자열이
    통째로 등장하는가"를 보는 규칙(직접 언급 보호, guard)에서 처리한다.
    - topic을 정규화(공백/특수문자 제거)한 문자열이 기사에 부분 문자열로
      그대로 포함되어 있으면 무조건 통과시킨다.
    - 예: topic="서울여대"인 기사에 "서울"과 "여대"가 각각 등장해도,
      "서울여대"라는 연속 문자열 자체가 없으면(예: "서울에 있는 이화여대")
      보호 대상이 아니다. 단어 단위로 쪼개 느슨하게 매칭하는 완화는 두지 않는다.

필터링 단계:
  0단계: 제외 키워드(동음이의어) LLM 생성, 세션당 1회
  1단계: (직접 언급 보호) topic 문자열이 기사에 통째로 등장하면 무조건 통과
    (임베딩/LLM 판정 생략)
  2단계: (임베딩) 나머지 기사에 대해 제외어와의 코사인 유사도로 명확한
    케이스는 즉시 컷/통과, 애매한 경계만 3단계로 넘긴다. 이때 제외어를
    그대로 임베딩하지 않고, topic 단어를 제거한 맥락어만 임베딩한다 —
    동음이의어(예: "당근" 회사 vs 채소)는 topic 단어 자체가 똑같이 겹치므로,
    "당근 재배"를 그대로 임베딩하면 "당근"이라는 글자 때문에 진짜 관련
    기사까지 유사도가 높게 나와 잘못 제외될 수 있기 때문이다.
  3단계: (LLM 정밀 판정) 경계 구간 기사만 재판정.
"""
from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel, Field

from app.embeddings import get_embeddings
from app.schemas import Article
from config import settings

_DEFAULT_EXCLUSION_LOW = 0.20
_DEFAULT_EXCLUSION_HIGH = 0.35
_LLM_BATCH_SIZE = 10



# ---------------------------------------------------------------------------
# 0단계: 제외 키워드(동음이의어) 자동 생성 — LLM은 이 유형만 담당한다
# ---------------------------------------------------------------------------

class _ExclusionTerms(BaseModel):
    exclusion_terms: list[str] = Field(default_factory=list)


_HOMONYM_SYSTEM_PROMPT = """\
당신은 뉴스 검색어의 동음이의어를 구분하기 위한 제외 표현을 생성합니다.

topic과 validation_terms를 함께 보고 사용자가 의도한 대상을 추론하세요.
사용자가 의도한 대상과 의미가 명확하게 다른 동음이의어 표현만 exclusion_terms로 생성하세요.

규칙:
1. 동일 대상을 가리키는 정식 명칭, 약어, 이전 명칭, 별칭, 영문명, 서비스명은 절대 넣지 마세요.
2. 단순히 같은 지역, 업종 또는 범주에 속하는 다른 기관이나 경쟁 대상을 넣지 마세요.
   - 대학 검색에서 다른 대학 이름을 넣지 마세요.
   - 기업 검색에서 경쟁 기업 이름을 넣지 마세요.
   - 관련 기사에서 함께 등장할 가능성이 있는 대상은 넣지 마세요.
3. topic 전체나 다른 고유명사의 일부 단어만 넣지 마세요.
   - `서울대`의 제외어로 `서울`, `대학교`를 넣지 마세요.
4. 사용자가 의도한 대상과 의미가 완전히 다른 동음이의어만 생성하세요.
   - 기업 애플과 과일 사과
   - 지역 강남과 방송인 강남
   - 기업 당근과 채소 당근
5. 동일 개체인지 확실하지 않거나 관련 기사에 함께 등장할 가능성이 있다면 넣지 마세요.
6. 확실한 동음이의어가 없으면 빈 리스트를 반환하세요.
7. 최대 6개까지 생성하고 JSON만 출력하세요.

예시 1
Input:
{{
"topic": "애플",
"validation_terms": ["기업", "아이폰", "IT"]
}}
Output:
{{
"exclusion_terms": [
"사과 재배",
"사과 농가",
"사과 가격",
"사과 품종",
"사과 수확"
]
}}

예시 2
Input:
{{
"topic": "서울대",
"validation_terms": ["서울대학교", "국립대학교", "대학 연구"]
}}
Output:
{{
"exclusion_terms": []
}}
설명:
서울과기대, 서울시립대, 서울여자대학교는 다른 대학이지만 서울대학교 관련 기사에
함께 등장할 수 있으므로 제외어로 사용하지 않습니다. (이런 "동종 카테고리" 노이즈는
LLM이 아니라 별도의 규칙으로 처리합니다.)

예시 3
Input:
{{
"topic": "당근",
"validation_terms": ["회사"]
}}
Output:
{{
"exclusion_terms": [
"당근 재배",
"당근 농가",
"당근 가격",
"당근 효능",
"당근 요리법"
]
}}
설명:
"당근마켓"은 topic "당근"의 이전 정식 명칭(서비스명)이며 같은 회사를 가리키므로
절대 제외어에 넣지 않습니다. "당근"과 이름이 비슷한 다른 회사(경쟁사 등)도
넣지 않습니다. 오직 채소 "당근"(먹는 작물)이라는 완전히 다른 의미의 동음이의어만
제외어로 생성합니다.
"""

_HOMONYM_FALLBACK_SYSTEM_PROMPT = """\
당신은 뉴스 검색어의 동음이의어를 구분하기 위한 제외 표현을 생성합니다.
직전 시도에서는 topic에 대해 확실한 동음이의어를 찾지 못했습니다.

이번에는 확신이 낮아도 괜찮으니, topic과 의미가 다른 대상으로 뉴스에 등장할
가능성이 있는 표현을 최소 1개는 반드시 제시하세요. 다만 아래 규칙은 그대로
지켜야 합니다.

규칙:
1. 동일 대상을 가리키는 정식 명칭, 약어, 이전 명칭, 별칭, 영문명, 서비스명은 절대 넣지 마세요.
2. 같은 지역, 업종 또는 범주에 속하는 다른 기관이나 경쟁 대상을 넣지 마세요.
3. topic 전체나 다른 고유명사의 일부 단어만 넣지 마세요.
4. topic과 validation_terms를 이어붙인 표현(예: "topic validation_term",
   "validation_term topic")은 넣지 마세요. 이는 의도한 대상 자체를 가리키는
   말이지 다른 개체가 아닙니다.
5. 확신이 낮더라도 최소 1개의 후보는 반드시 JSON으로 출력하세요.
"""

_homonym_llm = None


def _get_homonym_chain():
    global _homonym_llm
    if _homonym_llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=getattr(settings, "relevance_llm_model", None) or settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        _homonym_llm = llm.with_structured_output(_ExclusionTerms)
    return _homonym_llm


async def _try_generate_homonym_terms(
    topic: str,
    validation_terms: list[str],
    system_prompt: str,
) -> list[str]:
    user_prompt = (
        '{"topic": "%s", "validation_terms": %s}' % (topic, list(validation_terms))
    )
    try:
        result: _ExclusionTerms = await _get_homonym_chain().ainvoke(
            [("system", system_prompt), ("human", user_prompt)]
        )
        return [t.strip() for t in result.exclusion_terms if t.strip()]
    except Exception:
        return []


async def generate_exclusion_terms(
    topic: str,
    validation_terms: list[str] | None = None,
) -> list[str]:
    """topic의 동음이의어 제외어만 LLM으로 생성한다.

    "서울여대 vs 이화여대"류의 동종 카테고리 혼동은 여기서 다루지 않는다.
    그 문제는 filter_by_relevance() 안의 직접 언급 보호 규칙(topic 문자열이
    기사에 통째로 등장하는지 확인)이 맡는다 (LLM 예측 없이도 recall을
    지킬 수 있기 때문).

    LLM 출력 후처리:
      topic과 validation_terms를 이어붙인 표현(예: topic="강남",
      validation_terms=["방송인"] -> "방송인 강남", "강남 방송인")은
      "다른 개체"가 아니라 오히려 사용자가 의도한 대상 그 자체를 가리키는
      말이다. 프롬프트에서 이런 조합을 만들지 말라고 지시하지만, LLM이
      가끔 이를 어기고 생성하므로 사후에 한 번 더 걸러낸다.

    최소 1개 보장(best-effort):
      제외어가 하나도 없으면 필터가 사실상 아무 일도 하지 않으므로, 1차
      시도 결과가 (사후 필터링 후) 비어 있으면 더 관대한 프롬프트로 한 번
      더 시도한다. 그래도 비어 있으면 정말로 마땅한 후보가 없다는 뜻이므로
      빈 리스트를 그대로 반환한다 (없는 동음이의어를 억지로 지어내지는 않는다).
    """
    validation_terms = validation_terms or []

    raw_terms = await _try_generate_homonym_terms(topic, validation_terms, _HOMONYM_SYSTEM_PROMPT)
    terms = _drop_target_description_terms(raw_terms, topic, validation_terms)
    if terms:
        return terms

    raw_terms_retry = await _try_generate_homonym_terms(
        topic, validation_terms, _HOMONYM_FALLBACK_SYSTEM_PROMPT
    )
    return _drop_target_description_terms(raw_terms_retry, topic, validation_terms)


def _drop_target_description_terms(
    terms: list[str],
    topic: str,
    validation_terms: list[str],
) -> list[str]:
    """topic과 validation_terms를 이어붙인 조합과 일치하는 항목을 제거한다.

    예: topic="강남", validation_terms=["방송인"]일 때
        "방송인 강남", "강남 방송인" -> 제거 (같은 대상을 가리키는 표현)
        "강남구", "강남역" -> 유지 (실제로 다른 개체를 가리키는 동음이의어)
    """
    normalized_topic = _normalize_for_match(topic)
    normalized_validations = [
        _normalize_for_match(v) for v in validation_terms if _normalize_for_match(v)
    ]

    target_combos = {normalized_topic}
    for v in normalized_validations:
        target_combos.add(normalized_topic + v)
        target_combos.add(v + normalized_topic)

    return [t for t in terms if _normalize_for_match(t) not in target_combos]


# ---------------------------------------------------------------------------
# 직접 언급 보호(guard): topic 문자열 정확 일치
# ---------------------------------------------------------------------------

def _normalize_for_match(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def _mentions_topic_directly(article: Article, topic: str) -> bool:
    """기사에 topic 문자열 전체가 그대로(붙어 있는 하나의 이름으로) 등장하는지 판정한다.

    예: topic="서울여대"인 기사에서 "서울"과 "여대"가 각각 등장하더라도,
    "서울여대"라는 이름 자체가 없으면 직접 언급으로 인정하지 않는다
    (예: "서울에 있는 이화여대..." 기사는 "서울"과 "여대"는 포함하지만
    topic의 풀네임 "서울여대" 자체는 없으므로 보호 대상이 아니다).

    단어를 쪼개 부분적으로 겹치는지 보는 게 아니라, topic 문자열(정규화:
    공백/특수문자 제거, 소문자화)이 기사 안에 그대로 들어있는지만 확인하는
    단순한 규칙이다.
    """
    normalized_topic = _normalize_for_match(topic)
    if not normalized_topic:
        return False

    article_text = _normalize_for_match(f"{article.title} {article.content}")
    return normalized_topic in article_text


# ---------------------------------------------------------------------------
# 제외어 임베딩용 전처리: topic 단어 제거 (표면 글자 낚임 방지)
# ---------------------------------------------------------------------------

def _strip_topic_words(text: str, topic: str) -> str:
    """text에서 topic 단어(부분 문자열)를 제거하고 남은 맥락어만 반환한다.

    동음이의어(예: topic="당근")의 경우, 제외어 문자열("당근 재배")을
    그대로 임베딩하면 "의미"가 아니라 "글자가 같다"는 이유만으로 진짜
    관련 기사(회사 '당근')와도 유사도가 높게 나와 잘못 제외되는 문제가
    있다. topic 단어를 지우고 남은 맥락어("재배")만 임베딩하면, "채소로서의
    맥락"만 순수하게 담을 수 있어 이 문제를 줄일 수 있다.
    """
    if not topic:
        return text
    stripped = re.sub(re.escape(topic), " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", stripped).strip()


def _build_context_only_query(exclusion_terms: list[str], topic: str) -> str:
    """제외어 목록에서 topic 단어를 제거한 맥락어만 모아 임베딩용 쿼리 문자열을 만든다.

    topic을 제거했을 때 완전히 빈 문자열이 되는 항목(예: 제외어 자체가
    topic 단어 하나뿐인 경우)은 정보 손실을 피하기 위해 원본 그대로 남긴다.
    """
    context_terms = []
    for term in exclusion_terms:
        stripped = _strip_topic_words(term, topic)
        context_terms.append(stripped if stripped else term)
    return " ".join(context_terms)


# ---------------------------------------------------------------------------
# 임베딩 유사도
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) or 1e-10)
    b_norm = b / (np.linalg.norm(b) or 1e-10)
    return float(np.dot(a_norm, b_norm))


# ---------------------------------------------------------------------------
# LLM 경계 판정
# ---------------------------------------------------------------------------

class _ArticleJudgment(BaseModel):
    index: int = Field(description="입력에서 주어진 기사 번호")
    is_relevant: bool = Field(
        description="이 기사가 실제로 topic을 다루면 true, "
                    "제외 기준(exclusion_terms)에 해당하는 다른 개체를 다루면 false"
    )


class _JudgmentBatch(BaseModel):
    judgments: list[_ArticleJudgment]


_SYSTEM_PROMPT = """너는 뉴스 기사가 '{topic}'을 실제로 다루는지 판정하는 필터다.

각 기사는 1차 임베딩 필터에서 '애매하다'고 판정된 경계 사례들이다. 즉,
'{topic}'과 혼동되는 다음 대상들 중 하나를 다루고 있을 가능성이 있는 기사들이다:
{exclusion_terms}

판정 기준:
- 기사가 실제로 '{topic}'에 대한 내용이면 -> is_relevant: true
- 기사가 위에 나열된 혼동 대상(동음이의어)에 대한 내용이면 -> is_relevant: false
- 기사에 '{topic}'과 혼동 대상이 함께 언급되더라도, 기사의 실제 주제가
  '{topic}'이면 관련 있다고 판단하라.

신중하게 판단하라.
"""

_USER_PROMPT_TEMPLATE = """아래 기사들을 판정하라.

{articles_block}
"""


def _build_articles_block(articles: list[Article], indices: list[int]) -> str:
    lines = []
    for idx, article in zip(indices, articles):
        snippet = article.content[:500]
        lines.append(f"[{idx}] 제목: {article.title}\n본문: {snippet}\n")
    return "\n".join(lines)


async def _llm_judge_borderline(
    topic: str,
    exclusion_terms: list[str],
    articles: list[Article],
    indices: list[int],
) -> dict[int, bool]:
    from langchain_openai import ChatOpenAI

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
            topic=topic,
            exclusion_terms=", ".join(exclusion_terms),
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
            for idx in batch_indices:
                results[idx] = True

    return results


# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------

async def filter_by_direct_mention(articles: list[Article], topic: str) -> list[Article]:
    """topic 문자열이 기사에 통째로 있는 기사만 남기는 독립적인 필터.

    V1(동음이의어 제외어 필터)과는 완전히 다른 문제를 푼다:
      V1: "강남구" vs "방송인 강남"처럼 topic과 철자가 같은데 의미가 다른
          동음이의어를 구분하는 문제.
      이 함수: "서울여대"를 검색했는데 "서울"과 "여대"가 각각 매칭돼서
          "이화여대" 같은 철자가 다른 유사 기관 기사가 섞여 들어오는 문제.
          이 경우 두 대상은 애초에 철자 자체가 다르므로(동음이의어가 아님)
          임베딩/LLM 판정이 필요 없고, "topic 문자열이 실제로 있는가"만
          보면 충분하다.

    동음이의어 제외어나 임베딩은 전혀 쓰지 않는다. topic 문자열이 없는 기사는
    무조건 제외되므로, topic을 표현만 살짝 바꿔 쓴(paraphrase) 관련 기사까지
    걸러낼 수 있다는 점에 유의해야 한다 — 그 트레이드오프를 감수하는 필터다.
    """
    return [a for a in articles if _mentions_topic_directly(a, topic)]


async def filter_by_relevance(
    articles: list[Article],
    topic: str,
    validation_terms: list[str] | None = None,
    exclusion_terms: list[str] | None = None,
    exclusion_low_threshold: float | None = None,
    exclusion_high_threshold: float | None = None,
    apply_direct_mention_guard: bool = True,
) -> list[Article]:
    """0단계(제외어 자동 생성) + 직접 언급 보호(topic 문자열 정확 일치)
    + 임베딩 + LLM 경계 판정.

    validation_terms는 exclusion_terms를 자동 생성할 때(동음이의어 판정
    정확도용)만 쓰이고, exclusion_terms를 직접 넘기면 무시된다.

    apply_direct_mention_guard=True(기본값, 실서비스에서 항상 이 값을 씀):
      topic 문자열이 기사에 통째로 등장하면 제외어 판정과 무관하게 무조건
      통과시킨다.
    apply_direct_mention_guard=False:
      이 가드를 건너뛰고 순수하게 제외어(동음이의어) 기반 필터링만 적용한다.
      eval_filter.py에서 "제외어 필터 자체의 효과(V1)"와 "직접 언급 보호를
      추가했을 때의 효과(V2)"를 분리해서 측정하기 위한 용도다.
    """
    if not articles:
        return articles

    embedded = [a for a in articles if a.embedding is not None]
    if len(embedded) != len(articles):
        return articles

    if exclusion_terms is None:
        exclusion_terms = await generate_exclusion_terms(topic, validation_terms)
    normalized_exclusions = [t.strip() for t in exclusion_terms if t.strip()]

    kept: list[Article] = []
    remaining: list[Article] = []

    for article in embedded:
        # 직접 언급 보호: topic 문자열이 통째로 등장하면, 제외어 유무와
        # 무관하게 무조건 통과시킨다. (apply_direct_mention_guard=False면
        # 이 가드 자체를 적용하지 않는다.)
        if apply_direct_mention_guard and _mentions_topic_directly(article, topic):
            kept.append(article)
        else:
            remaining.append(article)

    if not normalized_exclusions:
        # 동음이의어 제외어가 없으면 나머지는 그대로 통과 (걸러낼 기준이 없음).
        kept.extend(remaining)
        return kept

    excl_low = exclusion_low_threshold if exclusion_low_threshold is not None else getattr(
        settings, "exclusion_low_threshold", _DEFAULT_EXCLUSION_LOW
    )
    excl_high = exclusion_high_threshold if exclusion_high_threshold is not None else getattr(
        settings, "exclusion_high_threshold", _DEFAULT_EXCLUSION_HIGH
    )

    neg_vec = np.array(
        await get_embeddings().aembed_query(
            _build_context_only_query(normalized_exclusions, topic)
        )
    )

    borderline: list[Article] = []
    borderline_indices: list[int] = []

    for i, article in enumerate(remaining):
        art_vec = np.array(article.embedding)
        sim_neg = _cosine_sim(neg_vec, art_vec)

        if sim_neg >= excl_high:
            continue  # 명확히 혼동 개체 -> 즉시 제외
        elif sim_neg < excl_low:
            kept.append(article)  # 명확히 무관한 혼동 -> 통과
        else:
            borderline.append(article)
            borderline_indices.append(i)

    if borderline:
        judgments = await _llm_judge_borderline(
            topic, normalized_exclusions, borderline, borderline_indices
        )
        for article, idx in zip(borderline, borderline_indices):
            if judgments.get(idx, True):
                kept.append(article)

    return kept