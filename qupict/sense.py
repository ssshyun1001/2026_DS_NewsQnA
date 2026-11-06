"""
[0단계] 검색어 의미 선택

검색어가 같아도 사용자가 원하는 대상은 다를 수 있다. ("애플" → 기업 Apple / 과일 사과)
이 모듈은 검색어 하나를 받아 LLM이 의미 후보를 만들고, 사용자가 그중 하나를 고르게 한다.
고른 의미(SelectedSense)가 이후 수집·필터링·검색의 기준이 된다.

흐름:
  generate_sense_candidates("애플")
      → SenseCandidates: 자주 쓰이는 순 최대 3개 (정의 + 별칭 + 연관어)
  후보가 2개 이상  → 사용자가 하나 선택       → select_sense(...)
  후보가 1개       → 자동 선택               → select_sense(..., 0)
  후보가 0개 / "기타" → 사용자가 직접 설명 입력 → build_custom_sense(...) → select_custom_sense(...)
  targets.yaml 일괄 처리 → 미리 적어 둔 의미 이름으로 찾기 → select_sense_by_name(...)

팀에서 정한 규칙:
  - 후보는 자주 쓰이는 순 TOP3, 나머지는 "기타(직접 입력)"로 처리
  - 브랜드·기업은 그 제품·서비스까지 하나의 의미로 묶는다 (애플 = 기업 Apple, 아이폰 포함)
  - LLM이 모르는 신조어·고유명사는 직접 입력으로 처리
  - 의미는 하나만 고른다 (단수 선택)

일관성: temperature=0으로 같은 검색어에는 최대한 같은 후보가 나오게 한다.
같은 검색어의 후보를 DB(query_senses)에 저장해 재사용하는 캐싱은 DB 쪽에서 처리한다.

직접 실행해 보기:
    python -m qupict.sense 애플
"""
from __future__ import annotations

import json
import re
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from qupict.config import settings
from qupict.llm import get_chat_model
from qupict.schemas import SearchQuery, SelectedSense, Sense, SenseCandidates, unique_terms

MAX_CANDIDATES = 3
MAX_RELATED_TERMS = 8


# ---------------------------------------------------------------------------
# 의미 후보 생성
# ---------------------------------------------------------------------------

class _SenseList(BaseModel):
    """LLM 구조화 출력 형식."""
    senses: list[Sense] = Field(default_factory=list)


_CANDIDATES_SYSTEM_PROMPT = """\
당신은 뉴스 검색어가 가리킬 수 있는 의미(대상)를 정리하는 도우미입니다.
검색어 하나를 보고, 한국 뉴스에서 이 검색어로 찾을 만한 서로 다른 의미를 뽑으세요.

규칙:
1. 한국 뉴스에서 자주 쓰이는 순서대로 최대 3개만 만드세요. 의미가 사실상 하나면 1개만 만드세요.
2. 같은 대상의 다른 표기(정식 명칭, 약어, 영문명)는 별개의 의미가 아닙니다. aliases에 넣으세요.
   예: "서울여대" → 의미 "서울여자대학교", aliases ["서울여자대학교", "SWU"]
3. 브랜드·기업은 그 회사의 제품·서비스까지 하나의 의미로 묶으세요. 제품을 따로 나누지 마세요.
   예: "애플" → "기업 Apple"(아이폰, 맥북 포함) 하나. "아이폰"을 별도 의미로 만들지 마세요.
4. name은 다른 후보와 바로 구분되는 짧은 이름으로 쓰세요. (예: "기업 Apple", "과일 사과", "방송인 강남", "지역 강남")
5. definition은 한 줄로 쓰세요.
6. aliases에는 확실히 알고 있는 다른 표기만 넣으세요. 검색어 원문은 넣지 마세요. 모르면 빈 리스트.
7. related_terms에는 그 의미를 다룬 기사에 자주 같이 나오는 단어를 5~8개 넣으세요.
   - 다른 후보와 겹치지 않는, 의미를 구분해 주는 단어를 고르세요.
   - 검색어 원문, "뉴스", "최근", "이슈" 같은 일반 단어는 넣지 마세요.
   예: 기업 Apple → ["아이폰", "맥북", "팀 쿡", "iOS", "애플워치"]
       과일 사과 → ["과수원", "농가", "수확", "사과값", "품종"]
8. 처음 보는 단어나 확신이 없는 신조어·고유명사는 지어내지 말고 senses를 빈 리스트로 두세요.
"""

_candidates_llm = None


def _get_candidates_llm():
    global _candidates_llm
    if _candidates_llm is None:
        llm = get_chat_model(temperature=0)
        _candidates_llm = llm.with_structured_output(_SenseList)
    return _candidates_llm


def _clean_sense(sense: Sense, query: str) -> Sense:
    """LLM 결과 정리: 공백·중복 제거, 검색어 원문이 별칭/연관어에 섞여 들어온 경우 제거."""
    q = query.strip()
    return Sense(
        name=sense.name.strip(),
        definition=sense.definition.strip(),
        aliases=[a for a in unique_terms(sense.aliases) if a != q],
        related_terms=[t for t in unique_terms(sense.related_terms) if t != q][:MAX_RELATED_TERMS],
        is_custom=sense.is_custom,
    )


def _drop_shared_terms(senses: list[Sense]) -> list[Sense]:
    """두 개 이상의 후보에 동시에 나온 연관어는 의미를 구분해 주지 못하므로 모두에서 뺀다.

    예: 기업 Apple과 과일 사과에 둘 다 "가격"이 있으면, "가격"이 나온 기사를
    어느 쪽으로도 판단할 수 없다. 배제어로 쓰면 오히려 관련 기사를 잘못 떨어뜨린다.
    """
    counts: dict[str, int] = {}
    for s in senses:
        for t in set(s.related_terms):
            counts[t] = counts.get(t, 0) + 1
    shared = {t for t, c in counts.items() if c > 1}
    if not shared:
        return senses
    return [s.model_copy(update={"related_terms": [t for t in s.related_terms if t not in shared]})
            for s in senses]


def generate_sense_candidates(query: str) -> SenseCandidates:
    """검색어 → 의미 후보 (자주 쓰이는 순, 최대 3개).

    LLM 호출이 실패하거나 후보를 못 만들면 빈 목록을 돌려준다.
    빈 목록이면 호출하는 쪽에서 "직접 입력"으로 넘어가면 된다.
    """
    query = query.strip()
    try:
        result: _SenseList = _get_candidates_llm().invoke(
            [("system", _CANDIDATES_SYSTEM_PROMPT), ("human", f"검색어: {query}")]
        )
        raw = result.senses
    except Exception:
        raw = []

    senses: list[Sense] = []
    seen_names: set[str] = set()
    for s in raw:
        s = _clean_sense(s, query)
        if not s.name or s.name in seen_names:  # 이름이 빈 후보, 같은 이름의 후보는 버린다
            continue
        seen_names.add(s.name)
        senses.append(s)
    senses = _drop_shared_terms(senses[:MAX_CANDIDATES])

    return SenseCandidates(query=query, senses=senses, model=settings.chat_model)


# ---------------------------------------------------------------------------
# "기타" 직접 입력 → 의미 정의
# ---------------------------------------------------------------------------

_CUSTOM_SYSTEM_PROMPT = """\
사용자가 뉴스 검색어와, 그 검색어로 찾고 싶은 대상을 직접 설명했습니다.
사용자의 설명을 그대로 존중해서 의미 하나를 정리하세요.

- name: 사용자의 설명을 줄인 짧은 이름
- definition: 사용자의 설명을 한 줄로 정리 (사용자가 말하지 않은 사실을 지어내지 마세요)
- aliases: 사용자가 언급했거나 확실히 아는 다른 표기만. 검색어 원문은 넣지 마세요.
- related_terms: 이 대상을 다룬 기사에 나올 만한 단어 3~8개. 설명에서 근거를 찾을 수 없으면 적게 넣으세요.
"""

_custom_llm = None


def _get_custom_llm():
    global _custom_llm
    if _custom_llm is None:
        llm = get_chat_model(temperature=0)
        _custom_llm = llm.with_structured_output(Sense)
    return _custom_llm


def build_custom_sense(query: str, description: str) -> Sense:
    """사용자가 직접 입력한 설명 → Sense. LLM이 실패해도 사용자 설명을 그대로 의미로 쓴다."""
    query, description = query.strip(), description.strip()
    try:
        sense: Sense = _get_custom_llm().invoke(
            [("system", _CUSTOM_SYSTEM_PROMPT),
             ("human", f"검색어: {query}\n찾고 싶은 대상: {description}")]
        )
        sense = _clean_sense(sense, query)
    except Exception:
        sense = Sense(name=description[:30], definition=description)

    # 직접 입력은 사용자의 말이 기준이므로 정의가 비면 원문을 넣는다
    if not sense.definition:
        sense = sense.model_copy(update={"definition": description})
    return sense.model_copy(update={"is_custom": True})


# ---------------------------------------------------------------------------
# 선택
# ---------------------------------------------------------------------------

def _make_selection(query: str, chosen: Sense, others: list[Sense]) -> SelectedSense:
    """선택 결과를 만든다. 배제어 = 고르지 않은 후보들의 연관어 (고른 의미의 단어와 겹치면 뺌)."""
    own = set(chosen.related_terms) | set(chosen.aliases) | {query}
    excluded = unique_terms([t for s in others for t in s.related_terms if t not in own])
    return SelectedSense(query=query, sense=chosen, excluded_terms=excluded, other_senses=others)


def select_sense(candidates: SenseCandidates, index: int) -> SelectedSense:
    """후보 목록에서 index번째(0부터) 의미를 고른다."""
    if not 0 <= index < len(candidates.senses):
        raise IndexError(f"후보는 {len(candidates.senses)}개입니다. (받은 번호: {index})")
    chosen = candidates.senses[index]
    others = [s for i, s in enumerate(candidates.senses) if i != index]
    return _make_selection(candidates.query, chosen, others)


def select_custom_sense(candidates: SenseCandidates, custom: Sense) -> SelectedSense:
    """직접 입력한 의미를 고른다. LLM 후보는 전부 '고르지 않은 의미'가 되어 배제어로 쓰인다."""
    return _make_selection(candidates.query, custom, list(candidates.senses))


def _normalize(text: str) -> str:
    return re.sub(r"[\s()\[\]·,./-]", "", text).lower()


def select_sense_by_name(candidates: SenseCandidates, name: str) -> SelectedSense | None:
    """targets.yaml에 적어 둔 의미 이름으로 후보를 찾는다. 못 찾으면 None.

    LLM이 붙인 이름이 매번 글자까지 똑같지는 않으므로 (예: "기업 Apple" / "기업Apple"),
    공백·괄호를 무시하고 비교한다. 정확히 같은 이름 → 별칭 → 이름 일부 포함 순서로 찾는다.
    None이면 호출하는 쪽에서 build_custom_sense()로 직접 입력 처리하면 된다.
    """
    target = _normalize(name)
    if not target:
        return None
    names = [_normalize(s.name) for s in candidates.senses]

    for i, n in enumerate(names):                      # 1) 이름이 정확히 같음
        if n == target:
            return select_sense(candidates, i)
    for i, s in enumerate(candidates.senses):          # 2) 별칭이 같음
        if any(_normalize(a) == target for a in s.aliases):
            return select_sense(candidates, i)
    partial = [i for i, n in enumerate(names) if target in n or n in target]
    if len(partial) == 1:                              # 3) 이름 일부 포함 (하나만 걸릴 때만)
        return select_sense(candidates, partial[0])
    return None


# ---------------------------------------------------------------------------
# 기존 검색 키워드 생성 (run.py가 아직 사용 중 — collect.py 개편 후 삭제 예정)
# ---------------------------------------------------------------------------

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
        llm = get_chat_model(temperature=0)
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


# ---------------------------------------------------------------------------
# 터미널에서 직접 확인하기: python -m qupict.sense 애플
# ---------------------------------------------------------------------------

def _choose_interactively(candidates: SenseCandidates) -> SelectedSense:
    if not candidates.senses:
        print("의미 후보를 만들지 못했습니다. 찾고 싶은 대상을 직접 설명해 주세요.")
        desc = input("설명> ").strip()
        return select_custom_sense(candidates, build_custom_sense(candidates.query, desc))

    if not candidates.needs_selection:
        print(f"의미가 하나뿐이라 자동으로 선택합니다: {candidates.senses[0].name}")
        return select_sense(candidates, 0)

    for i, s in enumerate(candidates.senses, 1):
        print(f"  {i}. {s.name} — {s.definition}")
    print("  0. 기타 (직접 입력)")
    while True:
        choice = input("번호> ").strip()
        if choice == "0":
            desc = input("찾고 싶은 대상을 설명해 주세요> ").strip()
            return select_custom_sense(candidates, build_custom_sense(candidates.query, desc))
        if choice.isdigit() and 1 <= int(choice) <= len(candidates.senses):
            return select_sense(candidates, int(choice) - 1)
        print(f"0~{len(candidates.senses)} 사이 번호를 입력하세요.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    q = sys.argv[1] if len(sys.argv) > 1 else input("검색어> ").strip()
    cands = generate_sense_candidates(q)
    selected = _choose_interactively(cands)
    print(json.dumps(selected.model_dump(), ensure_ascii=False, indent=2))
