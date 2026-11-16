"""
[0단계] 검색어 의미 선택

검색어가 같아도 사용자가 원하는 대상은 다를 수 있다. ("애플" → 기업 Apple / 과일 사과)
이 모듈은 검색어 하나를 받아 LLM이 의미 후보를 만들고, 사용자가 그중 하나를 고르게 한다.
고른 의미(SelectedSense)가 이후 수집·필터링·검색의 기준이 된다.

흐름:
  load_sense_candidates("애플")           ← DB에 있으면 DB 것, 없으면 LLM으로 만들어 DB에 저장
      → SenseCandidates: LLM 후보(자주 쓰이는 순 최대 3개) + 예전에 "기타"로 직접 입력해 저장된 의미
         (많이 고른 순으로 정렬. 직접 입력 의미는 최대 MAX_SAVED_CUSTOM개)
      (generate_sense_candidates = DB 없이 LLM만 부르는 함수)
  후보가 2개 이상  → 사용자가 하나 선택       → select_sense(...)
  후보가 1개       → 자동 선택               → select_sense(..., 0)
  후보가 0개 / "기타" → 사용자가 직접 설명 입력 → select_described_sense(...) (이름은 설명 그대로, 기준은 설명)
  고른 뒤          → remember_selection(...)  → 고른 횟수 +1, 새 직접 입력 의미는 DB에 저장 → 다음부터 후보로 나온다
  DB에 저장된 topic 다시 처리 → 의미 이름으로 후보 찾기 → select_sense_by_name(...)

후보가 쌓이는 방식 (저장 위치 = DB query_senses):
  - 처음 보는 검색어: LLM이 후보를 만들어 저장 (이후 LLM 안 부름)
  - "기타"로 새 설명이 들어오면: LLM이 정리한 의미를 저장 → 다음 사용자에게는 후보로 바로 보인다
  - 같은 설명을 다시 입력하거나 저장된 직접 입력 후보를 고르면 LLM을 부르지 않는다
  - 많이 고른 의미일수록 위에 보인다 (use_count)

의미·별칭·연관어는 LLM이 만든 후보만 쓴다 (사람이 적은 값으로 덮어쓰지 않는다).

팀에서 정한 규칙:
  - 후보는 자주 쓰이는 순 TOP3, 나머지는 "기타(직접 입력)"로 처리
  - 브랜드·기업은 그 제품·서비스까지 하나의 의미로 묶는다 (애플 = 기업 Apple, 아이폰 포함)
  - LLM이 모르는 신조어·고유명사는 직접 입력으로 처리
  - 의미는 하나만 고른다 (단수 선택)

일관성: temperature=0이어도 LLM 결과는 실행마다 조금씩 달라진다.
그래서 처음 만든 후보를 DB(query_senses)에 저장하고 이후에는 그것만 쓴다 (load_sense_candidates).

직접 실행해 보기:
    python -m qupict.sense 애플          # 검색어 하나, 터미널에서 직접 선택
"""
from __future__ import annotations

import json
import re
import sys

from pydantic import BaseModel, Field

from qupict.config import settings
from qupict.llm import get_structured_model
from qupict.schemas import SelectedSense, Sense, SenseCandidates, unique_terms

MAX_CANDIDATES = 3
MAX_RELATED_TERMS = 8
MAX_SAVED_CUSTOM = 3   # 후보로 보여 줄 직접 입력 의미 수 (많이 고른 순)


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
5. definition은 한 줄로 쓰세요. 다른 후보와 구분되는 핵심 정보(직업·분야, 대표 활동·소속)를 넣되,
   확실한 사실만 쓰세요. 사람이면 본명·나이 같은 틀리기 쉬운 개인 정보는 쓰지 마세요.
6. aliases의 각 표기는 그대로 뉴스 검색어로 쓰이고, 기사에 나오면 "이 대상을 언급한 기사"로 판단됩니다.
   틀리거나 뜻이 여러 개인 표기가 하나라도 있으면 엉뚱한 기사가 모이므로,
   한국 뉴스 기사에서 실제로 이 대상을 가리킬 때 쓰는 표기 중 확실히 아는 것만 넣으세요.
   - 조금이라도 확신이 없으면 넣지 마세요. 모르면 빈 리스트로 두세요. 빈 리스트여도 괜찮습니다.
   - 사람(연예인·운동선수·정치인 등)은 본명, 외국어 원래 이름, 예전 이름을 넣지 마세요.
     맞는 본명이어도 넣지 마세요.
     예: "아이유" → 의미 "가수 아이유", aliases ["IU"] (본명 "이지은"은 넣지 않음)
   - 그 자체로 뜻이 여러 개인 일반 단어("사과", "snow")나 1~2글자 약자("K", "Xe")는 넣지 마세요.
   - 제품·서비스·설명 문구("아이폰", "삼성전자 주식")는 별칭이 아니라 related_terms에 넣으세요.
   - 검색어 원문은 넣지 마세요.
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
        _candidates_llm = get_structured_model(_SenseList)
    return _candidates_llm


_HANJA_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def _has_hanja(text: str) -> bool:
    """한자가 섞인 단어인지. (예: "出荷", "積雪" — 한국 기사 본문과 거의 일치하지 않는다)"""
    return bool(_HANJA_RE.search(text))


def _clean_sense(sense: Sense, query: str) -> Sense:
    """LLM 결과 정리.

    - 공백·중복 제거
    - 검색어 원문은 별칭·연관어에서 뺀다 (배제어가 되면 모든 기사가 걸러짐)
    - 한자가 섞인 단어는 뺀다 (예: "出荷")
    - 검색어 안에 들어 있는 단어는 연관어에서 뺀다
      (예: 서울여대의 "여대" → 성신여대 기사까지 연관어 점수를 받게 됨)
    """
    q = query.strip()
    aliases = [a for a in unique_terms(sense.aliases) if a != q and not _has_hanja(a)]
    related = [
        t for t in unique_terms(sense.related_terms)
        if t != q and t not in q and not _has_hanja(t)
    ]
    return Sense(
        name=sense.name.strip(),
        definition=sense.definition.strip(),
        aliases=aliases,
        related_terms=related[:MAX_RELATED_TERMS],
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
        if not s.related_terms:
            # 연관어가 하나도 없다 = LLM이 이 의미를 잘 모른다 (예: "베라" → "인물 베라(확실하지 않음)")
            # 필터링에 쓸 정보가 없으므로 버리고, 후보가 0개가 되면 "직접 입력"으로 넘어가게 한다.
            continue
        seen_names.add(s.name)
        senses.append(s)
    senses = _drop_shared_terms(senses[:MAX_CANDIDATES])

    return SenseCandidates(query=query, senses=senses, model=settings.chat_model)


# ---------------------------------------------------------------------------
# DB 캐싱: 같은 검색어면 LLM을 다시 부르지 않고 저장된 후보를 쓴다
# ---------------------------------------------------------------------------

def load_sense_candidates(query: str) -> tuple[SenseCandidates, str]:
    """의미 후보를 가져온다. DB에 있으면 DB 것을, 없으면 LLM으로 만들어 DB에 저장한다.

    왜 필요한가: LLM은 temperature=0이어도 같은 검색어에 연관어를 조금씩 다르게 낸다.
    그러면 배제어(2단계 규칙 필터 기준)가 실행할 때마다 바뀌어 실험을 재현할 수 없다.
    처음 만든 후보를 DB(query_senses)에 저장해 두고 계속 쓰면 기준이 고정되고 LLM 비용도 0이 된다.
    예전에 "기타"로 직접 입력해 저장된 의미도 함께 돌려준다 (많이 고른 순, 최대 MAX_SAVED_CUSTOM개).

    돌려주는 값: (후보 목록, 출처) — 출처는 "DB", "LLM → DB 저장", "LLM (DB 없음)" 등

      - FIRESTORE_PROJECT가 비어 있으면 DB 없이 LLM만 쓴다 (DB를 안 띄운 팀원도 실행 가능)
      - DB 연결에 실패해도 멈추지 않고 LLM으로 계속한다 (경고만 출력)
      - LLM이 후보를 하나도 못 만들면 저장하지 않는다 (다음 실행 때 다시 시도하도록)
    """
    query = query.strip()
    if not settings.has_db:
        return generate_sense_candidates(query), "LLM (DB 없음)"

    from qupict import db  # DB 패키지(google-cloud-firestore)가 없는 환경에서도 이 파일을 import할 수 있게 여기서 불러온다

    try:
        cached = db.get_sense_candidates(query)
    except Exception as e:
        print(f"  [경고] DB에서 의미 후보를 읽지 못했습니다 ({type(e).__name__}). LLM으로 만듭니다.")
        return generate_sense_candidates(query), "LLM (DB 오류)"
    saved_custom = _trim_custom(cached) if cached is not None else None
    if saved_custom is not None and any(not s.is_custom for s in saved_custom.senses):
        n = sum(s.is_custom for s in saved_custom.senses)
        return saved_custom, "DB" + (f" (직접 입력 {n}개 포함)" if n else "")

    # LLM 후보가 아직 없다 (처음 보는 검색어, 또는 예전에 LLM이 후보를 못 만들어 직접 입력만 저장된 검색어)
    candidates = generate_sense_candidates(query)
    if not candidates.senses:
        source = "LLM (후보 없음, 저장 안 함)"
    else:
        try:
            db.save_sense_candidates(candidates)
            source = "LLM → DB 저장"
        except Exception as e:
            print(f"  [경고] 의미 후보를 DB에 저장하지 못했습니다 ({type(e).__name__}).")
            source = "LLM (DB 저장 실패)"
    if saved_custom is not None:   # 저장된 직접 입력 의미는 LLM 후보 뒤에 붙인다
        names = {s.name for s in candidates.senses}
        extra = [s for s in saved_custom.senses if s.name not in names]
        candidates = candidates.model_copy(update={
            "senses": [*candidates.senses, *extra],
            "custom_includes": saved_custom.custom_includes, "use_counts": saved_custom.use_counts})
        source += f" + DB 직접 입력 {len(extra)}개"
    return candidates, source


def _trim_custom(candidates: SenseCandidates) -> SenseCandidates:
    """직접 입력 의미는 많이 고른 순으로 MAX_SAVED_CUSTOM개만 남긴다 (DB가 이미 그 순서로 준다)."""
    kept, n = [], 0
    for s in candidates.senses:
        if s.is_custom:
            n += 1
            if n > MAX_SAVED_CUSTOM:
                continue
        kept.append(s)
    return candidates.model_copy(update={"senses": kept})


def remember_selection(candidates: SenseCandidates, selected: SelectedSense) -> None:
    """고른 의미를 DB에 기록한다: 고른 횟수 +1, 처음 보는 직접 입력 의미는 저장 (→ 다음부터 후보로 나온다).
    DB가 없거나 실패해도 멈추지 않는다 (경고만 출력)."""
    if not settings.has_db:
        return
    from qupict import db

    includes: list[str] = []
    if selected.sense.is_custom:   # '다른 의미'로 두지 않은 LLM 후보 = 포함 후보 (다음에 고를 때 그대로 재현)
        others = {o.name for o in selected.other_senses}
        includes = [s.name for s in candidates.senses if not s.is_custom and s.name not in others]
    try:
        db.record_sense_use(selected.query, selected.sense, includes)
    except Exception as e:
        print(f"  [경고] 고른 의미를 DB에 기록하지 못했습니다 ({type(e).__name__}).")


# ---------------------------------------------------------------------------
# "기타" 직접 입력 → 의미 정의
# ---------------------------------------------------------------------------

_CUSTOM_SYSTEM_PROMPT = """\
사용자가 뉴스 검색어와, 그 검색어로 찾고 싶은 대상을 직접 설명했습니다.
사용자의 설명을 그대로 존중해서 의미 하나를 정리하세요. 기준은 검색어가 아니라 사용자의 설명입니다.
(예: 검색어 "서울여대" + 설명 "서울 소재 여자대학교" → 찾는 대상은 서울여자대학교 한 곳이 아니라 서울에 있는 여자대학교 전체)

- definition: 사용자의 설명을 한 줄로 정리 (사용자가 말하지 않은 사실을 지어내지 마세요)
- aliases: 이 대상을 가리키는 다른 표기. 검색어 원문은 넣지 마세요.
  대상이 여러 개를 묶은 것이면(예: "서울 소재 여자대학교", "국내 이동통신 3사") 그 각각의 이름을
  기사에 흔히 쓰는 표기로 넣으세요 (예: 이화여대, 숙명여대, 성신여대 …). 확실히 아는 것만, 최대 8개.
- related_terms: 이 대상을 다룬 기사에 나올 만한 단어 3~8개. aliases에 넣은 이름은 빼세요.
  설명에서 근거를 찾을 수 없으면 적게 넣으세요.
- includes: 아래 [검색어의 다른 뜻 후보] 중 사용자가 찾는 대상과 같거나 그 안에 포함되는 후보의 이름 (그대로 복사).
  예: 설명이 "서울 소재 여자대학교"면 후보 "서울여자대학교"는 포함됩니다. 해당 없으면 빈 목록.
"""

# 직접 입력한 의미의 이름 = 사용자가 쓴 설명 그대로 (이 길이까지).
# 예전에는 LLM이 줄인 이름을 써서, "서울 소재 여자대학교"가 "서울여자대학교"로 바뀌어 LLM 후보와 이름이 같아졌다
# → 같은 topic(검색어, 의미 이름)으로 합쳐져 예전 서울여자대학교 수집·판정을 그대로 쓰고 topic 기준까지 덮어썼다
# (확인 2026-11-15, topic 9). 이름이 같아지는 건 사용자가 후보 이름을 그대로 입력했을 때뿐이다.
CUSTOM_NAME_MAX_CHARS = 30


class _CustomSense(BaseModel):
    """직접 입력 → LLM이 정리한 의미 (이름은 사용자 설명을 그대로 쓰므로 묻지 않는다)."""
    definition: str = Field(description="사용자의 설명을 한 줄로 정리")
    aliases: list[str] = Field(default_factory=list, description="대상을 가리키는 표기. 여러 개를 묶은 대상이면 각각의 이름")
    related_terms: list[str] = Field(default_factory=list, description="기사에 나올 만한 단어 3~8개")
    includes: list[str] = Field(default_factory=list,
                                description="[검색어의 다른 뜻 후보] 중 찾는 대상과 같거나 그 안에 포함되는 후보 이름")


_custom_llm = None


def _get_custom_llm():
    global _custom_llm
    if _custom_llm is None:
        _custom_llm = get_structured_model(_CustomSense)
    return _custom_llm


def _custom_name(description: str) -> str:
    name = " ".join(description.split())
    return name if len(name) <= CUSTOM_NAME_MAX_CHARS else name[:CUSTOM_NAME_MAX_CHARS - 1].rstrip() + "…"


def describe_custom_sense(query: str, description: str,
                          candidates: list[Sense] = ()) -> tuple[Sense, list[str]]:
    """사용자가 직접 입력한 설명 → (Sense, 그 대상에 포함되는 LLM 후보 이름들).
    candidates: 이 검색어의 LLM 의미 후보. 찾는 대상에 포함되는 후보는 '다른 의미'로 두지 않는다 (select_described_sense).
    LLM이 실패해도 사용자 설명을 그대로 의미로 쓴다."""
    query, description = query.strip(), description.strip()
    listing = "\n".join(f"- {s.name}: {s.definition}" for s in candidates) or "- (없음)"
    includes: list[str] = []
    try:
        out: _CustomSense = _get_custom_llm().invoke(
            [("system", _CUSTOM_SYSTEM_PROMPT),
             ("human", f"검색어: {query}\n찾고 싶은 대상: {description}\n\n[검색어의 다른 뜻 후보]\n{listing}")]
        )
        sense = _clean_sense(Sense(name=_custom_name(description), definition=out.definition,
                                   aliases=out.aliases[:MAX_RELATED_TERMS], related_terms=out.related_terms), query)
        known = {s.name for s in candidates}
        includes = [n.strip() for n in out.includes if n.strip() in known]
    except Exception:
        sense = Sense(name=_custom_name(description), definition=description)

    # 직접 입력은 사용자의 말이 기준이므로 정의가 비면 원문을 넣는다
    if not sense.definition:
        sense = sense.model_copy(update={"definition": description})
    # 포함되는 후보의 이름·별칭도 이 대상의 표기로 쓴다 (예: "서울여자대학교" 전체 이름으로 쓴 기사도 찾게)
    included = [s for s in candidates if s.name in includes]
    aliases = unique_terms([*sense.aliases, *(t for s in included for t in [s.name, *s.aliases])])
    aliases = [a for a in aliases if a != query and _custom_name(a) != sense.name]
    return sense.model_copy(update={"is_custom": True, "aliases": aliases}), includes


def build_custom_sense(query: str, description: str) -> Sense:
    """사용자가 직접 입력한 설명 → Sense (LLM 후보와 견주지 않는 예전 방식. 새 코드는 select_described_sense)."""
    return describe_custom_sense(query, description)[0]


# ---------------------------------------------------------------------------
# 선택
# ---------------------------------------------------------------------------

SAME_SENSE_MIN_SHARED_TERMS = 2


def _make_selection(query: str, chosen: Sense, others: list[Sense]) -> SelectedSense:
    """선택 결과를 만든다. 배제어 = 고르지 않은 후보들의 연관어 (고른 의미의 단어와 겹치면 뺌).

    직접 입력한 의미가 사실은 LLM 후보 중 하나와 같은 의미일 수 있다.
    (예: 직접 입력 "보건용 마스크" ↔ LLM 후보 "방역 마스크") 그 후보를 '다른 의미'로 두면
    방역·황사 같은 단어가 배제어가 되어 원하는 기사를 오히려 걸러 버린다.
    그래서 고른 의미와 연관어가 2개 이상 겹치는 후보는 같은 의미로 보고 제외한다.
    (LLM 후보끼리는 공통 연관어를 미리 지우므로 이 규칙에 걸리지 않는다)
    """
    chosen_terms = set(chosen.related_terms)
    others = [s for s in others
              if len(chosen_terms & set(s.related_terms)) < SAME_SENSE_MIN_SHARED_TERMS]
    own = chosen_terms | set(chosen.aliases) | {query}
    excluded = unique_terms([t for s in others for t in s.related_terms if t not in own])
    return SelectedSense(query=query, sense=chosen, excluded_terms=excluded, other_senses=others)


def select_sense(candidates: SenseCandidates, index: int) -> SelectedSense:
    """후보 목록에서 index번째(0부터) 의미를 고른다."""
    if not 0 <= index < len(candidates.senses):
        raise IndexError(f"후보는 {len(candidates.senses)}개입니다. (받은 번호: {index})")
    chosen = candidates.senses[index]
    # '다른 의미'(배제어 · LLM 대조 기준)는 LLM 후보만. 저장된 직접 입력 의미는 다른 의미로 쓰지 않는다
    # (직접 입력은 LLM 후보와 겹치거나 더 넓은 대상일 때가 많아, 배제어로 쓰면 원하는 기사를 걸러 버린다)
    includes = set(candidates.custom_includes.get(chosen.name, [])) if chosen.is_custom else set()
    others = [s for i, s in enumerate(candidates.senses)
              if i != index and not s.is_custom and s.name not in includes]
    return _make_selection(candidates.query, chosen, others)


def select_custom_sense(candidates: SenseCandidates, custom: Sense,
                        includes: list[str] = ()) -> SelectedSense:
    """직접 입력한 의미를 고른다. LLM 후보는 '고르지 않은 의미'가 되어 배제어 · LLM 검증의 대조 기준으로 쓰인다.
    includes: 찾는 대상에 포함되는 후보 이름 → 다른 의미로 두지 않는다
    (예: "서울 소재 여자대학교"를 찾는데 후보 "서울여자대학교"를 다른 의미로 두면 그 기사가 무관이 된다)."""
    others = [s for s in candidates.senses if not s.is_custom and s.name not in set(includes)]
    return _make_selection(candidates.query, custom, others)


def select_described_sense(candidates: SenseCandidates, description: str) -> SelectedSense:
    """사용자가 직접 설명한 대상으로 의미를 고른다 ("기타 (직접 입력)"). 웹 · 터미널이 이 함수를 쓴다.

    설명이 LLM 후보의 이름과 같으면(예: "서울여자대학교") 그 후보를 고른 것으로 본다.
    직접 입력으로 만들면 같은 topic(검색어, 의미 이름)에 직접 입력 값이 덮어써져
    그 후보 topic의 정의 · 연관어가 바뀌기 때문이다 (topic은 수집 · 판정 · 라벨이 쌓인 공용 데이터).
    예전에 같은 설명으로 저장된 직접 입력 의미가 있으면 그것을 고른다 (LLM을 다시 부르지 않는다)."""
    wanted = _normalize(description)
    as_custom = _normalize(_custom_name(description))
    for i, s in enumerate(candidates.senses):
        if wanted and _normalize(s.name) in ((as_custom,) if s.is_custom else (wanted,)):
            return select_sense(candidates, i)
    llm_senses = [s for s in candidates.senses if not s.is_custom]
    custom, includes = describe_custom_sense(candidates.query, description, llm_senses)
    return select_custom_sense(candidates, custom, includes)


def _normalize(text: str) -> str:
    return re.sub(r"[\s()\[\]·,./-]", "", text).lower()


def select_sense_by_name(candidates: SenseCandidates, name: str) -> SelectedSense | None:
    """의미 이름(예: DB topic에 저장된 이름)으로 후보를 찾는다. 못 찾으면 None.

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
# 터미널에서 직접 확인하기: python -m qupict.sense 애플
# ---------------------------------------------------------------------------

def choose_sense_interactively(candidates: SenseCandidates) -> SelectedSense:
    """터미널에서 후보를 보여 주고 번호로 고르게 한다. (run.py도 이 함수를 쓴다)
    고른 의미는 DB에 기록한다 (고른 횟수, 새 직접 입력 의미 저장)."""
    selected = _choose(candidates)
    remember_selection(candidates, selected)
    return selected


def sense_label(candidates: SenseCandidates, s: Sense) -> str:
    """후보 옆에 붙이는 표시: 직접 입력으로 저장된 의미인지, 몇 번 골랐는지."""
    marks = ["직접 입력"] if s.is_custom else []
    if candidates.use_counts.get(s.name):
        marks.append(f"{candidates.use_counts[s.name]}회 선택")
    return f" [{' · '.join(marks)}]" if marks else ""


def _choose(candidates: SenseCandidates) -> SelectedSense:
    if not candidates.senses:
        print("의미 후보를 만들지 못했습니다. 찾고 싶은 대상을 직접 설명해 주세요.")
        desc = input("설명> ").strip()
        return select_described_sense(candidates, desc)

    if not candidates.needs_selection:
        print(f"의미가 하나뿐이라 자동으로 선택합니다: {candidates.senses[0].name}")
        return select_sense(candidates, 0)

    for i, s in enumerate(candidates.senses, 1):
        print(f"  {i}. {s.name} — {s.definition}{sense_label(candidates, s)}")
    print("  0. 기타 (직접 입력)")
    while True:
        choice = input("번호> ").strip()
        if choice == "0":
            desc = input("찾고 싶은 대상을 설명해 주세요> ").strip()
            return select_described_sense(candidates, desc)
        if choice.isdigit() and 1 <= int(choice) <= len(candidates.senses):
            return select_sense(candidates, int(choice) - 1)
        print(f"0~{len(candidates.senses)} 사이 번호를 입력하세요.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    q = sys.argv[1] if len(sys.argv) > 1 else input("검색어> ").strip()
    cands, source = load_sense_candidates(q)
    print(f"의미 후보 출처: {source}")
    selected = choose_sense_interactively(cands)
    print(json.dumps(selected.model_dump(), ensure_ascii=False, indent=2))
