"""
eval_retrieval.py — HyDE / Multi-Query Retriever 개선 효과 측정 스크립트

목적:
  검색 텍스트 확장 방식 4가지를 비교한다.
    Base         : 질문 원문만 그대로 검색 (use_hyde=False, use_multi_query=False)
    HyDE only    : HyDE 가상 문장만 추가 (use_hyde=True,  use_multi_query=False)
    MultiQ only  : 표현이 다른 변형 질문만 추가 (use_hyde=False, use_multi_query=True)
    Both         : HyDE + Multi-Query 모두 적용 (use_hyde=True,  use_multi_query=True)

  HyDE와 Multi-Query는 서로 다른 축의 개선(가상 문서 생성 vs 질문 재구성)이라,
  각각의 단독 효과와 함께 사용했을 때의 상호작용(시너지 또는 상쇄)을 같이 확인한다.

지금 구조에 맞춰 바뀐 점:
  - 검증 키워드(validation_terms) 대신 고른 의미 이름(sense)으로 대상을 정한다
  - 검색 대상
      DB 모드(기본): DB에 그 topic이 있으면 실제 서비스 검색(pgvector 청크 검색, search_topic)을 쓴다.
                    수집·필터링을 다시 하지 않는다 (collect-all → filter-all → index-all 뒤에 실행)
      메모리 모드  : DB에 topic이 없거나 --memory를 주면, 예전처럼 그 자리에서 수집 → 필터링 →
                    FAISS 메모리 검색을 한다 (run.py ask --memory와 같은 방식, LLM 판정 비용이 든다)
  - 유사도 기준(RELEVANCE_SIMILARITY_THRESHOLD, 기본 0.5)은 서비스와 같게 적용된다

골든셋:
  4개 주제(제논, 당근, 서울여대, 강남)에 대해
  각각 별도의 골든 질문셋을 둔다. 주제별로 실제 대화에서 검색에 실패했던
  질문들과, 원래도 성공했던 키워드성 질문을 함께 포함한다.

  서울여대 골든셋은 2026-07-27/07-23에 실제로 크롤링된 기사(정의학원 제12대
  이사장 이/취임, 2027수시박람회, 노원구 대학연합회 MOU 관련 기사)를 확인하고
  만든 것이다.

  강남 골든셋은 2026-07-29에 실제로 크롤링된 기사(방송인 강남의 '라디오스타'
  출연 관련 기사)를 확인하고 만든 것이다.

  다만 두 골든셋 모두 수집 기간/시점에 따라 다른 기사가 섞일 수 있으므로,
  실행 결과가 낮게 나오면 그 시점에 크롤링된 기사가 골든셋과 다른 내용일
  가능성을 먼저 의심할 것. (2026년 7월 기사로 만든 질문이라, 최근 30일만 수집한
  DB에는 답이 되는 기사가 없을 수 있다 → 아래 [DEBUG] 줄로 먼저 확인)

판정 기준:
  각 질문마다 "이 답을 포함하는 기사라면 반드시 등장해야 하는 키워드"를 정해두고,
  검색된 결과(DB 모드: 청크 내용 + 기사 제목, 메모리 모드: 기사 제목 + 본문) 중 하나라도
  그 키워드를 포함하면 성공(hit)으로 센다.
  완전한 정답 여부(faithfulness)까지는 보지 않는, 순수 검색 성공률(retrieval hit rate) 지표다.
  (정답 근거 기사 링크로 Recall@5·MRR을 재려면 python run.py exp-b)

사용법 (프로젝트 맨 위 폴더에서):
  python old/eval_retrieval.py 제논
  python old/eval_retrieval.py 당근
  python old/eval_retrieval.py 서울여대
  python old/eval_retrieval.py 강남
  python old/eval_retrieval.py 강남 --memory     (DB 대신 그 자리에서 수집해서)
  (인자를 주지 않으면 GOLDEN_SETS에 등록된 모든 주제를 순서대로 평가하고,
   마지막에 주제별 + 전체 평균 결과를 요약해서 보여준다)
  비용: 질문 1개당 LLM 최대 6번 (조건 4가지의 HyDE·Multi-Query)
"""
from __future__ import annotations

import asyncio
import sys

from _common import get_db_topic, pick_sense

TOP_K = 5   # 검색 결과 수 (DB 모드: 청크 5개 = 실제 답변에 쓰는 개수)


# ============================================================
# 비교할 4가지 검색 조건: (라벨, use_hyde, use_multi_query)
# ============================================================
CONDITIONS: list[tuple[str, bool, bool]] = [
    ("Base", False, False),
    ("HyDE only", True, False),
    ("MultiQ only", False, True),
    ("Both", True, True),
]


# ============================================================
# 주제별 골든셋
# 각 주제는 {topic(검색어), sense(의미 이름), period_days, questions} 를 가진다.
# ============================================================

GOLDEN_SETS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # 제논 — AI 기업 제논의 대표, 행사, 제품·서비스 및 사업 방향 관련
    # 기사 기준으로 구성한 골든셋.
    # 게임 캐릭터·화학 원소 등 동음이의어 관련 기사는 정답으로 인정하지 않는다.
    # ------------------------------------------------------------------
    "제논": {
        "topic": "제논",
        "sense": "AI 기업 제논",          # 예전 검증어: AI기업 (targets.yaml에 없음 → DB에 없으면 후보에서 고름)
        "period_days": 50,
        "questions": [
            {
                "question": "제논의 대표는 누구야?",
                "expected_keywords": ["고석태"],
            },
            {
                "question": "고석태는 누구야?",
                "expected_keywords": ["제논", "대표"],
            },
            {
                "question": "제논이 최근 개최한 행사는 뭐야?",
                "expected_keywords": ["AIXperience", "AI Experience","교정행정 디지털 전환·AI 활용 국제 워크숍"],
            },
            {
                "question": "제논의 원에이전트는 어떤 서비스야?",
                "expected_keywords": ["원에이전트", "One Agent"],
            },
            {
                "question": "제논의 GEN 2.0은 뭐야?",
                "expected_keywords": ["GEN 2.0"],
            },
            {
                "question": "제논의 GenOS 2.0은 뭐야?",
                "expected_keywords": ["GenOS 2.0", "GenOS"],
            },
            {
                "question": "제논의 GenD는 어떤 제품이야?",
                "expected_keywords": ["GenD"],
            },
            {
                "question": "제논의 GenBuilder는 어떤 서비스야?",
                "expected_keywords": ["GenBuilder"],
            },
            {
                "question": "제논의 GenA는 어떤 서비스야?",
                "expected_keywords": ["GenA"],
            },
            {
                "question": "제논이 말하는 액셔너블 AI는 뭐야?",
                "expected_keywords": ["액셔너블 AI", "Actionable AI"],
            },
            {
                "question": "제논은 피지컬 AI 분야에서 무엇을 추진하고 있어?",
                "expected_keywords": ["피지컬 AI", "Physical AI"],
            },
            {
                "question": "제논이 최근 집중하는 사업 방향은 뭐야?",
                "expected_keywords": ["AI 에이전트", "에이전틱 AI", "액셔너블 AI"],
            },
            {
                "question": "제논이 최근 코스닥 진출에 성공했어 실패했어?",
                "expected_keywords": ["예비심사", "상장"],
            },
        ],
    },

    # ------------------------------------------------------------------
    # 당근 — 당근마켓 알바의 운수종사자 전자서류 제출 서비스와
    # 당근 커뮤니티 T&S팀의 AI 기반 안전 관리 기사 기준으로 구성한 골든셋.
    # 채소 당근 관련 기사는 정답으로 인정하지 않는다.
    # ------------------------------------------------------------------
    "당근": {
        "topic": "당근",
        "sense": "기업 당근(당근마켓)",   # 예전 검증어: 회사, 플랫폼
        "period_days": 50,
        "questions": [
            {
                "question": "당근마켓은 어떤 기관과 운수종사자 채용 서비스를 시작했어?",
                "expected_keywords": ["한국교통안전공단", "TS"],
            },
            {
                "question": "당근마켓 알바에서 전자적으로 제출할 수 있게 된 서류는 뭐야?",
                "expected_keywords": ["운수종사자 경력 및 자격 등에 관한 내역서", "경력내역서"],
            },
            {
                "question": "운수회사 구직자가 기존에 겪었던 불편은 뭐였어?",
                "expected_keywords": ["직접 방문", "서류 발급", "출력"],
            },
            {
                "question": "운수회사는 당근마켓 알바를 통해 어떤 정보를 확인할 수 있어?",
                "expected_keywords": ["경력과 자격 정보", "경력", "자격"],
            },
            {
                "question": "운수종사자 전자서류 제출 서비스는 어떤 사업의 하나로 마련됐어?",
                "expected_keywords": ["디지털서비스 개방", "행정안전부"],
            },
            {
                "question": "당근이 커뮤니티 안전 관리를 위해 만든 팀은 뭐야?",
                "expected_keywords": ["커뮤니티 T&S팀", "트러스트&세이프티"],
            },
            {
                "question": "당근 커뮤니티 T&S팀은 AI를 어디에 활용해?",
                "expected_keywords": ["욕설", "피싱 링크", "자동 분류", "이상 징후"],
            },
            {
                "question": "당근에서 AI가 판단한 뒤 최종 결정을 내리는 주체는 누구야?",
                "expected_keywords": ["사람", "상담사"],
            },
            {
                "question": "당근모임의 월간 활성 이용자 수는 몇 명을 넘었어?",
                "expected_keywords": ["1000만", "1천만"],
            },
            {
                "question": "당근 커뮤니티 T&S팀이 정책을 설계할 때 중요하게 보는 기준은 뭐야?",
                "expected_keywords": ["이용자 안전", "법률 준수", "서비스 품질"],
            },
        ],
    },

    # ------------------------------------------------------------------
    # 서울여대 — 제12대 이사장 이·취임, 2027학년도 수시모집 및
    # 노원구 대학연합회 관련 기사 기준으로 구성한 골든셋.
    # ------------------------------------------------------------------
    "서울여대": {
        "topic": "서울여대",
        "sense": "서울여자대학교",       # 예전 검증어: 서울여자대학교
        "period_days": 50,
        "questions": [
            {
                "question": "서울여대 재단 이름이 뭐야?",
                "expected_keywords": ["정의학원"],
            },
            {
                "question": "서울여대 제12대 이사장은 누구야?",
                "expected_keywords": ["황성은"],
            },
            {
                "question": "서울여대 제11대 이사장은 누구였어?",
                "expected_keywords": ["김경진"],
            },
            {
                "question": "서울여대 총장은 누구야?",
                "expected_keywords": ["이윤선"],
            },
            {
                "question": "황성은 이사장은 어떤 경력을 가지고 있어?",
                "expected_keywords": ["창동염광교회", "장로회신학대"],
            },
            {
                "question": "서울여대가 올해 신설한 학과는 뭐야?",
                "expected_keywords": ["AI융합학부", "인공지능전공"],
            },
            {
                "question": "서울여대 2027학년도 수시 정원내 모집인원은 몇 명이야?",
                "expected_keywords": ["890명"],
            },
            {
                "question": "서울여대 2027수시 원서접수 기간은 언제야?",
                "expected_keywords": ["9월8일", "9월 8일"],
            },
            {
                "question": "서울여대는 어떤 구에 있어?",
                "expected_keywords": ["노원구"],
            },
            {
                "question": "서울여대와 같은 지역에 있는 대학교는?",
                "expected_keywords": [
                    "광운대",
                    "삼육대",
                    "서울과학기술대",
                    "인덕대",
                    "한국성서대",
                    "육군사관학교",
                ],
            },
        ],
    },

    # ------------------------------------------------------------------
    # 강남 — 방송인 강남의 라디오스타 출연과 아내 이상화,
    # 스포츠 마사지·마라톤·취미 관련 기사 기준으로 구성한 골든셋.
    # 지역명 강남 관련 기사는 정답으로 인정하지 않는다.
    # ------------------------------------------------------------------
    "강남": {
        "topic": "강남",
        "sense": "방송인 강남",          # 예전 검증어: 방송인, 연예인
        "period_days": 50,
        "questions": [
            {
                "question": "강남의 아내는 누구야?",
                "expected_keywords": ["이상화"],
            },
            {
                "question": "강남은 최근 어떤 방송에 출연했어?",
                "expected_keywords": ["라디오스타", "라스"],
            },
            {
                "question": "강남이 아내 이상화를 위해 딴 자격증은 뭐야?",
                "expected_keywords": ["스포츠 마사지"],
            },
            {
                "question": "강남은 아내 이상화와 함께 어떤 특훈을 받았어?",
                "expected_keywords": ["마라톤", "태극기"],
            },
            {
                "question": "강남이 마라톤에 도전한 장소는 어디야?",
                "expected_keywords": ["북극"],
            },
            {
                "question": "강남이 최근 빠져 있는 취미는 뭐야?",
                "expected_keywords": ["포켓몬 카드"],
            },
        ],
    },
}


_DEBUG_PREVIEW_LIMIT = 5


def _texts(found) -> list[str]:
    """검색 결과를 글로: DB 모드는 청크(db.ChunkHit), 메모리 모드는 기사(Article)"""
    return [f"{x.article.title} {x.content}" if hasattr(x, "article") else f"{x.title} {x.content}"
            for x in found]


def _is_hit(found, expected_keywords: list[str]) -> bool:
    if not found:
        return False
    text = " ".join(_texts(found)).lower()
    return any(keyword.lower() in text for keyword in expected_keywords)


async def build_searcher(golden_set: dict, use_memory: bool = False):
    """(검색 함수, 검색 대상 기사 목록). DB에 topic이 있으면 DB, 없거나 --memory면 그 자리에서 수집한다.
    검색 함수: search(question, use_hyde, use_multi_query) → 검색 결과 목록"""
    query, sense_name = golden_set["topic"], golden_set["sense"]

    record = None if use_memory else get_db_topic(query, sense_name)
    if record is not None:
        from qupict import db
        from qupict.retrieval import search_topic

        articles = db.get_topic_articles(record.id, statuses=("관련", "보류"))
        print(f"[{query}] DB topic '{query} → {sense_name}' 사용 (관련·보류 기사 {len(articles)}건, pgvector 청크 검색)")

        def search(question: str, use_hyde: bool, use_multi_query: bool):
            return search_topic(record.id, question, record.selected, k=TOP_K,
                                use_hyde=use_hyde, use_multi_query=use_multi_query)
        return search, articles

    if not use_memory:
        print(f"[{query}] DB에 topic '{query} → {sense_name}'이 없어 그 자리에서 수집합니다 (메모리 모드).")
    import run
    from qupict.retrieval import retrieve_relevant_articles, topic_context

    selected = pick_sense(query, sense_name)
    session = await run.collect_news(selected, golden_set["period_days"])
    if session is None:
        return None, []

    def search(question: str, use_hyde: bool, use_multi_query: bool):
        return retrieve_relevant_articles(session.store, session.articles, question,
                                          topic=topic_context(selected), top_k=TOP_K,
                                          use_hyde=use_hyde, use_multi_query=use_multi_query)
    return search, session.articles


async def run_eval(golden_set_name: str, golden_set: dict, use_memory: bool = False) -> dict[str, tuple[int, int]]:
    """골든셋 하나를 4가지 조건으로 평가한다.

    반환값: {condition_label: (hits, n)}. 검색 대상을 만들지 못하면 빈 dict를 반환한다.
    """
    topic = golden_set["topic"]
    sense_name = golden_set["sense"]
    questions = golden_set["questions"]

    search, articles = await build_searcher(golden_set, use_memory)
    if search is None:
        print(f"[{golden_set_name}] 검색 대상 구축에 실패했습니다. 검색어/의미/기간을 확인해주세요.")
        return {}

    # 디버그: 골든셋 관련 기사가 검색 대상(필터 통과 기사)에 남아있는지 확인.
    debug_keywords = {kw for item in questions for kw in item["expected_keywords"]}
    found_titles = [
        a.title for a in articles
        if any(kw.lower() in f"{a.title} {a.content}".lower() for kw in debug_keywords)
    ]
    if found_titles:
        print(f"[DEBUG] 검색 대상({len(articles)}건) 안에서 골든셋 관련 기사 {len(found_titles)}건 발견"
              f" (최대 {_DEBUG_PREVIEW_LIMIT}건만 미리보기):")
        for t in found_titles[:_DEBUG_PREVIEW_LIMIT]:
            print(f"  - {t}")
        if len(found_titles) > _DEBUG_PREVIEW_LIMIT:
            print(f"  ...외 {len(found_titles) - _DEBUG_PREVIEW_LIMIT}건")
    else:
        print(f"[DEBUG] 검색 대상({len(articles)}건) 안에 골든셋 관련 기사가 하나도 없음 "
              f"→ 수집 기간 밖이거나 관련성 필터·크롤링 단계에서 탈락했을 가능성이 높음")

    print(f"\n=== 골든셋: {golden_set_name} (검색어='{topic}', 의미={sense_name}) ===")
    header = f"{'질문':<40}" + "".join(f"{label:<14}" for label, _, _ in CONDITIONS)
    print(header)
    print("-" * len(header))

    hits: dict[str, int] = {label: 0 for label, _, _ in CONDITIONS}

    for item in questions:
        question = item["question"]
        expected = item["expected_keywords"]

        row = f"{question:<40}"
        for label, use_hyde, use_multi_query in CONDITIONS:
            result = search(question, use_hyde, use_multi_query)
            hit = _is_hit(result, expected)
            hits[label] += hit
            row += f"{'O' if hit else 'X':<14}"
        print(row)

    n = len(questions)
    print("-" * len(header))
    summary = "  ".join(f"{label} = {hits[label]}/{n} ({hits[label]/n:.1%})" for label, _, _ in CONDITIONS)
    print(f"[{golden_set_name}] 검색 성공률(Hit Rate): {summary}\n")

    return {label: (hits[label], n) for label, _, _ in CONDITIONS}


async def run_all(use_memory: bool = False) -> None:
    # per_topic[topic_name] = {label: (hits, n)}
    per_topic: dict[str, dict[str, tuple[int, int]]] = {}

    for name, golden_set in GOLDEN_SETS.items():
        result = await run_eval(name, golden_set, use_memory)
        if result:
            per_topic[name] = result

    if not per_topic:
        print("평가된 골든셋이 없습니다.")
        return

    labels = [label for label, _, _ in CONDITIONS]

    print("=" * 90)
    print("전체 요약")
    print("=" * 90)
    header = f"{'주제':<10} {'문항 수':<8}" + "".join(f"{label:<20}" for label in labels)
    print(header)
    print("-" * len(header))
    for name, result in per_topic.items():
        n = next(iter(result.values()))[1]
        row = f"{name:<10} {n:<8}"
        for label in labels:
            hits, _ = result[label]
            row += f"{f'{hits}/{n} ({hits/n:.1%})':<20}"
        print(row)
    print("-" * len(header))

    # macro-average: 주제별 hit rate를 동일 가중치로 평균 (주제 수로 나눔)
    print("Macro 평균 (주제별 동일 가중치):")
    for label in labels:
        macro = sum(result[label][0] / result[label][1] for result in per_topic.values()) / len(per_topic)
        print(f"  {label:<12}: {macro:.1%}")

    # micro-average: 전체 문항을 다 합쳐서 계산 (문항 수가 많은 주제에 더 큰 가중치)
    print("\nMicro 평균 (전체 문항 기준):")
    total_n = sum(result[labels[0]][1] for result in per_topic.values())
    for label in labels:
        total_hits = sum(result[label][0] for result in per_topic.values())
        print(f"  {label:<12}: {total_hits}/{total_n} ({total_hits/total_n:.1%})")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--memory"]
    use_memory = "--memory" in sys.argv[1:]
    if args:
        name = args[0]
        if name not in GOLDEN_SETS:
            print(f"'{name}'에 해당하는 골든셋이 없습니다. 사용 가능: {list(GOLDEN_SETS.keys())}")
            return
        asyncio.run(run_eval(name, GOLDEN_SETS[name], use_memory))
    else:
        asyncio.run(run_all(use_memory))


if __name__ == "__main__":
    main()
