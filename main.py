"""
AI News Q&A Agent - 메인 엔트리포인트

흐름:
  [1] 관심 분야 입력 (예: "서울여대")
  [2] 검증 키워드 입력 (예: "전공")   ← 뉴스 검색 정확도를 높이기 위한 용도.
                                        "원문 키워드 + 검증어" 조합 쿼리를 우선
                                        검색해 동음이의어 노이즈를 줄인다.
  [3] 기간 입력 (숫자, 예: 7)
        -> Topic Analysis (주제 -> 검색 키워드 추출)
        -> 제외 키워드 자동 생성 (LLM이 topic만 보고 동음이의어/유사 명칭 등
           혼동 개체를 추론. 예: "서울여대" -> 숙명여대, 성신여대, 이화여대 등)
        -> News Search API (검증어를 조합 쿼리로 활용해 실시간 메타데이터 수집)
        -> Async Crawling (본문 추출)
        -> Embedding + Semantic Deduplication (중복 제거를 먼저 수행)
        -> Relevance Filtering (제외 키워드 기준으로 혼동 개체 기사 제거)
        -> In-Memory Vector Store 구축   ← 여기까지 한 번만 실행, 세션 동안 메모리에 유지

  검증어와 제외 키워드의 역할 분담:
    - 검증어(사용자 입력): "검색"을 좁히는 용도. "서울여대 전공"처럼 조합 쿼리를
      만들어 애초에 노이즈가 덜 섞인 검색 결과를 받는다.
    - 제외 키워드(LLM 자동 생성): "필터링"에서 쓰는 용도. topic과 혼동되는
      다른 개체(동음이의어/유사 명칭)를 실제로 다루는 기사를 사후에 걸러낸다.
    두 단계가 서로 다른 지점에서 노이즈를 줄이므로 상호 보완적이다.

  중복 제거를 관련성 필터보다 먼저 수행하는 이유:
    같은 사안을 다룬 유사 기사가 여러 건 있을 때, 필터를 먼저 적용하면
    LLM 경계 판정(2단계)이 사실상 같은 내용을 여러 번 반복해서 검사하게 된다.
    중복 제거를 먼저 하면 대표 기사 1건만 필터를 거치므로 LLM 호출 수가 줄어든다.

  제외 키워드를 세션 시작 시 한 번만 생성하는 이유:
    topic 하나당 1회만 필요한 정보이므로, 기사별로 반복 생성하지 않고 미리
    확정해 filter_by_relevance에 고정값으로 전달한다.

사용법:
    python main.py                                    # 완전 인터랙티브
    python main.py "서울여대"                          # topic만 인자, 나머지는 인터랙티브
    python main.py "서울여대" "전공" 7                  # topic + 검증어 + 기간(일)까지 인자로 전달
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS

from config import validate_settings
from app.query_understanding import analyze_topic
from app.news_search import search_news
from app.crawler import crawl_articles
from app.relevance_filter import filter_by_relevance, generate_exclusion_terms
from app.deduplication import embed_articles, deduplicate_articles
from app.vector_store import build_in_memory_store, retrieve_relevant_articles
from app.rag_chain import generate_answer
from app.schemas import Answer, Article

_EXIT_COMMANDS = {"종료", "exit", "quit", "q"}

if sys.platform.startswith("win"):
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_INCOMPLETE_JAMO = set(
    "ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
    "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
)


def _looks_corrupted(text: str) -> bool:
    return bool(text) and text[-1] in _INCOMPLETE_JAMO


def _read_line(prompt: str) -> str:
    while True:
        text = input(prompt).strip()
        if not text:
            print("입력이 비어 있습니다. 다시 입력해주세요.")
            continue
        if _looks_corrupted(text):
            print(
                f"입력이 깨진 것 같습니다 ('{text}'). "
                "한/영 전환 직후거나 IME 조합 중 Enter가 눌렸을 수 있어요. 다시 입력해주세요."
            )
            continue
        return text


def _read_validation_terms() -> list[str]:
    """검색 정확도를 높이기 위한 검증어를 콤마(,)로 구분해 여러 개 입력받는다.

    (필터링에는 더 이상 쓰이지 않고, 뉴스 검색 API 조합 쿼리에만 쓰인다.)
    예: '전공, 입학'
    """
    raw = _read_line(
        "검증 키워드를 입력하세요 (검색 정확도 향상용, 예: 전공 / 여러 개는 콤마로 구분, 없으면 Enter): "
    )
    terms = [t.strip() for t in raw.split(",") if t.strip()]
    return terms


def _read_period_days() -> int:
    while True:
        raw = input("기간을 숫자로 입력하세요 (최근 며칠, 예: 7): ").strip()
        if not raw:
            print("값이 비어 있습니다. 다시 입력해주세요.")
            continue
        try:
            days = int(raw)
        except ValueError:
            print(f"'{raw}'는 숫자가 아닙니다. 정수로 입력해주세요 (예: 7).")
            continue
        if days <= 0:
            print("기간은 1 이상의 정수여야 합니다.")
            continue
        return days


@dataclass
class NewsSession:
    topic: str
    articles: list[Article]
    store: FAISS
    validation_terms: list[str] = field(default_factory=list)
    exclusion_terms: list[str] = field(default_factory=list)


async def collect_news(
    topic: str,
    period_days: int,
    validation_terms: list[str] | None = None,
) -> NewsSession | None:
    """[1단계] 관심 분야 + 검증어 + 기간을 입력받아 뉴스를 수집/정제하고 인메모리 벡터스토어를 구축한다.

    핵심 변경점:
      - 검증어는 관련성 필터(포지티브 필터)에서는 더 이상 쓰이지 않는다.
        대신 뉴스 검색 단계에서 "키워드+검증어" 조합 쿼리로 검색 노이즈를
        줄이는 용도로만 쓰인다 (news_search.py의 filter_terms 로직).
      - 관련성 필터는 제외 키워드(exclusion_terms)만으로 동작한다. 이는
        topic만 보고 LLM이 자동 추론한 "혼동 개체"(동음이의어 + 유사 명칭)다.
      - 중복 제거를 관련성 필터보다 먼저 수행해, 유사 기사 여러 건이 각각
        LLM 경계 판정을 반복하지 않도록 한다.
    """
    validation_terms = validation_terms or []

    search_query = analyze_topic(topic, period_days)
    # 검증어는 검색 조합 쿼리 용도로만 채운다 (관련성 필터에는 전달하지 않음).
    search_query.filter_terms = validation_terms

    print(
        f"[1/7] 주제 분석 완료 → 키워드: {search_query.optimized_keywords}, "
        f"검증어(검색용): {validation_terms or '없음'}, 기간: 최근 {search_query.period_days}일"
    )

    exclusion_terms = await generate_exclusion_terms(topic)
    if exclusion_terms:
        print(f"[2/7] 제외 키워드 자동 생성 완료 → {exclusion_terms}")
    else:
        print("[2/7] 제외 키워드 없음 (혼동 위험이 낮은 주제로 판단됨)")

    metas = await search_news(search_query)
    print(f"[3/7] 뉴스 검색 완료 → {len(metas)}건 발견")
    if not metas:
        print("관련 뉴스를 찾지 못했습니다. 다른 관심 분야로 다시 시도해주세요.")
        return None

    articles = await crawl_articles(metas)
    print(f"[4/7] 본문 크롤링 완료 → {len(articles)}건 본문 확보")
    if not articles:
        print("기사 본문을 가져오지 못했습니다. 다른 관심 분야로 다시 시도해주세요.")
        return None

    # 임베딩을 먼저 1회 계산해서 중복 제거와 relevance filter가 모두 재사용한다.
    articles = await embed_articles(articles)

    # 중복 제거를 먼저 수행: 같은 사안을 다룬 유사 기사들을 대표 기사 1건으로 합쳐서,
    # 다음 단계인 관련성 필터(특히 LLM 경계 판정)가 같은 내용을 반복 검사하지 않게 한다.
    deduped = deduplicate_articles(articles)
    print(f"[5/7] 중복 제거 완료 → {len(articles)}건 → {len(deduped)}건")

    filtered = await filter_by_relevance(deduped, topic, exclusion_terms=exclusion_terms)
    print(f"[6/7] 관련성 필터 완료 → {len(filtered)}건 (제외어: {exclusion_terms or '없음'})")

    if not filtered:
        print(
            "관련성 검증을 통과한 기사가 없습니다. 제외 키워드가 너무 광범위했을 수 있어요. "
            "다른 관심 분야로 다시 시도해주세요."
        )
        return None

    store = build_in_memory_store(filtered)
    print(f"[7/7] 인메모리 벡터스토어 구축 완료 → 총 {len(filtered)}건의 기사에 대해 질문할 수 있습니다.\n")

    return NewsSession(
        topic=topic,
        articles=filtered,
        store=store,
        validation_terms=validation_terms,
        exclusion_terms=exclusion_terms,
    )


def ask_once(session: NewsSession, question: str) -> Answer:
    relevant = retrieve_relevant_articles(session.store, session.articles, question, topic=session.topic)
    return generate_answer(question, relevant)


def print_answer(answer: Answer) -> None:
    print("\n" + "=" * 60)
    print("질문:", answer.question)
    print("-" * 60)
    print(answer.answer)
    print("-" * 60)
    if answer.sources:
        print("출처:")
        for i, src in enumerate(answer.sources, start=1):
            print(f"  [{i}] {src.title} ({src.press or '알 수 없음'}) - {src.link}")
    else:
        print("출처: 관련성 있는 기사를 찾지 못해 이번 답변에는 뉴스 출처가 없습니다.")
    print("=" * 60)


def run_qa_loop(session: NewsSession) -> None:
    print(f"'{session.topic}' 관련 뉴스 {len(session.articles)}건이 준비되었습니다.")
    print(f"이제 이 뉴스에 대해 자유롭게 질문하세요. (종료하려면 '{'/'.join(sorted(_EXIT_COMMANDS))}' 입력)\n")

    while True:
        question = _read_line("질문> ")
        if question.lower() in _EXIT_COMMANDS:
            print("QA 세션을 종료합니다.")
            break

        answer = ask_once(session, question)
        print_answer(answer)


def main() -> None:
    validate_settings()

    if len(sys.argv) >= 4 and sys.argv[-1].isdigit():
        # 예: python main.py "서울여대" "전공" 7
        topic = sys.argv[1]
        validation_terms = [t.strip() for t in sys.argv[2].split(",") if t.strip()]
        period_days = int(sys.argv[-1])
    elif len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
        validation_terms = _read_validation_terms()
        period_days = _read_period_days()
    else:
        topic = _read_line("관심 분야를 입력하세요: ")
        validation_terms = _read_validation_terms()
        period_days = _read_period_days()

    session = asyncio.run(collect_news(topic, period_days, validation_terms))
    if session is None:
        return

    run_qa_loop(session)


if __name__ == "__main__":
    main()