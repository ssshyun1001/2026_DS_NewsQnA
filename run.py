"""
QuPicT 실행 파일 — 검색어 하나로 수집부터 질문 답변까지 (메모리에서 실행, DB 사용 전 버전)

흐름:
  [0단계] 검색어 입력 → LLM이 의미 후보 제시 → 사용자가 하나 선택      (qupict/sense.py)
  [1단계] 고른 의미의 연관어·별칭으로 네이버 뉴스 검색 → 본문·이미지 수집  (qupict/collect.py)
  [2·3단계] 임베딩 → 중복 제거 → 관련성 필터                              (qupict/filtering.py, 아직 기존 방식)
  [5단계] FAISS 검색 → 근거 기반 답변                                      (qupict/retrieval.py, qupict/qa.py)

  기존 "검증 키워드" 입력은 없어졌다. 의미 선택이 그 역할을 대신한다.
  (검색: "검색어 + 검증어" 조합 → "검색어 + 고른 의미의 연관어" 조합)

  중복 제거를 관련성 필터보다 먼저 하는 이유:
    같은 사안을 다룬 비슷한 기사가 여러 건이면, 필터를 먼저 할 경우 LLM 판정이
    같은 내용을 여러 번 검사하게 된다. 대표 기사 1건만 남긴 뒤 필터를 거치면 LLM 호출이 줄어든다.

TODO: DB가 준비되면 단계별 명령(init-db, senses, collect, filter, index, ask)으로 바꾼다.

사용법:
    python run.py                 # 검색어·기간 모두 입력받기
    python run.py 애플            # 검색어만 인자로, 기간은 입력받기
    python run.py 애플 30         # 검색어 + 기간(일)
    python run.py init-db         # DB 테이블 만들기 (schema.sql)
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS

from qupict.config import validate_settings
from qupict.sense import choose_sense_interactively, generate_sense_candidates
from qupict.collect import build_search_terms, collect_articles
from qupict.filtering import (
    deduplicate_articles,
    embed_articles,
    filter_by_relevance,
    generate_exclusion_terms,
)
from qupict.retrieval import build_in_memory_store, retrieve_relevant_articles
from qupict.qa import generate_answer
from qupict.schemas import Answer, Article, SelectedSense

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
    selected: SelectedSense | None = None
    exclusion_terms: list[str] = field(default_factory=list)


async def collect_news(selected: SelectedSense, period_days: int) -> NewsSession | None:
    """[1~5단계] 고른 의미로 뉴스를 수집·정제하고 인메모리 벡터스토어를 만든다."""
    topic = selected.query
    print(f"[1/6] 검색어 → {build_search_terms(selected)} (최근 {period_days}일)")

    # 관련성 필터는 아직 기존 방식이다. 고른 의미 이름을 넘겨서 LLM이 '다른 의미'를 제외어로 만들게 한다.
    # (기존에는 사용자가 입력한 검증 키워드가 이 역할을 했다. 필터링 개편 때 SelectedSense를 직접 쓰도록 바꾼다)
    exclusion_terms = await generate_exclusion_terms(topic, [selected.sense.name])
    print(f"[2/6] 제외 키워드 → {exclusion_terms or '없음'}")

    metas, articles = await collect_articles(selected, period_days)
    with_images = sum(1 for a in articles if a.images)
    print(f"[3/6] 뉴스 수집 완료 → 검색 {len(metas)}건, 본문 확보 {len(articles)}건 (이미지 있는 기사 {with_images}건)")
    if not articles:
        print("기사 본문을 가져오지 못했습니다. 다른 검색어나 기간으로 다시 시도해주세요.")
        return None

    # 임베딩을 먼저 1회 계산해서 중복 제거와 relevance filter가 모두 재사용한다.
    articles = await embed_articles(articles)

    # 중복 제거를 먼저 수행: 같은 사안을 다룬 유사 기사들을 대표 기사 1건으로 합쳐서,
    # 다음 단계인 관련성 필터(특히 LLM 경계 판정)가 같은 내용을 반복 검사하지 않게 한다.
    deduped = deduplicate_articles(articles)
    print(f"[4/6] 중복 제거 완료 → {len(articles)}건 → {len(deduped)}건")

    filtered = await filter_by_relevance(deduped, topic, exclusion_terms=exclusion_terms)
    print(f"[5/6] 관련성 필터 완료 → {len(filtered)}건 (제외어: {exclusion_terms or '없음'})")

    if not filtered:
        print(
            "관련성 검증을 통과한 기사가 없습니다. 제외 키워드가 너무 광범위했을 수 있어요. "
            "다른 검색어로 다시 시도해주세요."
        )
        return None

    store = build_in_memory_store(filtered)
    print(f"[6/6] 인메모리 벡터스토어 구축 완료 → 총 {len(filtered)}건의 기사에 대해 질문할 수 있습니다.\n")

    return NewsSession(
        topic=topic,
        articles=filtered,
        store=store,
        selected=selected,
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
    args = sys.argv[1:]
    if args[:1] == ["init-db"]:              # python run.py init-db
        from qupict.db import init_db
        init_db()
        print("DB 테이블을 만들었습니다 (schema.sql).")
        return

    validate_settings()
    if args and args[-1].isdigit():          # 예: python run.py 애플 30
        period_days = int(args.pop())
    else:
        period_days = None
    topic = " ".join(args) if args else _read_line("검색어를 입력하세요: ")
    if period_days is None:
        period_days = _read_period_days()

    # [0단계] 의미 선택
    selected = choose_sense_interactively(generate_sense_candidates(topic))
    print(f"[0/6] 선택한 의미 → {selected.sense.name}: {selected.sense.definition}")

    session = asyncio.run(collect_news(selected, period_days))
    if session is None:
        return

    run_qa_loop(session)


if __name__ == "__main__":
    main()