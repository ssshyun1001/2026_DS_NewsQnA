"""
AI News Q&A Agent - 메인 엔트리포인트

흐름:
  [1] 관심 분야 입력 (예: "AI 기업 제논")
  [2] 기간 입력 (숫자, 예: 7)
        -> Topic Analysis (주제 -> 검색 키워드/관련성 검증어 추출)
        -> News Search API (실시간 메타데이터 수집)
        -> Async Crawling (본문 추출)
        -> Relevance Filtering (동음이의어/무관 기사 제거)
        -> Embedding + Semantic Deduplication (중복 제거)
        -> In-Memory Vector Store 구축   ← 여기까지 한 번만 실행, 세션 동안 메모리에 유지
  [3] 질문 반복 입력 루프
        -> (매 질문마다) 벡터스토어에서 관련 기사 재검색 -> LLM 답변 생성 (수집된 뉴스만 근거로 사용)
        -> "종료" 입력 시까지 반복

사용법:
    python main.py                          # 인터랙티브: 관심 분야 -> 기간 -> 질문 반복
    python main.py "AI 기업 제논"           # 관심 분야만 인자로 전달, 기간은 인터랙티브 입력
    python main.py "AI 기업 제논" 7         # 관심 분야 + 기간(일)까지 인자로 전달
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS

from config import validate_settings
from app.query_understanding import analyze_topic
from app.news_search import search_news
from app.crawler import crawl_articles
from app.relevance_filter import filter_by_relevance
from app.deduplication import embed_articles, deduplicate_articles
from app.vector_store import build_in_memory_store, retrieve_relevant_articles
from app.rag_chain import generate_answer
from app.schemas import Answer, Article

_EXIT_COMMANDS = {"종료", "exit", "quit", "q"}

# Windows 콘솔(특히 PyCharm 내장 터미널)에서 한글 IME 입력이 인코딩 버퍼링
# 타이밍 문제로 깨지는 경우가 있어, stdin/stdout을 UTF-8로 명시적으로 고정한다.
if sys.platform.startswith("win"):
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass  # 일부 환경(예: 리다이렉션된 스트림)에서는 reconfigure가 불가할 수 있음

_INCOMPLETE_JAMO = set(
    "ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
    "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
)


def _looks_corrupted(text: str) -> bool:
    """입력 끝이 조합되지 않은 한글 낱자모(예: 'ㄷ', 'ㅇ')로 끝나면 IME 입력 버그로 판단한다."""
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
    """관심 분야 수집이 끝난 뒤 QA 루프 동안 유지되는 세션 상태."""
    topic: str
    articles: list[Article]
    store: FAISS


async def collect_news(topic: str, period_days: int) -> NewsSession | None:
    """[1단계] 관심 분야 + 기간을 입력받아 뉴스를 수집/정제하고 인메모리 벡터스토어를 구축한다."""
    search_query = analyze_topic(topic, period_days)
    print(
        f"[1/6] 주제 분석 완료 → 키워드: {search_query.optimized_keywords}, "
        f"검증기준: {search_query.filter_terms}, 기간: 최근 {search_query.period_days}일"
    )

    metas = await search_news(search_query)
    print(f"[2/6] 뉴스 검색 완료 → {len(metas)}건 발견")
    if not metas:
        print("관련 뉴스를 찾지 못했습니다. 다른 관심 분야로 다시 시도해주세요.")
        return None

    articles = await crawl_articles(metas)
    print(f"[3/6] 본문 크롤링 완료 → {len(articles)}건 본문 확보")
    if not articles:
        print("기사 본문을 가져오지 못했습니다. 다른 관심 분야로 다시 시도해주세요.")
        return None

    articles = filter_by_relevance(articles, search_query.filter_terms)
    print(f"[4/6] 관련성 필터 완료 → {len(articles)}건 (동음이의어/무관 기사 제거)")
    if not articles:
        print(
            "관련성 검증을 통과한 기사가 없습니다. 검색어가 너무 모호했을 수 있어요. "
            "다른 표현으로 다시 시도해주세요."
        )
        return None

    articles = await embed_articles(articles)
    deduped = deduplicate_articles(articles)
    print(f"[5/6] 중복 제거 완료 → {len(articles)}건 → {len(deduped)}건")

    store = build_in_memory_store(deduped)
    print(f"[6/6] 인메모리 벡터스토어 구축 완료 → 총 {len(deduped)}건의 기사에 대해 질문할 수 있습니다.\n")

    return NewsSession(topic=topic, articles=deduped, store=store)


def ask_once(session: NewsSession, question: str) -> Answer:
    """[2단계] 이미 구축된 세션의 벡터스토어에서 관련 기사를 재검색하고 답변을 생성한다."""
    relevant = retrieve_relevant_articles(session.store, session.articles, question)
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

    if len(sys.argv) >= 3 and sys.argv[-1].isdigit():
        # 예: python main.py "AI 기업 제논" 7
        topic = " ".join(sys.argv[1:-1])
        period_days = int(sys.argv[-1])
    elif len(sys.argv) > 1:
        # 예: python main.py "AI 기업 제논"  → 기간은 인터랙티브로 입력받음
        topic = " ".join(sys.argv[1:])
        period_days = _read_period_days()
    else:
        topic = _read_line("관심 분야를 입력하세요: ")
        period_days = _read_period_days()

    session = asyncio.run(collect_news(topic, period_days))
    if session is None:
        return

    run_qa_loop(session)


if __name__ == "__main__":
    main()