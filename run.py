"""
QuPicT 서비스 실행 파일 — 사용자가 검색어로 뉴스를 모으고 질문하는 서비스만 들어 있다

    python run.py 애플 30                 검색어 + 기간(일) → 의미 선택 → 수집 → 필터 → 인덱싱 → 질문
    python run.py                         검색어·기간을 입력받기
    python run.py 애플 30 --max 30        수집할 최대 기사 수 (기본 100)
    python run.py 애플 30 --memory        DB 없이 메모리에서 (예전 방식, FAISS 검색)

흐름 (실제 일은 qupict/의 단계 파일이 한다)
    [0단계] 의미 선택     qupict/sense.py
    [1단계] 수집          qupict/collect.py
    [2·3단계] 필터링      qupict/filtering.py
    [4단계] 인덱싱        qupict/indexing.py
    [5단계] 검색 · 답변   qupict/retrieval.py · qupict/qa.py
    저장                  qupict/db.py (서비스 DB)

다른 용도의 실행 파일
    python admin.py ...       서비스 DB 관리: 12개 대상 미리 수집·판정·인덱싱, 현황, 테이블 만들기
    python evaluate.py ...    평가: 정답 라벨, 질문 세트, 실험 A·B·C, 유사도 확인
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from typing import Callable

from qupict.cli import read_line, read_period_days, setup_console
from qupict.config import settings, validate_settings
from qupict.schemas import Answer, Article, SelectedSense

setup_console()

_EXIT_COMMANDS = {"종료", "exit", "quit", "q"}

# 예전에 run.py에 있다가 다른 실행 파일로 옮긴 명령 → 안내만 한다
MOVED_COMMANDS = {
    "collect-all": "admin.py", "filter-all": "admin.py", "index-all": "admin.py",
    "status": "admin.py", "init-db": "admin.py",
    "label": "evaluate.py", "make-qa": "evaluate.py", "exp-a": "evaluate.py",
    "exp-b": "evaluate.py", "exp-c": "evaluate.py", "sim": "evaluate.py",
}


# ---------------------------------------------------------------------------
# 답변 출력 · 질문 반복
# ---------------------------------------------------------------------------

def print_answer(answer: Answer) -> None:
    print("\n" + "=" * 60)
    print("질문:", answer.question)
    print("-" * 60)
    print(answer.answer)
    print("-" * 60)
    if answer.sources:
        # 답변에서 실제로 인용한 기사만 보여 준다 (번호는 답변 속 "(기사 N)"과 같게 유지)
        shown = answer.cited or list(range(1, len(answer.sources) + 1))
        print("출처:")
        for i in shown:
            src = answer.sources[i - 1]
            kind = f" [{answer.source_kinds[i - 1]}]" if i <= len(answer.source_kinds) else ""
            print(f"  [{i}] {src.title} ({src.press or '알 수 없음'}){kind} - {src.link}")
        if answer.cited and len(answer.cited) < len(answer.sources):
            print(f"  (검색은 됐지만 답변에 쓰이지 않은 기사 {len(answer.sources) - len(answer.cited)}건은 뺐습니다)")
    else:
        print("출처: 관련성 있는 기사를 찾지 못해 이번 답변에는 뉴스 출처가 없습니다.")
    print("=" * 60)


def run_qa_loop(label: str, answer_fn: Callable[[str], Answer]) -> None:
    print(f"\n'{label}' 뉴스에 대해 자유롭게 질문하세요. (종료: {'/'.join(sorted(_EXIT_COMMANDS))})\n")
    while True:
        question = read_line("질문> ")
        if question.lower() in _EXIT_COMMANDS:
            print("질문을 마칩니다.")
            break
        print_answer(answer_fn(question))


def _choose_sense(topic: str) -> SelectedSense:
    """[0단계] 의미 후보(DB 캐시 또는 LLM) → 사용자 선택"""
    from qupict.sense import choose_sense_interactively, load_sense_candidates

    candidates, source = load_sense_candidates(topic)
    print(f"[0단계] 의미 후보 → {source}")
    selected = choose_sense_interactively(candidates)
    print(f"[0단계] 선택한 의미 → {selected.sense.name}: {selected.sense.definition}")
    return selected


# ---------------------------------------------------------------------------
# DB 버전 (기본)
# ---------------------------------------------------------------------------

async def prepare_topic(selected: SelectedSense, period_days: int, max_articles: int = 100) -> int:
    """[1~4단계] 수집 → 필터 → 인덱싱을 DB에 쌓고 topic id를 돌려준다. 이미 한 일은 건너뛴다."""
    from qupict import db
    from qupict.collect import build_search_terms, collect_to_db
    from qupict.filtering import filter_topic
    from qupict.indexing import index_topic

    print(f"[1단계] 검색어 → {build_search_terms(selected)} (최근 {period_days}일)")
    collected = await collect_to_db(selected, period_days, max_articles)
    print(f"[1단계] 수집 → {collected.summary()}")
    filtered = await filter_topic(collected.topic_id, selected)
    print(f"[2·3단계] 필터링 (새 기사만) → {filtered.summary()}")
    indexed = await index_topic(collected.topic_id)
    print(f"[4단계] 인덱싱 (새 기사만) → {indexed.summary()}")
    usable = db.get_topic_articles(collected.topic_id, period_days=period_days, statuses=("관련", "보류"))
    print(f"→ 최근 {period_days}일 기사 중 질문에 쓸 수 있는 기사(관련·보류): {len(usable)}건")
    return collected.topic_id


def run_ask_db(selected: SelectedSense, period_days: int, max_articles: int = 100) -> None:
    from qupict.qa import generate_answer_from_chunks
    from qupict.retrieval import search_topic

    topic_id = asyncio.run(prepare_topic(selected, period_days, max_articles))

    def answer(question: str) -> Answer:
        hits = search_topic(topic_id, question, selected, period_days=period_days)
        return generate_answer_from_chunks(question, hits)

    run_qa_loop(f"{selected.query} ({selected.sense.name})", answer)


# ---------------------------------------------------------------------------
# 메모리 버전 (DB 없이, 예전 방식)
# ---------------------------------------------------------------------------

# 질문 답변에 쓸 판정. '보류'도 넣는다: LLM이 확신하지 못한 기사를 버리면 놓치는 정보(Recall)가 생기기 때문
ANSWER_STATUSES = ("관련", "보류")


@dataclass
class NewsSession:
    topic: str
    articles: list[Article]
    store: object  # FAISS


async def collect_news(selected: SelectedSense, period_days: int, max_articles: int = 100) -> NewsSession | None:
    """[메모리 버전] 고른 의미로 뉴스를 수집·필터링하고 FAISS 인메모리 벡터스토어를 만든다."""
    from qupict.collect import build_search_terms, collect_articles
    from qupict.filtering import filter_articles
    from qupict.retrieval import build_in_memory_store

    print(f"[1/4] 검색어 → {build_search_terms(selected)} (최근 {period_days}일)")
    metas, articles = await collect_articles(selected, period_days, max_articles)
    print(f"[2/4] 뉴스 수집 완료 → 검색 {len(metas)}건, 본문 확보 {len(articles)}건")
    if not articles:
        print("기사 본문을 가져오지 못했습니다. 다른 검색어나 기간으로 다시 시도해주세요.")
        return None
    outcome = await filter_articles(articles, selected)
    print(f"[3/4] 필터링 완료 → {outcome.summary()}")
    keep = outcome.links_with(*ANSWER_STATUSES)
    filtered = [a for a in articles if a.link in keep]
    if not filtered:
        print("관련 기사로 판정된 기사가 없습니다. 다른 검색어·의미·기간으로 다시 시도해주세요.")
        return None
    store = build_in_memory_store(filtered)
    print(f"[4/4] 인메모리 벡터스토어 구축 완료 → {len(filtered)}건(관련·보류)")
    return NewsSession(topic=selected.query, articles=filtered, store=store)


def run_ask_memory(selected: SelectedSense, period_days: int, max_articles: int = 100) -> None:
    from qupict.qa import generate_answer
    from qupict.retrieval import retrieve_relevant_articles

    session = asyncio.run(collect_news(selected, period_days, max_articles))
    if session is None:
        return

    def answer(question: str) -> Answer:
        relevant = retrieve_relevant_articles(session.store, session.articles, question, topic=session.topic)
        return generate_answer(question, relevant)

    run_qa_loop(session.topic, answer)


def _use_stored_criteria(selected: SelectedSense) -> SelectedSense:
    """같은 검색어·의미의 topic이 이미 DB에 있으면 저장된 기준을 그대로 쓴다 (DB·메모리 버전 모두).

    캐시된 LLM 후보에는 지어낸 정보가 남아 있을 수 있다 (실제: 방송인 강남 → "본명 전형수").
    그대로 쓰면 그 이름으로 검색하고 판정도 흔들린다. targets.yaml로 사람이 고쳐 둔 기준을 우선한다.
    """
    if not settings.database_url:
        return selected
    try:
        from qupict import db
        record = db.get_topic(selected.query, selected.sense.name)
    except Exception as e:
        print(f"  [경고] DB에서 저장된 기준을 읽지 못했습니다 ({type(e).__name__}). 고른 의미 그대로 씁니다.")
        return selected
    if record is None:
        return selected
    print(f"[0단계] DB에 저장된 기준을 사용 → 정의: {record.selected.sense.definition}")
    return record.selected


def cmd_ask(words: list[str], memory: bool, max_articles: int = 100) -> None:
    validate_settings()
    period_days = int(words.pop()) if words and words[-1].isdigit() else None
    topic = " ".join(words) if words else read_line("검색어를 입력하세요: ")
    if period_days is None:
        period_days = read_period_days()
    if not memory and not settings.database_url:
        print("DATABASE_URL이 없어 DB 없이(메모리 버전으로) 실행합니다.")
        memory = True
    selected = _use_stored_criteria(_choose_sense(topic))
    if memory:
        run_ask_memory(selected, period_days, max_articles)
    else:
        run_ask_db(selected, period_days, max_articles)


# ---------------------------------------------------------------------------
# 명령 해석
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python run.py", description="QuPicT 서비스: 검색어로 뉴스를 모으고 질문하기")
    p.add_argument("words", nargs="*", help="검색어 [기간(일)]")
    p.add_argument("--memory", action="store_true", help="DB 없이 메모리에서 (예전 방식)")
    p.add_argument("--max", type=int, default=100, dest="max_articles", help="수집할 최대 기사 수 (기본 100)")
    return p


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in MOVED_COMMANDS:       # 예전 명령을 쓰면 새 실행 파일을 알려 준다
        print(f"'{argv[0]}' 명령은 {MOVED_COMMANDS[argv[0]]}로 옮겼습니다: "
              f"python {MOVED_COMMANDS[argv[0]]} {' '.join(argv)}")
        sys.exit(2)
    if argv and argv[0] == "ask":                # 예전처럼 python run.py ask 애플 30 도 그대로 동작
        argv = argv[1:]
    args = _parser().parse_args(argv)
    cmd_ask(list(args.words), args.memory, args.max_articles)


if __name__ == "__main__":
    main()
