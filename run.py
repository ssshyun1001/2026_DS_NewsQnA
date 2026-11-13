"""
QuPicT 서비스 실행 파일 — 사용자가 검색어로 뉴스를 모으고 질문하는 서비스만 들어 있다

    python run.py 애플 30                 검색어 + 기간(일) → 의미 선택 → 수집 → 필터 → 인덱싱 → 질문
    python run.py                         검색어·기간을 입력받기
    python run.py 애플 30 --max 30        수집할 최대 기사 수 (기본 100)
    python run.py 애플 30 --memory        DB 없이 메모리에서 (예전 방식, FAISS 검색)
    python run.py 애플 30 --timing        단계별 걸린 시간 표 보기 (속도 측정, qupict/timing.py)

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
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from qupict import timing
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

def print_answer_header(question: str) -> None:
    print("\n" + "=" * 60)
    print("질문:", question)
    print("-" * 60)


def print_answer(answer: Answer, streamed: bool = False) -> None:
    """streamed=True: 머리(질문)와 답변 글은 스트리밍하면서 이미 찍었으니 출처부터 찍는다."""
    if not streamed:
        print_answer_header(answer.question)
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


def run_qa_loop(label: str, answer_fn: Callable[[str], Answer], streamed: bool = False,
                first_question: str | None = None) -> None:
    """streamed=True: answer_fn이 질문 머리와 답변 글을 스트리밍으로 직접 찍는다.
    first_question: 주면 사용자 입력을 받기 전에 이 질문에 먼저 답한다 (고정 첫 질문: 수집된 기사 전체 요약)."""
    if first_question:
        timing.reset()
        print_answer(answer_fn(first_question), streamed=streamed)
        timing.report(f"질문 답변 — {first_question[:20]}")
    print(f"\n'{label}' 뉴스에 대해 자유롭게 질문하세요. (종료: {'/'.join(sorted(_EXIT_COMMANDS))})\n")
    while True:
        question = read_line("질문> ")
        if question.lower() in _EXIT_COMMANDS:
            print("질문을 마칩니다.")
            break
        timing.reset()
        print_answer(answer_fn(question), streamed=streamed)
        timing.report(f"질문 답변 — {question[:20]}")


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

async def prepare_topic(selected: SelectedSense, period_days: int, max_articles: int = 100,
                        log: Callable[[str], None] = print) -> int:
    """[1~4단계] 수집 → 필터 → 인덱싱을 DB에 쌓고 topic id를 돌려준다. 이미 한 일은 건너뛴다.
    log: 진행 상황을 한 줄씩 받을 함수 (기본 = 터미널에 출력. web.py는 화면에 보여 줄 목록에 쌓는다)"""
    from qupict import db
    from qupict.collect import build_search_terms, collect_to_db
    from qupict.filtering import filter_topic
    from qupict.indexing import index_topic

    timing.reset()
    log(f"[1단계] 검색어 → {build_search_terms(selected)} (최근 {period_days}일)")
    with timing.timed("[1단계] 수집"):
        collected = await collect_to_db(selected, period_days, max_articles)
    log(f"[1단계] 수집 → {collected.summary()}")
    with timing.timed("[2·3단계] 필터링"):
        filtered = await filter_topic(collected.topic_id, selected)
    log(f"[2·3단계] 필터링 (새 기사만) → {filtered.summary()}")
    # 질문 가능 기사 세기는 판정만 보므로(인덱싱 결과와 무관) 인덱싱하는 동안 동시에 해 둔다
    usable_task = asyncio.create_task(asyncio.to_thread(
        db.get_topic_articles, collected.topic_id, period_days=period_days, statuses=("관련", "보류")))
    # 준비 단계는 본문만. 이미지 → 글(OCR·VLM)은 "준비 완료" 뒤 add_image_text가 백그라운드에서 한다
    with timing.timed("[4단계] 인덱싱 (본문)"):
        indexed = await index_topic(collected.topic_id, describe_images=False)
    log(f"[4단계] 인덱싱 (새 기사만) → {indexed.summary()}")
    with timing.timed("질문 가능 기사 세기(DB, 인덱싱과 동시)"):
        usable = await usable_task
    log(f"→ 최근 {period_days}일 기사 중 질문에 쓸 수 있는 기사(관련·보류): {len(usable)}건")
    timing.report("수집 → 필터링 → 인덱싱")
    return collected.topic_id


IMAGE_TEXT_NAMES = {"ocr": "이미지 속 글자 읽기(OCR)", "vlm": "이미지 설명(VLM)"}


def image_text_mode() -> str | None:
    """준비 완료 뒤 할 이미지 → 글 방법 ("ocr" / "vlm"). 하지 않으면 None.
    설정은 IMAGE_TEXT (qupict/config.py). OCR인데 easyocr가 설치되어 있지 않으면 하지 않는다."""
    mode = settings.image_text
    if mode not in IMAGE_TEXT_NAMES:
        return None
    if mode == "ocr":
        import importlib.util
        if importlib.util.find_spec("easyocr") is None:
            return None
    return mode


async def add_image_text(topic_id: int, log: Callable[[str], None] = print) -> None:
    """[4단계 뒤] 기사 이미지를 글로 바꿔(OCR 또는 VLM) 이미지 청크를 더한다.
    "준비 완료" 뒤에 돌려서, 그동안 질문을 받는다. 끝나면 그 뒤 질문부터 이미지 내용도 검색된다."""
    from qupict.indexing import index_topic

    mode = image_text_mode()
    if mode is None:
        return
    name = IMAGE_TEXT_NAMES[mode]
    timing.reset()
    log(f"[이미지] {name}를 뒤에서 진행합니다. 그동안 질문해도 됩니다.")
    with timing.timed(name):
        result = await index_topic(topic_id, describe_images=(mode == "vlm"), ocr_images=(mode == "ocr"))
    log(f"[이미지] {name} 끝 → {result.summary()}")
    timing.report(name)


def start_image_text_in_background(topic_id: int, log: Callable[[str], None] = print) -> threading.Thread:
    """add_image_text를 다른 스레드에서 돌린다 (터미널 질문 반복과 동시에). 실패해도 질문에는 영향 없음."""
    def work() -> None:
        try:
            asyncio.run(add_image_text(topic_id, log=log))
        except Exception as e:
            log(f"[이미지] 실패해서 건너뜁니다 ({type(e).__name__}: {e})")

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread


# 준비(수집 → 필터링 → 인덱싱)가 끝난 뒤 "준비 완료"를 알리기 전에 기다리는 시간(초).
# 팀 결정(2026-11-13): 준비 직후 첫 질문에서 저장된 내용을 가끔 못 찾는 것 같아 여유를 둔다.
# 참고: 저장은 각 단계 안에서 commit까지 끝난 뒤 다음 단계로 넘어가므로(db.connect), 이 대기가 없어도
#   저장 자체는 끝나 있다. 원인이 확인되면 0으로 줄여도 된다.
PREPARE_SETTLE_SECONDS = 3


def wait_after_prepare(log: Callable[[str], None] = print) -> None:
    """준비가 끝난 뒤 PREPARE_SETTLE_SECONDS초 기다린다 (0이면 바로 돌아온다)."""
    if PREPARE_SETTLE_SECONDS > 0:
        log(f"저장 내용을 정리하는 중… ({PREPARE_SETTLE_SECONDS}초)")
        time.sleep(PREPARE_SETTLE_SECONDS)


def oldest_searchable_or_none(topic_id: int, period_days: int | None):
    """질문에 쓸 수 있는 기사 중 가장 오래된 발행일 (답변에서 "○○부터 오늘까지" 기간을 밝힐 때).
    조회가 실패해도 질문 답변은 해야 하므로 None을 돌려준다 (그때 답변은 기간 없이 "최신 뉴스를 못 찾았다"고 한다)."""
    from qupict.retrieval import oldest_searchable_date
    try:
        return oldest_searchable_date(topic_id, period_days)
    except Exception as e:
        print(f"  [경고] 기사 기간 조회 실패 ({type(e).__name__}) — 답변에 기간을 표시하지 않습니다")
        return None


def run_ask_db(selected: SelectedSense, period_days: int, max_articles: int = 100) -> None:
    from qupict.qa import answer_period, answer_subject, first_question, generate_answer_from_chunks
    from qupict.retrieval import search_topic

    topic_id = asyncio.run(prepare_topic(selected, period_days, max_articles))
    wait_after_prepare()
    oldest = oldest_searchable_or_none(topic_id, period_days)
    if image_text_mode():
        # 질문 입력 줄과 섞이지 않게 한 줄 띄우고 찍는다
        start_image_text_in_background(topic_id, log=lambda s: print(f"\n{s}"))

    def answer(question: str) -> Answer:
        with timing.timed("[5단계] 근거 검색") as t:
            hits = search_topic(topic_id, question, selected, period_days=period_days)
            t.note = f"기준 통과 청크 {len(hits)}개"
        # 답변은 스트리밍: 만들어지는 글자를 바로 화면에 찍는다 (첫 글자가 빨리 보임, 완성된 답변은 같음)
        print_answer_header(question)
        result = generate_answer_from_chunks(question, hits, on_token=lambda s: print(s, end="", flush=True),
                                             subject=answer_subject(selected), period=answer_period(oldest))
        print()
        return result

    run_qa_loop(f"{selected.query} ({selected.sense.name})", answer, streamed=True,
                first_question=first_question(selected.query))


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
    # {중복 기사 link: 대표 기사 link} — 웹 화면의 "많이 보도된 기사"(qupict/hot.py)가 센다
    duplicates: dict[str, str] = field(default_factory=dict)


async def collect_news(selected: SelectedSense, period_days: int, max_articles: int = 100,
                       log: Callable[[str], None] = print) -> NewsSession | None:
    """[메모리 버전] 고른 의미로 뉴스를 수집·필터링하고 FAISS 인메모리 벡터스토어를 만든다.
    log: 진행 상황을 한 줄씩 받을 함수 (기본 = 터미널에 출력. web.py는 화면에 보여 줄 목록에 쌓는다)"""
    from qupict.collect import build_search_terms, collect_articles
    from qupict.filtering import filter_articles
    from qupict.retrieval import build_in_memory_store

    timing.reset()
    log(f"[1/4] 검색어 → {build_search_terms(selected)} (최근 {period_days}일)")
    with timing.timed("[1단계] 검색 + 본문 크롤링") as t:
        metas, articles = await collect_articles(selected, period_days, max_articles)
        t.note = f"검색 {len(metas)}건 → 본문 {len(articles)}건"
    log(f"[2/4] 뉴스 수집 완료 → 검색 {len(metas)}건, 본문 확보 {len(articles)}건")
    if not articles:
        log("기사 본문을 가져오지 못했습니다. 다른 검색어나 기간으로 다시 시도해주세요.")
        return None
    with timing.timed("[2·3단계] 필터링"):
        outcome = await filter_articles(articles, selected)
    log(f"[3/4] 필터링 완료 → {outcome.summary()}")
    keep = outcome.links_with(*ANSWER_STATUSES)
    filtered = [a for a in articles if a.link in keep]
    if not filtered:
        log("관련 기사로 판정된 기사가 없습니다. 다른 검색어·의미·기간으로 다시 시도해주세요.")
        return None
    with timing.timed("[4단계] 인메모리 벡터스토어"):
        store = build_in_memory_store(filtered)
    log(f"[4/4] 인메모리 벡터스토어 구축 완료 → {len(filtered)}건(관련·보류)")
    timing.report("수집 → 필터링 → 벡터스토어 (메모리 버전)")
    return NewsSession(topic=selected.query, articles=filtered, store=store, duplicates=outcome.duplicates)


def run_ask_memory(selected: SelectedSense, period_days: int, max_articles: int = 100) -> None:
    from qupict.qa import answer_period, answer_subject, first_question, generate_answer, oldest_date
    from qupict.retrieval import retrieve_relevant_articles

    session = asyncio.run(collect_news(selected, period_days, max_articles))
    if session is None:
        return
    oldest = oldest_date(session.articles)

    def answer(question: str) -> Answer:
        with timing.timed("[5단계] 근거 검색"):
            relevant = retrieve_relevant_articles(session.store, session.articles, question, topic=session.topic)
        return generate_answer(question, relevant, subject=answer_subject(selected), period=answer_period(oldest))

    run_qa_loop(session.topic, answer, first_question=first_question(selected.query))


def cmd_ask(words: list[str], memory: bool, max_articles: int = 100) -> None:
    validate_settings()
    period_days = int(words.pop()) if words and words[-1].isdigit() else None
    topic = " ".join(words) if words else read_line("검색어를 입력하세요: ")
    if period_days is None:
        period_days = read_period_days()
    if not memory and not settings.database_url:
        print("DATABASE_URL이 없어 DB 없이(메모리 버전으로) 실행합니다.")
        memory = True
    from qupict.collect import warm_up_parser
    warm_up_parser()        # 크롤링 해석용 프로세스(약 2초)를 의미를 고르는 동안 미리 띄워 둔다
    selected = _choose_sense(topic)
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
    p.add_argument("--timing", action="store_true", help="단계별 걸린 시간을 표로 보여 주기 (속도 측정용)")
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
    if args.timing:
        timing.enable()
    cmd_ask(list(args.words), args.memory, args.max_articles)


if __name__ == "__main__":
    main()
