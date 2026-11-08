"""
QuPicT 실행 파일

질문하기 (검색어 하나)
    python run.py 애플 30                 검색어 + 기간(일) → 의미 선택 → 수집 → 필터 → 인덱싱 → 질문
    python run.py ask 애플 30             위와 같음
    python run.py ask 애플 30 --max 30    수집할 최대 기사 수 (기본 100)
    python run.py ask 애플 30 --memory    DB 없이 메모리에서 (예전 방식, FAISS 검색)

모이는 날: targets.yaml 대상 일괄 처리 (--target 강남 처럼 검색어 하나만 할 수도 있음)
    python run.py collect-all [--days 30] [--max 300]   0·1·2단계: 의미 선택 → 뉴스 수집 → 중복 제거 → DB 저장
    python run.py filter-all  [--redo]                  2·3단계: 중복 제거 → 규칙 → LLM 판정 → DB 저장
    python run.py label       [--target 강남] [--limit 150]  최신 대표 기사 150건에 정답 라벨 (관련 / 무관 / 보류)
    python run.py index-all   [--no-images]             4단계: 이미지 설명 → 청크 → 임베딩 → DB 저장
    python run.py status                                topic별 진행 현황

실험 (기획안 7단계, 결과는 experiments/results/에 저장)
    python run.py exp-a   [--target 강남]               실험 A: 필터링 방법 비교 (정답 라벨 필요)
    python run.py make-qa [--target 강남] [--suggest]   실험 B·C용 질문 세트 만들기 (experiments/qa_set.yaml)
    python run.py exp-b   [--target 강남] [--faiss]     실험 B: 검색 방식 비교 (Recall@5, MRR)
    python run.py exp-c   [--target 강남] [--limit 10]  실험 C: LLM 단독 vs 기본 RAG vs QuPicT 답변 채점
    python run.py sim     [--target 강남] ["질문"]      유사도 확인: 기준값 없이 청크 유사도 보기
                                                        (질문 없이 → 질문 세트로 기준값 0.40/0.45/0.50 비교)

기타
    python run.py init-db                               DB 테이블 만들기 (schema.sql)

모든 일괄 명령은 "아직 안 한 것만" 처리하므로 여러 번 실행해도 안전하고 비용이 다시 들지 않는다.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from typing import Callable

from qupict.config import settings, validate_settings
from qupict.schemas import Answer, Article, SelectedSense

_EXIT_COMMANDS = {"종료", "exit", "quit", "q"}
DEFAULT_PERIOD_DAYS = 30

if sys.platform.startswith("win"):
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 입력 도우미 (한글 입력 깨짐 확인)
# ---------------------------------------------------------------------------

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
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("기간은 1 이상의 정수로 입력해주세요 (예: 7).")


def _choose_number(prompt: str, count: int) -> int:
    """1~count 중 하나를 입력받아 0부터 시작하는 번호로 돌려준다."""
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        print(f"1~{count} 사이 번호를 입력하세요.")


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
        question = _read_line("질문> ")
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
# ask — DB 버전
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
# ask — 메모리 버전 (DB 없이, 예전 방식)
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
    topic = " ".join(words) if words else _read_line("검색어를 입력하세요: ")
    if period_days is None:
        period_days = _read_period_days()
    if not memory and not settings.database_url:
        print("DATABASE_URL이 없어 DB 없이(메모리 버전으로) 실행합니다.")
        memory = True
    selected = _use_stored_criteria(_choose_sense(topic))
    if memory:
        run_ask_memory(selected, period_days, max_articles)
    else:
        run_ask_db(selected, period_days, max_articles)


# ---------------------------------------------------------------------------
# 일괄 명령 (targets.yaml)
# ---------------------------------------------------------------------------

def _for_each_target(query: str | None, title: str, step) -> None:
    """targets.yaml 대상마다 step(target)을 실행하고 결과 요약을 출력한다. 하나가 실패해도 나머지는 계속한다."""
    from qupict.pipeline import pick_targets, target_label

    targets = pick_targets(query)
    print(f"{title}: 대상 {len(targets)}개\n")
    failed = []
    for i, target in enumerate(targets, 1):
        try:
            result = asyncio.run(step(target))
            print(f"[{i}/{len(targets)}] {target_label(target)} → {result.summary()}")
        except Exception as e:
            failed.append(target_label(target))
            print(f"[{i}/{len(targets)}] {target_label(target)} → 실패: {type(e).__name__}: {str(e)[:150]}")
    print(f"\n완료 (실패 {len(failed)}개{': ' + ', '.join(failed) if failed else ''})")


def cmd_collect_all(query: str | None, days: int, max_articles: int | None = None) -> None:
    from qupict.pipeline import COLLECT_MAX_PER_TARGET, LABEL_PER_TARGET, collect_target
    validate_settings()
    max_articles = COLLECT_MAX_PER_TARGET if max_articles is None else max_articles
    _for_each_target(query, f"수집 → 중복 제거 (최근 {days}일, 대상당 최대 {max_articles}건, "
                            f"라벨 목표: 대표 기사 {LABEL_PER_TARGET}건)",
                     lambda t: collect_target(t, days, max_articles))


def cmd_filter_all(query: str | None, redo: bool) -> None:
    from qupict.pipeline import filter_target
    _for_each_target(query, "필터링" + (" (처음부터 다시)" if redo else " (새 기사만)"),
                     lambda t: filter_target(t, redo=redo))


def cmd_index_all(query: str | None, describe_images: bool) -> None:
    from qupict.pipeline import index_target
    _for_each_target(query, "인덱싱" + ("" if describe_images else " (이미지 설명 없이)"),
                     lambda t: index_target(t, describe_images=describe_images))


def cmd_status() -> None:
    from qupict.pipeline import LABEL_PER_TARGET, topic_status

    rows = topic_status()
    if not rows:
        print("아직 DB에 topic이 없습니다. collect-all 또는 ask로 수집하세요.")
        return
    print(f"{'id':>3} {'검색어 → 의미':<28} {'기사':>4} {'중복':>4} {'미판정':>5} {'규칙무관':>7} "
          f"{'관련':>4} {'무관':>4} {'보류':>4} {'대표':>4} {'라벨':>9} {'청크':>4}")
    for r in rows:
        name = f"{r.query} → {r.sense_name}"[:28]
        print(f"{r.topic_id:>3} {name:<28} {r.articles:>4} {r.duplicates:>4} {r.unjudged:>5} {r.rule_irrelevant:>7} "
              f"{r.llm_relevant:>4} {r.llm_irrelevant:>4} {r.llm_hold:>4} {r.to_label:>4} "
              f"{f'{r.labeled}/{LABEL_PER_TARGET}':>9} {r.chunked:>4}")
    print(f"대표 = 중복이 아닌 기사, 라벨 = 정답 라벨 수 / 대상당 목표 {LABEL_PER_TARGET}건 (최신 대표 기사부터)")


# ---------------------------------------------------------------------------
# label — 정답 라벨 붙이기
# ---------------------------------------------------------------------------

LABEL_KEYS = {"1": "관련", "2": "무관", "3": "보류"}
LABEL_SAVE_EVERY = 5     # 공용 DB가 멀어서(저장 1번 ≈ 1초) 5건마다 모아서 저장한다. 끝낼 때도 저장.
LABEL_GUIDE = """\
  1 관련  고른 대상이 기사의 주요 소재다
  2 무관  다른 의미의 기사이거나, 고른 대상이 스쳐 지나가듯 언급만 된다
  3 보류  기사만 보고는 판단하기 어렵다
  s 건너뛰기   b 이전 기사로   q 저장하고 끝내기"""


def cmd_label(query: str | None, limit: int | None = None) -> None:
    """대상마다 중복 제거 후 최신 대표 기사 limit건(기본 150)에 정답 라벨을 붙인다."""
    from qupict import db
    from qupict.pipeline import LABEL_PER_TARGET, articles_to_label, topic_status

    limit = LABEL_PER_TARGET if limit is None else limit
    topics = [t for t in topic_status()
              if t.to_label > t.labeled and t.labeled < limit and (query is None or t.query == query)]
    if not topics:
        print(f"라벨을 붙일 기사가 없습니다. 모든 대상이 {limit}건을 채웠거나, 아직 collect-all을 안 했습니다.")
        return
    for i, t in enumerate(topics, 1):
        print(f"  {i}. {t.query} → {t.sense_name}  (라벨 {t.labeled}/{limit}, 대표 기사 {t.to_label}건)")
    topic = topics[0] if len(topics) == 1 else topics[_choose_number("번호> ", len(topics))]
    record = db.get_topic(topic.query, topic.sense_name)

    # [2단계 중복 제거] 비슷한 기사는 대표 기사 하나만 라벨을 붙이면 된다 (임베딩만, LLM 판정 없음)
    from qupict.filtering import dedup_topic
    print("\n중복 기사 찾는 중 (임베딩만 씀, LLM 판정 없음)…")
    duplicates = asyncio.run(dedup_topic(topic.topic_id))
    print(f"  비슷한 기사 {len(duplicates)}건을 중복으로 표시했습니다 → 대표 기사만 보여 줍니다."
          if duplicates else "  새로 찾은 중복 기사는 없습니다.")
    remaining = limit - topic.labeled
    items = articles_to_label(topic.topic_id, limit=remaining)      # 최신 대표 기사부터 남은 개수만
    if not items:
        print("이 topic은 라벨을 붙일 대표 기사가 더 없습니다.")
        return
    print(f"  목표 {limit}건 중 {topic.labeled}건 완료 → 최신 대표 기사부터 {len(items)}건을 보여 줍니다."
          + (f" (대표 기사가 모자라 {remaining - len(items)}건 부족: collect-all로 더 수집하세요)"
             if len(items) < remaining else ""))

    sense = record.selected.sense
    print(f"\n[찾는 대상] {topic.query} → {sense.name}: {sense.definition}")
    print("※ 시스템 판정은 보여 주지 않습니다 (정답이 시스템 판정에 끌려가지 않게).")
    print(LABEL_GUIDE)

    pending: dict[str, str] = {}
    done: set[str] = set()                   # 이번에 라벨을 붙인 기사 (b로 돌아가 고쳐도 한 번만 셈)

    def flush():
        nonlocal pending
        if pending:
            db.save_labels(topic.topic_id, pending)
            pending = {}

    i = 0
    try:
        while i < len(items):
            a = items[i]
            date = a.pub_date.strftime("%Y-%m-%d") if a.pub_date else "날짜 없음"
            print("\n" + "-" * 70)
            print(f"[{i + 1}/{len(items)}] {a.title}")
            print(f"  {a.press or '언론사 모름'} · {date} · 찾은 검색어: {a.search_term}")
            if a.description:
                print(f"  요약: {a.description}")
            print(f"  본문: {a.content[:600]}{'…' if len(a.content) > 600 else ''}")
            key = input("라벨> ").strip().lower()
            if key == "q":
                break
            if key == "b":
                i = max(0, i - 1)
                continue
            if key == "s":
                i += 1
                continue
            if key not in LABEL_KEYS:
                print("1 / 2 / 3 / s / b / q 중에 입력하세요.")
                continue
            pending[a.link] = LABEL_KEYS[key]
            done.add(a.link)
            i += 1
            if len(pending) >= LABEL_SAVE_EVERY:
                flush()
    finally:
        flush()
    left = max(0, remaining - len(done))
    print(f"\n이번에 {len(done)}건 라벨을 저장했습니다. (목표 {limit}건까지 남은 라벨: {left}건"
          + (", 건너뛴 기사는 다음에 다시 나옵니다)" if left else ", 이 대상은 완료)"))


# ---------------------------------------------------------------------------
# 실험
# ---------------------------------------------------------------------------

def cmd_make_qa(query: str | None, suggest: bool) -> None:
    from experiments.qa_set import QA_PATH, make_qa
    from qupict import db
    from qupict.pipeline import topic_status

    topics = [t for t in topic_status() if t.llm_relevant and (query is None or t.query == query)]
    if not topics:
        print("'관련' 판정을 받은 기사가 있는 topic이 없습니다. 먼저 collect-all → filter-all을 실행하세요.")
        return
    for i, t in enumerate(topics, 1):
        print(f"  {i}. {t.query} → {t.sense_name}  (관련 기사 {t.llm_relevant}건)")
    topic = topics[0] if len(topics) == 1 else topics[_choose_number("번호> ", len(topics))]
    articles = db.get_topic_articles(topic.topic_id, statuses=("관련",))
    added = make_qa(topic.query, topic.sense_name, articles, suggest=suggest)
    print(f"\n질문 {added}개를 추가했습니다 → {QA_PATH}")


def cmd_sim(query: str | None, question: str | None) -> None:
    """유사도 확인. 질문이 있으면 그 질문 하나, 없으면 질문 세트 전체 (experiments/similarity.py)"""
    from experiments import similarity
    from qupict.pipeline import pick_targets, target_label

    if not question:
        similarity.main_qa(query)
        return
    if not query:
        print('질문 하나를 볼 때는 --target 이 필요합니다. 예: python run.py sim --target 강남 "질문"')
        return
    targets = pick_targets(query)
    if len(targets) > 1:                         # 눈·후드·마스크처럼 의미가 여러 개인 검색어
        for i, t in enumerate(targets, 1):
            print(f"  {i}. {target_label(t)}")
    target = targets[0] if len(targets) == 1 else targets[_choose_number("번호> ", len(targets))]
    similarity.main_question(target["query"], str(target["sense"]).strip(), question)


# ---------------------------------------------------------------------------
# 명령 해석
# ---------------------------------------------------------------------------

COMMANDS = {"ask", "collect-all", "filter-all", "index-all", "label", "status", "init-db",
            "make-qa", "exp-a", "exp-b", "exp-c", "sim"}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python run.py", description="QuPicT")
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="검색어 하나로 질문하기")
    ask.add_argument("words", nargs="*", help="검색어 [기간(일)]")
    ask.add_argument("--memory", action="store_true", help="DB 없이 메모리에서 (예전 방식)")
    ask.add_argument("--max", type=int, default=100, dest="max_articles", help="수집할 최대 기사 수 (기본 100)")

    c = sub.add_parser("collect-all", help="targets.yaml 대상 수집")
    c.add_argument("--target", help="이 검색어만 (예: 강남)")
    c.add_argument("--days", type=int, default=DEFAULT_PERIOD_DAYS, help="최근 며칠 (기본 30)")
    c.add_argument("--max", type=int, default=None, dest="max_articles",
                   help="대상당 최대 기사 수 (기본 300: 중복을 빼고 라벨 150건이 남게)")

    f = sub.add_parser("filter-all", help="targets.yaml 대상 필터링")
    f.add_argument("--target")
    f.add_argument("--redo", action="store_true", help="판정을 지우고 처음부터 다시 (정답 라벨은 유지)")

    ix = sub.add_parser("index-all", help="targets.yaml 대상 인덱싱")
    ix.add_argument("--target")
    ix.add_argument("--no-images", action="store_true", help="이미지 설명 없이 (LLM 비용 0)")

    lb = sub.add_parser("label", help="정답 라벨 붙이기")
    lb.add_argument("--target")
    lb.add_argument("--limit", type=int, help="대상당 라벨 목표 (기본 150, 최신 대표 기사부터)")

    sub.add_parser("status", help="topic별 진행 현황")

    ea = sub.add_parser("exp-a", help="실험 A: 필터링 방법 비교")
    ea.add_argument("--target")
    mq = sub.add_parser("make-qa", help="실험 B·C용 질문 세트 만들기")
    mq.add_argument("--target")
    mq.add_argument("--suggest", action="store_true", help="LLM이 질문·정답 초안을 써 줌")
    eb = sub.add_parser("exp-b", help="실험 B: 검색 방식 비교")
    eb.add_argument("--target")
    eb.add_argument("--faiss", action="store_true", help="예전 FAISS 검색도 비교")
    ec = sub.add_parser("exp-c", help="실험 C: 답변 신뢰성 비교")
    ec.add_argument("--target")
    ec.add_argument("--limit", type=int, help="질문 수 제한 (비용 조절)")
    sm = sub.add_parser("sim", help="유사도 확인 (기준값 없이)")
    sm.add_argument("question", nargs="?", help="볼 질문 (없으면 질문 세트 전체)")
    sm.add_argument("--target")
    sub.add_parser("init-db", help="DB 테이블 만들기")
    return p


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv = ["ask", *argv]                    # python run.py 애플 30 → ask 애플 30
    if not argv:
        argv = ["ask"]
    args = _parser().parse_args(argv)

    if args.command == "ask":
        cmd_ask(list(args.words), args.memory, args.max_articles)
    elif args.command == "collect-all":
        cmd_collect_all(args.target, args.days, args.max_articles)
    elif args.command == "filter-all":
        cmd_filter_all(args.target, args.redo)
    elif args.command == "index-all":
        cmd_index_all(args.target, not args.no_images)
    elif args.command == "label":
        cmd_label(args.target, args.limit)
    elif args.command == "status":
        cmd_status()
    elif args.command == "exp-a":
        from experiments.exp_a_filtering import main as exp_a
        exp_a(args.target)
    elif args.command == "make-qa":
        cmd_make_qa(args.target, args.suggest)
    elif args.command == "exp-b":
        from experiments.exp_b_retrieval import main as exp_b
        exp_b(args.target, args.faiss)
    elif args.command == "exp-c":
        from experiments.exp_c_answers import main as exp_c
        exp_c(args.target, args.limit)
    elif args.command == "sim":
        cmd_sim(args.target, args.question)
    elif args.command == "init-db":
        from qupict.db import init_db
        init_db()
        print("DB 테이블을 만들었습니다 (schema.sql).")


if __name__ == "__main__":
    main()
