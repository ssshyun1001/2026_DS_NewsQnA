"""
QuPicT 평가 실행 파일 — 정답 라벨, 질문 세트, 실험 A·B·C (기획안 7단계)

    python evaluate.py label   [--target 강남] [--limit 150]  최신 대표 기사 150건에 정답 라벨 (관련 / 무관 / 보류)
    python evaluate.py make-qa [--target 강남] [--suggest]    실험 B·C용 질문 세트 만들기 (experiments/qa_set.yaml)
    python evaluate.py exp-a   [--target 강남]                실험 A: 필터링 방법 비교 (정답 JSON evalset/labeled_set_*.json 기준)
    python evaluate.py exp-b   [--target 강남] [--faiss]      실험 B: 검색 방식 비교 (Recall@5, MRR)
    python evaluate.py exp-c   [--target 강남] [--limit 10]   실험 C: LLM 단독 vs 기본 RAG vs QuPicT 답변 채점
    python evaluate.py sim     [--target 강남] ["질문"]       유사도 확인: 기준값 없이 청크 유사도 보기
                                                              (질문 없이 → 질문 세트로 기준값 0.40/0.45/0.50 비교)

    결과는 experiments/results/에 저장된다. 실험 코드는 experiments/ 폴더에 있다.

※ 지금은 평가도 서비스 DB를 읽는다 (라벨은 DB의 article_topics.label에 저장).
  팀에서 "평가는 정답 JSON만 쓴다"로 정해서, 라벨·실험이 JSON 평가셋을 쓰도록 바꿀 예정이다.
  그때 이 파일의 명령 이름은 그대로 두고 안쪽만 바뀐다.

다른 용도의 실행 파일
    python run.py ...       서비스: 검색어로 질문하기
    python admin.py ...     서비스 DB 관리: 미리 수집·판정·인덱싱, 현황
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from qupict.cli import choose_number, setup_console

setup_console()


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
        print(f"라벨을 붙일 기사가 없습니다. 모든 대상이 {limit}건을 채웠거나, 아직 python admin.py collect로 대상을 추가하지 않았습니다.")
        return
    for i, t in enumerate(topics, 1):
        print(f"  {i}. {t.query} → {t.sense_name}  (라벨 {t.labeled}/{limit}, 대표 기사 {t.to_label}건)")
    topic = topics[0] if len(topics) == 1 else topics[choose_number("번호> ", len(topics))]
    record = db.get_topic(topic.query, topic.sense_name)

    # [2단계 중복 제거] 비슷한 기사는 대표 기사 하나만 라벨을 붙이면 된다 (임베딩만, LLM 판정 없음)
    from qupict.filtering import dedup_topic
    print("\n중복 기사 찾는 중 (임베딩만 씀, LLM 판정 없음)…")
    duplicates = asyncio.run(dedup_topic(topic.topic_id, record.selected.match_terms))
    print(f"  비슷한 기사 {len(duplicates)}건을 중복으로 표시했습니다 → 대표 기사만 보여 줍니다."
          if duplicates else "  새로 찾은 중복 기사는 없습니다.")
    remaining = limit - topic.labeled
    items = articles_to_label(topic.topic_id, limit=remaining)      # 최신 대표 기사부터 남은 개수만
    if not items:
        print("이 topic은 라벨을 붙일 대표 기사가 더 없습니다.")
        return
    print(f"  목표 {limit}건 중 {topic.labeled}건 완료 → 최신 대표 기사부터 {len(items)}건을 보여 줍니다."
          + (f" (대표 기사가 모자라 {remaining - len(items)}건 부족: python admin.py collect-all로 더 수집하세요)"
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
# 질문 세트 · 유사도 확인
# ---------------------------------------------------------------------------

def cmd_make_qa(query: str | None, suggest: bool) -> None:
    from experiments.qa_set import QA_PATH, make_qa
    from qupict import db
    from qupict.pipeline import topic_status

    topics = [t for t in topic_status() if t.llm_relevant and (query is None or t.query == query)]
    if not topics:
        print("'관련' 판정을 받은 기사가 있는 topic이 없습니다. 먼저 python admin.py collect-all → filter-all을 실행하세요.")
        return
    for i, t in enumerate(topics, 1):
        print(f"  {i}. {t.query} → {t.sense_name}  (관련 기사 {t.llm_relevant}건)")
    topic = topics[0] if len(topics) == 1 else topics[choose_number("번호> ", len(topics))]
    articles = db.get_topic_articles(topic.topic_id, statuses=("관련",))
    added = make_qa(topic.query, topic.sense_name, articles, suggest=suggest)
    print(f"\n질문 {added}개를 추가했습니다 → {QA_PATH}")


def cmd_sim(query: str | None, question: str | None) -> None:
    """유사도 확인. 질문이 있으면 그 질문 하나, 없으면 질문 세트 전체 (experiments/similarity.py)"""
    from experiments import similarity
    from qupict.pipeline import stored_topics, topic_label

    if not question:
        similarity.main_qa(query)
        return
    if not query:
        print('질문 하나를 볼 때는 --target 이 필요합니다. 예: python evaluate.py sim --target 강남 "질문"')
        return
    records = stored_topics(query)
    if len(records) > 1:                         # 눈·후드·마스크처럼 의미가 여러 개인 검색어
        for i, r in enumerate(records, 1):
            print(f"  {i}. {topic_label(r)}")
    record = records[0] if len(records) == 1 else records[choose_number("번호> ", len(records))]
    similarity.main_question(record.selected.query, record.selected.sense.name, question)


# ---------------------------------------------------------------------------
# 명령 해석
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python evaluate.py", description="QuPicT 평가 (정답 라벨 · 실험 A·B·C)")
    sub = p.add_subparsers(dest="command", required=True)

    lb = sub.add_parser("label", help="정답 라벨 붙이기")
    lb.add_argument("--target")
    lb.add_argument("--limit", type=int, help="대상당 라벨 목표 (기본 150, 최신 대표 기사부터)")

    mq = sub.add_parser("make-qa", help="실험 B·C용 질문 세트 만들기")
    mq.add_argument("--target")
    mq.add_argument("--suggest", action="store_true", help="LLM이 질문·정답 초안을 써 줌")

    ea = sub.add_parser("exp-a", help="실험 A: 필터링 방법 비교 (정답 JSON 기준)")
    ea.add_argument("--target")
    eb = sub.add_parser("exp-b", help="실험 B: 검색 방식 비교")
    eb.add_argument("--target")
    eb.add_argument("--faiss", action="store_true", help="예전 FAISS 검색도 비교")
    ec = sub.add_parser("exp-c", help="실험 C: 답변 신뢰성 비교")
    ec.add_argument("--target")
    ec.add_argument("--limit", type=int, help="질문 수 제한 (비용 조절)")

    sm = sub.add_parser("sim", help="유사도 확인 (기준값 없이)")
    sm.add_argument("question", nargs="?", help="볼 질문 (없으면 질문 세트 전체)")
    sm.add_argument("--target")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "label":
        cmd_label(args.target, args.limit)
    elif args.command == "make-qa":
        cmd_make_qa(args.target, args.suggest)
    elif args.command == "exp-a":
        from evalset.exp_a_filtering import main as exp_a
        exp_a(args.target)
    elif args.command == "exp-b":
        from experiments.exp_b_retrieval import main as exp_b
        exp_b(args.target, args.faiss)
    elif args.command == "exp-c":
        from experiments.exp_c_answers import main as exp_c
        exp_c(args.target, args.limit)
    elif args.command == "sim":
        cmd_sim(args.target, args.question)


if __name__ == "__main__":
    main()
