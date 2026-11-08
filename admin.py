"""
QuPicT 서비스 DB 관리 실행 파일 — 서비스 DB를 미리 채우고 상태를 보는 명령

왜 필요한가
    python run.py 강남 30 은 질문할 때 수집 → 필터 → 인덱싱을 그 자리에서 해서 몇 분씩 걸린다.
    targets.yaml의 대상(12개)을 미리 한꺼번에 처리해 서비스 DB에 쌓아 두면 바로 질문할 수 있다.

    python admin.py collect-all [--days 30] [--max 300]   0·1·2단계: 의미 선택 → 뉴스 수집 → 중복 제거 → DB 저장
    python admin.py filter-all  [--redo]                  2·3단계: 규칙 → LLM 판정 → DB 저장
    python admin.py index-all   [--no-images]             4단계: 이미지 설명 → 청크 → 임베딩 → DB 저장
    python admin.py status                                topic별 진행 현황
    python admin.py init-db                               DB 테이블 만들기 (schema.sql, 처음 한 번)

    --target 강남 처럼 검색어 하나만 할 수도 있다.
    모든 일괄 명령은 "아직 안 한 것만" 처리하므로 여러 번 실행해도 안전하고 비용이 다시 들지 않는다.

다른 용도의 실행 파일
    python run.py ...         서비스: 검색어로 질문하기
    python evaluate.py ...    평가: 정답 라벨, 질문 세트, 실험 A·B·C, 유사도 확인
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from qupict.cli import setup_console
from qupict.config import validate_settings

setup_console()

DEFAULT_PERIOD_DAYS = 30


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
        print("아직 DB에 topic이 없습니다. python admin.py collect-all 또는 python run.py로 수집하세요.")
        return
    print(f"{'id':>3} {'검색어 → 의미':<28} {'기사':>4} {'중복':>4} {'미판정':>5} {'규칙무관':>7} "
          f"{'관련':>4} {'무관':>4} {'보류':>4} {'대표':>4} {'라벨':>9} {'청크':>4}")
    for r in rows:
        name = f"{r.query} → {r.sense_name}"[:28]
        print(f"{r.topic_id:>3} {name:<28} {r.articles:>4} {r.duplicates:>4} {r.unjudged:>5} {r.rule_irrelevant:>7} "
              f"{r.llm_relevant:>4} {r.llm_irrelevant:>4} {r.llm_hold:>4} {r.to_label:>4} "
              f"{f'{r.labeled}/{LABEL_PER_TARGET}':>9} {r.chunked:>4}")
    print(f"대표 = 중복이 아닌 기사, 라벨 = 정답 라벨 수 / 대상당 목표 {LABEL_PER_TARGET}건 (최신 대표 기사부터)")


def cmd_init_db() -> None:
    from qupict.db import init_db
    init_db()
    print("DB 테이블을 만들었습니다 (schema.sql).")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python admin.py", description="QuPicT 서비스 DB 관리")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect-all", help="targets.yaml 대상 수집 + 중복 제거")
    c.add_argument("--target", help="이 검색어만 (예: 강남)")
    c.add_argument("--days", type=int, default=DEFAULT_PERIOD_DAYS, help="최근 며칠 (기본 30)")
    c.add_argument("--max", type=int, default=None, dest="max_articles",
                   help="대상당 최대 기사 수 (기본 300: 중복을 빼고 대표 기사 150건이 남게)")

    f = sub.add_parser("filter-all", help="targets.yaml 대상 필터링")
    f.add_argument("--target")
    f.add_argument("--redo", action="store_true", help="판정을 지우고 처음부터 다시 (정답 라벨은 유지)")

    ix = sub.add_parser("index-all", help="targets.yaml 대상 인덱싱")
    ix.add_argument("--target")
    ix.add_argument("--no-images", action="store_true", help="이미지 설명 없이 (LLM 비용 0)")

    sub.add_parser("status", help="topic별 진행 현황")
    sub.add_parser("init-db", help="DB 테이블 만들기")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "collect-all":
        cmd_collect_all(args.target, args.days, args.max_articles)
    elif args.command == "filter-all":
        cmd_filter_all(args.target, args.redo)
    elif args.command == "index-all":
        cmd_index_all(args.target, not args.no_images)
    elif args.command == "status":
        cmd_status()
    elif args.command == "init-db":
        cmd_init_db()


if __name__ == "__main__":
    main()
