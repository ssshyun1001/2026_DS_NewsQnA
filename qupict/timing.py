"""
단계별 시간 측정 — 어느 단계가 느린지 숫자로 보기 위한 도구 (최적화 전에 먼저 재 보기)

    python run.py 강남 30 --timing        수집~인덱싱이 끝날 때, 질문마다 답변이 끝날 때 시간표를 보여 준다
    QUPICT_TIMING=1 python admin.py ...   다른 실행 파일에서는 환경변수로 켠다 (요약표는 각 실행 파일이 report를 불러야 나옴)

코드에서 쓰는 법

    from qupict import timing

    with timing.timed("본문 크롤링") as t:      # with 블록 안의 시간을 잰다 (await 기다린 시간 포함)
        articles = await crawl_articles(metas)
        t.note = f"{len(articles)}건"           # 끝난 뒤에야 아는 정보는 note에 적는다

    timing.add("청크 임베딩", 3.2, "기사 40건")   # 반복문 안에서 직접 잰 시간을 합쳐서 한 줄로 남길 때

    timing.reset()                              # 기록 비우기 (질문 하나, 수집 한 번 단위로)
    timing.report("질문 답변")                   # 요약표 출력

  - 꺼져 있으면(기본) 아무것도 기록·출력하지 않는다. 서비스 동작·속도에 영향 없음.
  - timed 안에서 다시 timed를 쓰면 요약표에서 들여쓰기로 보인다 (예: 1단계 수집 > 본문 크롤링).
  - 비율(%)은 맨 바깥 단계들의 합(= 총 시간) 기준이다.
  - asyncio.gather로 동시에 도는 작업 "안"에서는 쓰지 않는다 (들여쓰기 단계가 꼬인다). gather 바깥을 잰다.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

enabled: bool = os.getenv("QUPICT_TIMING", "") not in ("", "0")


@dataclass
class Record:
    label: str
    seconds: float = 0.0
    note: str = ""
    depth: int = 0          # 0 = 맨 바깥 단계, 1 = 그 안의 세부 단계 ...


# 기록은 스레드마다 따로 둔다: 웹 서버에서 준비 작업 · 질문 답변 · 백그라운드 이미지 글자 읽기가
# 서로 다른 스레드에서 동시에 돌아도 표가 섞이지 않게.
_local = threading.local()


def _state():
    if not hasattr(_local, "records"):
        _local.records, _local.depth = [], 0
    return _local


def enable(on: bool = True) -> None:
    global enabled
    enabled = on


def reset() -> None:
    s = _state()
    s.records.clear()
    s.depth = 0


@contextmanager
def timed(label: str, note: str = "") -> Iterator[Record]:
    """with 블록에 걸린 시간을 잰다. 꺼져 있어도 Record를 돌려주므로 t.note = ... 를 그대로 써도 된다."""
    s = _state()
    record = Record(label, note=note, depth=s.depth)
    if not enabled:
        yield record
        return
    s.records.append(record)         # 시작할 때 넣어야 바깥 단계가 안쪽 단계보다 위에 찍힌다
    s.depth += 1
    start = time.perf_counter()
    try:
        yield record
    finally:
        record.seconds = time.perf_counter() - start
        s.depth -= 1


def add(label: str, seconds: float, note: str = "") -> None:
    """직접 잰 시간을 한 줄로 남긴다 (반복문에서 여러 번 잰 시간을 합친 값 등). 지금 단계의 안쪽으로 들어간다."""
    if enabled:
        s = _state()
        s.records.append(Record(label, seconds, note, s.depth))


def records() -> list[Record]:
    return list(_state().records)


def format_report(title: str) -> str:
    recs = _state().records
    total = sum(r.seconds for r in recs if r.depth == 0)
    lines = [f"⏱ 시간 측정: {title} (총 {total:.2f}초)"]
    for r in recs:
        share = f"{r.seconds / total * 100:4.0f}%" if total > 0 else "    -"
        name = "  " * r.depth + r.label
        note = f"  ({r.note})" if r.note else ""
        lines.append(f"  {name:<28} {r.seconds:7.2f}초 {share}{note}")
    return "\n".join(lines)


def report(title: str) -> None:
    """요약표를 출력한다. 꺼져 있거나 기록이 없으면 아무것도 하지 않는다."""
    if enabled and _state().records:
        print("\n" + format_report(title))
