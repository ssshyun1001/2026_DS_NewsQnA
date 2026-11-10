"""
동시 실행 도우미 — 서로 기다릴 필요가 없는 일을 한꺼번에 돌려 기다리는 시간을 줄인다

왜 필요한가 (experiments/benchmark.py로 잰 값, 2026-11-10)
  팀 공용 DB(Supabase, 싱가포르)는 새 연결 한 번에 약 1.4~1.6초, 열린 연결에서 쿼리 한 번에 약 80ms가 걸린다.
  db.connect()는 이제 커넥션 풀(db.py)에서 연결을 빌려 쓰므로 연결 시간은 처음 한 번뿐이지만,
  DB 함수 하나는 여전히 0.2~0.3초(왕복 몇 번)가 걸린다. 서로 상관없는 DB 함수는 동시에 부르면 그 시간도 겹친다.

  run_parallel(f, g, ...)          인자 없는 함수들을 동시에 실행, 결과는 넣은 순서대로
  map_parallel(fn, items)          items 각각에 fn을 동시에 실행, 결과는 items 순서대로
  timed_call(fn, *args)            (결과, 걸린 초) — 동시에 돌린 일의 시간을 각각 남길 때

  - 결과 순서는 항상 넣은 순서와 같다 → 차례로 실행했을 때와 결과가 똑같다.
  - 하나가 예외를 내면 그 예외를 그대로 올린다 (차례로 실행했을 때처럼). 다른 일은 끝까지 돈다.
  - DB_PARALLEL: DB에 동시에 여는 연결 수 상한. Supabase 무료 플랜 세션 풀러는 동시 연결 수가
    적어서(약 15) 넉넉히 낮게 둔다.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

DB_PARALLEL = 4


def run_parallel(*calls: Callable[[], Any], max_workers: int | None = None) -> list[Any]:
    """인자 없는 함수들을 스레드로 동시에 실행하고 결과를 넣은 순서대로 돌려준다."""
    if len(calls) <= 1:
        return [c() for c in calls]
    with ThreadPoolExecutor(max_workers=max_workers or len(calls)) as pool:
        futures = [pool.submit(c) for c in calls]
        return [f.result() for f in futures]


def map_parallel(fn: Callable[[T], R], items: Iterable[T], max_workers: int = DB_PARALLEL) -> list[R]:
    """items 각각에 fn을 동시에(최대 max_workers개) 실행하고 결과를 items 순서대로 돌려준다."""
    items = list(items)
    if len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(fn, items))


def timed_call(fn: Callable[..., R], *args, **kwargs) -> tuple[R, float]:
    """fn(*args, **kwargs)의 (결과, 걸린 초)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start
