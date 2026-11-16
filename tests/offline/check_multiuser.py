"""
여러 사용자가 동시에 쓸 때 — 공유 자원이 안전하게 나뉘는지 (LLM · Kiwi를 실제로 부르지 않는다, 크레딧 0)

  1) Kiwi: 여러 스레드가 동시에 처음 불러도 딱 한 번만 만든다 (약 500MB라 여러 번 만들면 메모리가 넘친다)
  2) 임베딩 클라이언트: 이벤트 루프마다 따로, 같은 루프 안에서는 하나, 루프 밖(동기)에서는 서버 전체가 하나
     (다른 루프의 비동기 연결을 같이 쓰면 "Connection error."가 난다)
  3) llm_slot: 여러 루프(사용자 작업)가 동시에 요청해도 서버 전체 동시 요청 수가 상한을 넘지 않는다
실행: python tests/run_offline.py
"""
import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def check_kiwi_once() -> None:
    import kiwipiepy
    from qupict import filtering

    made = []

    class SlowKiwi:                                   # 만드는 데 시간이 걸리는 가짜 Kiwi
        def __init__(self):
            made.append(self)
            time.sleep(0.2)

    real = kiwipiepy.Kiwi
    kiwipiepy.Kiwi = SlowKiwi
    filtering._kiwi = None
    try:
        got = []
        threads = [threading.Thread(target=lambda: got.append(filtering.get_kiwi())) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(made) == 1, len(made)                              # 5명이 동시에 불러도 한 번만 만듦
        assert all(k is made[0] for k in got)                         # 모두 같은 것을 씀
        filtering.warm_up_kiwi().join(2)                              # 이미 있으면 다시 만들지 않음
        assert len(made) == 1
    finally:
        kiwipiepy.Kiwi = real
        filtering._kiwi = None
    print("1 ok: Kiwi는 동시에 불러도 한 번만 만든다 (warm_up_kiwi 포함)")


def check_embeddings_per_loop() -> None:
    from qupict import llm

    llm._new_embeddings = lambda: object()            # 실제 클라이언트 대신 가짜 (만들어진 것끼리 구별만 하면 됨)
    llm._embeddings, llm._embeddings_by_loop = None, type(llm._embeddings_by_loop)()

    async def two_in_one_loop():
        return llm.get_embeddings(), llm.get_embeddings()

    results = []
    threads = [threading.Thread(target=lambda: results.append(asyncio.run(two_in_one_loop()))) for _ in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert all(a is b for a, b in results)                            # 같은 루프 안에서는 하나를 공유
    assert len({id(a) for a, _ in results}) == 3                      # 루프(작업)마다 다른 클라이언트
    sync1, sync2 = llm.get_embeddings(), llm.get_embeddings()         # 루프 밖(동기 호출)
    assert sync1 is sync2 and all(sync1 is not a for a, _ in results)
    print("2 ok: 임베딩 클라이언트는 이벤트 루프마다 따로, 루프 밖에서는 하나")


def check_llm_slot_limit() -> None:
    from qupict import llm

    limit = 3
    llm._llm_slots = threading.BoundedSemaphore(limit)
    active = {"now": 0, "max": 0}
    counter_lock = threading.Lock()

    async def request():
        async with llm.llm_slot():
            with counter_lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            await asyncio.sleep(0.05)                 # 요청이 오가는 동안
            with counter_lock:
                active["now"] -= 1

    async def one_user():
        await asyncio.gather(*(request() for _ in range(5)))

    start = time.perf_counter()
    threads = [threading.Thread(target=lambda: asyncio.run(one_user())) for _ in range(3)]   # 사용자 3명 × 요청 5개
    for t in threads: t.start()
    for t in threads: t.join()
    assert active["max"] == limit, active                            # 서버 전체 동시 요청은 상한까지만
    assert time.perf_counter() - start < 2                            # 차례를 기다릴 뿐 멈추지 않음
    llm._llm_slots = threading.BoundedSemaphore(llm.LLM_GLOBAL_CONCURRENCY)
    print(f"3 ok: llm_slot: 3명 x 5요청이 동시에 와도 동시 요청은 {limit}개까지")


def main() -> None:
    check_kiwi_once()
    check_embeddings_per_loop()
    check_llm_slot_limit()
    print("ALL PASS")


if __name__ == "__main__":
    main()
