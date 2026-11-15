"""
단계별 시간 측정 (qupict/timing.py) — 기록·들여쓰기·요약표, 꺼져 있을 때 아무것도 안 함, run.py --timing

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import asyncio, contextlib, io, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from qupict import timing

# 1) 꺼져 있으면(기본) 기록도 출력도 없다. t.note = ... 는 그대로 써도 에러가 나지 않는다
timing.enable(False); timing.reset()
with timing.timed("꺼짐") as t:
    t.note = "무시됨"
timing.add("꺼짐2", 1.0)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    timing.report("x")
assert timing.records() == [] and buf.getvalue() == ""
print("1 ok: 꺼져 있으면 기록·출력 없음")

# 2) 켜면: 안쪽 timed는 depth 1, add는 지금 단계 안쪽, 비율은 맨 바깥 단계 합 기준
timing.enable(True); timing.reset()

async def flow():
    with timing.timed("1단계") as outer:
        with timing.timed("크롤링") as t:
            await asyncio.sleep(0.05)
            t.note = "성공 3 / 시도 4건"
        timing.add("청크 임베딩", 0.02, "요청 2번")
        outer.note = "끝"
    with timing.timed("2단계"):
        time.sleep(0.05)

asyncio.run(flow())
recs = timing.records()
assert [(r.label, r.depth) for r in recs] == [("1단계", 0), ("크롤링", 1), ("청크 임베딩", 1), ("2단계", 0)], recs
assert recs[0].seconds >= recs[1].seconds >= 0.04 and recs[1].note == "성공 3 / 시도 4건"
text = timing.format_report("테스트")
total = recs[0].seconds + recs[3].seconds
assert f"총 {total:.2f}초" in text and "    크롤링" in text and "(성공 3 / 시도 4건)" in text, text
print("2 ok: 들여쓰기·메모·총 시간")

# 3) with 블록 안에서 예외가 나도 시간은 기록되고 depth가 원래대로 돌아온다
timing.reset()
try:
    with timing.timed("실패 단계"):
        raise ValueError("boom")
except ValueError:
    pass
with timing.timed("다음 단계"):
    pass
assert [(r.label, r.depth) for r in timing.records()] == [("실패 단계", 0), ("다음 단계", 0)]
print("3 ok: 예외 후에도 depth 복구")

# 3-1) 기록은 스레드마다 따로 (웹에서 준비·질문·이미지 글자 읽기가 동시에 돌아도 표가 섞이지 않게)
import threading
timing.reset()
with timing.timed("이 스레드"):
    pass
other = {}
def work():
    timing.reset()
    with timing.timed("다른 스레드"):
        pass
    other["labels"] = [r.label for r in timing.records()]
t = threading.Thread(target=work); t.start(); t.join()
assert [r.label for r in timing.records()] == ["이 스레드"] and other["labels"] == ["다른 스레드"]
print("3-1 ok: 스레드마다 따로 기록")

# 4) run.py --timing 이 측정을 켠다 (서비스 자체는 가짜로 바꿔 실행하지 않음)
timing.enable(False)
import run
called = {}
run.cmd_ask = lambda words, memory, max_articles=100: called.__setitem__("ask", (words, timing.enabled))
run.main(["강남", "30"]); assert called["ask"] == (["강남", "30"], False)
run.main(["강남", "30", "--timing"]); assert called["ask"] == (["강남", "30"], True)
print("4 ok: run.py --timing")
timing.enable(False); timing.reset()
