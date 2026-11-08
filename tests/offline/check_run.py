"""
실행 파일 3개: run.py(서비스) · admin.py(서비스 DB 관리) · evaluate.py(평가) — 명령·라벨링·대상 처리 (가짜 DB)

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import builtins, sys
from datetime import datetime, timezone
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import admin, evaluate, run
from qupict import pipeline as P, db, sense
from qupict.schemas import Sense, SenseCandidates, SelectedSense

# 1) 명령 해석: 서비스(run.py) / 서비스 DB 관리(admin.py) / 평가(evaluate.py)
called = {}
run.cmd_ask = lambda words, memory, max_articles=100: called.__setitem__("ask", (words, memory, max_articles))
admin.cmd_collect_all = lambda q, d, m: called.__setitem__("collect", (q, d, m))
admin.cmd_filter_all = lambda q, redo: called.__setitem__("filter", (q, redo))
admin.cmd_index_all = lambda q, img: called.__setitem__("index", (q, img))
admin.cmd_status = lambda: called.__setitem__("status", True)
evaluate.cmd_label = lambda q, limit=None: called.__setitem__("label", (q, limit))
evaluate.cmd_make_qa = lambda q, suggest: called.__setitem__("make-qa", (q, suggest))
evaluate.cmd_sim = lambda q, question: called.__setitem__("sim", (q, question))
run.main(["애플", "30"]); assert called["ask"] == (["애플", "30"], False, 100)
run.main(["ask", "애플", "--memory"]); assert called["ask"] == (["애플"], True, 100)    # 예전 "ask" 단어도 그대로
run.main(["강남", "30", "--max", "15"]); assert called["ask"] == (["강남", "30"], False, 15)
admin.main(["collect-all", "--target", "강남", "--max", "15"]); assert called["collect"] == ("강남", 30, 15)
admin.main(["filter-all", "--redo"]); assert called["filter"] == (None, True)
admin.main(["index-all", "--no-images"]); assert called["index"] == (None, False)
admin.main(["status"]); assert called["status"]
admin.main(["collect-all"]); assert called["collect"] == (None, 30, None)             # 기본 300은 cmd_collect_all 안에서
evaluate.main(["label", "--target", "눈"]); assert called["label"] == ("눈", None)   # 기본 150은 cmd_label 안에서
evaluate.main(["label", "--limit", "20"]); assert called["label"] == (None, 20)
evaluate.main(["make-qa", "--suggest"]); assert called["make-qa"] == (None, True)
evaluate.main(["sim", "--target", "강남", "질문"]); assert called["sim"] == ("강남", "질문")
print("1 ok: 명령 해석 (실행 파일 3개)")

# 1-1) 서비스 실행 파일에서 옮긴 명령을 쓰면 새 실행 파일을 알려 주고 끝낸다 (서비스가 실행되지 않음)
import contextlib, io
called.clear()
for old_cmd, new_file in [("collect-all", "admin.py"), ("status", "admin.py"), ("label", "evaluate.py"), ("exp-a", "evaluate.py")]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            run.main([old_cmd, "--target", "강남"])
            raise AssertionError("끝나지 않음")
        except SystemExit as e:
            assert e.code == 2
    assert f"python {new_file} {old_cmd} --target 강남" in buf.getvalue(), buf.getvalue()
assert "ask" not in called
print("1-1 ok: 옮긴 명령 안내")

# 2) 대상별 실행 (admin.py): 하나가 실패해도 나머지는 계속
import importlib; importlib.reload(admin)
P.pick_targets = lambda q=None: [{"query": "a", "sense": "x"}, {"query": "b", "sense": "y"}]
class R:
    def summary(self): return "ok"
async def step(t):
    if t["query"] == "a": raise RuntimeError("boom")
    return R()
out = []
builtins_print = builtins.print
builtins.print = lambda *a, **k: out.append(" ".join(map(str, a)))
admin._for_each_target(None, "테스트", step)
builtins.print = builtins_print
assert any("a → x → 실패: RuntimeError" in o for o in out) and any("b → y → ok" in o for o in out) and any("실패 1개" in o for o in out)
print("2 ok: 대상 하나 실패해도 계속")

# 3) resolve_target: 직접 입력 의미는 DB에 있으면 LLM 안 부름, 이름은 targets.yaml 그대로
cands = SenseCandidates(query="베라", senses=[Sense(name="음료 베라", definition="d", related_terms=["음료"])])
P.load_sense_candidates = lambda q: (cands, "DB")
llm_calls = []
sense.build_custom_sense = lambda q, d: (llm_calls.append(d), Sense(name="LLM이 붙인 이름", definition=d, related_terms=["아이스크림"]))[1]
saved = []
db.save_topic = lambda sel: (saved.append(sel), 42)[1]
db.get_topic = lambda q, name: None
tid, sel = P.resolve_target({"query": "베라", "sense": "배스킨라빈스", "description": "아이스크림", "aliases": ["배라"]})
assert tid == 42 and sel.sense.name == "배스킨라빈스" and sel.sense.is_custom and sel.sense.aliases == ["배라"] and llm_calls == ["아이스크림"]
class Rec: selected = SelectedSense(query="베라", sense=Sense(name="배스킨라빈스", definition="저장된 정의", related_terms=["아이스크림"], is_custom=True))
db.get_topic = lambda q, name: Rec() if name == "배스킨라빈스" else None
llm_calls.clear()
tid, sel = P.resolve_target({"query": "베라", "sense": "배스킨라빈스", "aliases": ["배라"]})
assert llm_calls == [] and sel.sense.definition == "저장된 정의" and sel.sense.aliases == ["배라"]
assert sel.excluded_terms == ["음료"]
print("3 ok: 직접 입력 의미 재사용 (LLM 0번, 이름 고정, 사람 값 반영)")

# 4) 라벨링 (evaluate.py): 1/2/3/s/b/q, 5건마다 저장 + 끝낼 때 저장
importlib.reload(evaluate)
from qupict import pipeline as P2
items = [P2.LabelItem(f"L{i}", f"제목{i}", "언론", datetime(2026, 11, 1, tzinfo=timezone.utc), "강남", "", "본문") for i in range(8)]
P2.topic_status = lambda: [P2.TopicStatus(3, "강남", "방송인 강남", 8, 0, 0, 0, 0, 0, 0, 0, 8, 0)]
label_limits = []
def fake_to_label(tid, include_labeled=False, limit=None):
    label_limits.append(limit)
    return items[:limit] if limit is not None else items
P2.articles_to_label = fake_to_label
class Rec2: selected = SelectedSense(query="강남", sense=Sense(name="방송인 강남", definition="M.I.B 출신"))
db.get_topic = lambda q, n: Rec2()
saves = []
db.save_labels = lambda tid, labels: saves.append(dict(labels))
import qupict.filtering as F2
dedup_calls = []
async def fake_dedup(tid):
    dedup_calls.append(tid)
    return {"D1": "L0", "D2": "L3"}
F2.dedup_topic = fake_dedup
keys = iter(["1", "2", "x", "b", "3", "s", "1", "1", "2", "q"])
builtins.input = lambda prompt="": next(keys)
out.clear(); builtins.print = lambda *a, **k: out.append(" ".join(map(str, a)))
evaluate.cmd_label("강남")
builtins.print = builtins_print
assert dedup_calls == [3] and any("2건을 중복으로 표시" in o for o in out)   # 라벨 전에 중복 제거부터
# L0=관련 → L1=무관 → 잘못된 키 x(그대로) → b로 L1에 돌아가 3(보류)로 고침 → L2 건너뜀(s) → L3 관련, L4 관련, L5 무관(5건 → 저장) → q
assert saves == [{"L0": "관련", "L1": "보류", "L3": "관련", "L4": "관련", "L5": "무관"}], saves
assert not any("관련 기사" in o and "시스템" in o for o in out) and any("시스템 판정은 보여 주지 않습니다" in o for o in out)
assert label_limits == [150] and any("남은 라벨: 145건" in o for o in out)      # 기본 목표 150건, 5건 저장
print("4 ok: 라벨링 키·되돌리기·모아서 저장", saves)

# 5) 라벨 목표: 이미 붙인 라벨을 빼고 최신 대표 기사부터 남은 개수만, 다 채운 대상은 목록에서 빠짐
label_limits.clear(); saves.clear(); out.clear()
P2.topic_status = lambda: [P2.TopicStatus(3, "강남", "방송인 강남", 8, 0, 0, 0, 0, 0, 0, 6, 8, 0)]   # 라벨 6건
keys = iter(["1", "2", "1", "1"])
builtins.input = lambda prompt="": next(keys)
builtins.print = lambda *a, **k: out.append(" ".join(map(str, a)))
evaluate.cmd_label("강남", limit=10)
builtins.print = builtins_print
assert label_limits == [4] and len(saves[0]) == 4                   # 10 - 6 = 4건만 보여 주고 저장
assert any("이 대상은 완료" in o for o in out)
P2.topic_status = lambda: [P2.TopicStatus(3, "강남", "방송인 강남", 8, 0, 0, 0, 0, 0, 0, 10, 8, 0)]
out.clear(); builtins.print = lambda *a, **k: out.append(" ".join(map(str, a)))
evaluate.cmd_label("강남", limit=10)
builtins.print = builtins_print
assert any("라벨을 붙일 기사가 없습니다" in o for o in out)         # 목표를 채운 대상은 나오지 않음
print("5 ok: 대상당 라벨 목표 (최신 대표 기사부터 남은 개수만)")

# 6) collect-all 한 대상: 수집 → 중복 제거 → 대표 기사 수, 150건 미만이면 알려 줌
import asyncio
from qupict.collect import CollectResult
P3 = importlib.reload(P2)
P3.resolve_target = lambda t: (3, None)
async def fake_collect(sel, days, mx): return CollectResult(3, searched=mx, reused=0, crawled=mx, crawl_failed=0, images=0)
async def fake_dedup2(tid): return {"x": "y", "z": "y"}
P3.collect_to_db, P3.dedup_topic, P3.count_representatives = fake_collect, fake_dedup2, lambda tid: 120
r = asyncio.run(P3.collect_target({"query": "강남", "sense": "방송인 강남"}))
assert r.collected.searched == 300 and r.duplicates == 2 and r.representatives == 120
assert "대표 기사 120건" in r.summary() and "30건 적음" in r.summary() and "--max를 늘려" in r.summary()
async def fake_collect_few(sel, days, mx): return CollectResult(3, searched=200, reused=0, crawled=190, crawl_failed=10, images=0)
P3.collect_to_db = fake_collect_few
assert "--days를 늘려" in asyncio.run(P3.collect_target({"query": "강남", "sense": "방송인 강남"})).summary()
P3.count_representatives = lambda tid: 160
assert "적음" not in asyncio.run(P3.collect_target({"query": "강남", "sense": "방송인 강남"})).summary()
print("6 ok: collect-all 수집 → 중복 제거 → 대표 기사 수")
print("ALL PASS")
