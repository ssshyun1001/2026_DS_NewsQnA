"""
run.py 명령·라벨링·대상 처리 (가짜 DB)

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import builtins, sys
from datetime import datetime, timezone
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import run
from qupict import pipeline as P, db, sense
from qupict.schemas import Sense, SenseCandidates, SelectedSense

# 1) 명령 해석: "python run.py 애플 30" → ask, 각 명령 옵션
called = {}
run.cmd_ask = lambda words, memory, max_articles=100: called.__setitem__("ask", (words, memory, max_articles))
run.cmd_collect_all = lambda q, d, m: called.__setitem__("collect", (q, d, m))
run.cmd_filter_all = lambda q, redo: called.__setitem__("filter", (q, redo))
run.cmd_index_all = lambda q, img: called.__setitem__("index", (q, img))
run.cmd_label = lambda q: called.__setitem__("label", q)
run.main(["애플", "30"]); assert called["ask"] == (["애플", "30"], False, 100)
run.main(["ask", "애플", "--memory"]); assert called["ask"] == (["애플"], True, 100)
run.main(["강남", "30", "--max", "15"]); assert called["ask"] == (["강남", "30"], False, 15)
run.main(["collect-all", "--target", "강남", "--max", "15"]); assert called["collect"] == ("강남", 30, 15)
run.main(["filter-all", "--redo"]); assert called["filter"] == (None, True)
run.main(["index-all", "--no-images"]); assert called["index"] == (None, False)
run.main(["label", "--target", "눈"]); assert called["label"] == "눈"
print("1 ok: 명령 해석")

# 2) 대상별 실행: 하나가 실패해도 나머지는 계속
import importlib; importlib.reload(run)
P.pick_targets = lambda q=None: [{"query": "a", "sense": "x"}, {"query": "b", "sense": "y"}]
class R:
    def summary(self): return "ok"
async def step(t):
    if t["query"] == "a": raise RuntimeError("boom")
    return R()
out = []
builtins_print = builtins.print
builtins.print = lambda *a, **k: out.append(" ".join(map(str, a)))
run._for_each_target(None, "테스트", step)
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

# 4) 라벨링: 1/2/3/s/b/q, 5건마다 저장 + 끝낼 때 저장
importlib.reload(run)
from qupict import pipeline as P2
items = [P2.LabelItem(f"L{i}", f"제목{i}", "언론", datetime(2026, 11, 1, tzinfo=timezone.utc), "강남", "", "본문") for i in range(8)]
P2.topic_status = lambda: [P2.TopicStatus(3, "강남", "방송인 강남", 8, 0, 0, 0, 0, 0, 0, 0, 8, 0)]
P2.articles_to_label = lambda tid, include_labeled=False: items
class Rec2: selected = SelectedSense(query="강남", sense=Sense(name="방송인 강남", definition="M.I.B 출신"))
db.get_topic = lambda q, n: Rec2()
saves = []
db.save_labels = lambda tid, labels: saves.append(dict(labels))
keys = iter(["1", "2", "x", "b", "3", "s", "1", "1", "2", "q"])
builtins.input = lambda prompt="": next(keys)
out.clear(); builtins.print = lambda *a, **k: out.append(" ".join(map(str, a)))
run.cmd_label("강남")
builtins.print = builtins_print
# L0=관련 → L1=무관 → 잘못된 키 x(그대로) → b로 L1에 돌아가 3(보류)로 고침 → L2 건너뜀(s) → L3 관련, L4 관련, L5 무관(5건 → 저장) → q
assert saves == [{"L0": "관련", "L1": "보류", "L3": "관련", "L4": "관련", "L5": "무관"}], saves
assert not any("관련 기사" in o and "시스템" in o for o in out) and any("시스템 판정은 보여 주지 않습니다" in o for o in out)
print("4 ok: 라벨링 키·되돌리기·모아서 저장", saves)
print("ALL PASS")
