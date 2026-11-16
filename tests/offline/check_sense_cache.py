"""
0단계: 의미 후보 DB 캐싱 (가짜 DB)

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import dataclasses, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from qupict import sense, db
from qupict.config import settings
from qupict.schemas import Sense, SenseCandidates

llm_calls, saved = [], []
apple = SenseCandidates(query="애플", senses=[Sense(name="기업 Apple", definition="d", related_terms=["아이폰"])], model="m")
def fake_generate(q): llm_calls.append(q); return apple if q == "애플" else SenseCandidates(query=q, senses=[], model="m")
sense.generate_sense_candidates = fake_generate
db.save_sense_candidates = lambda c: saved.append(c.query)
store = {}
db.get_sense_candidates = lambda q: store.get(q)
def reset(): llm_calls.clear(); saved.clear()

# 1) DB 주소 없음 → LLM만, DB 안 씀
sense.settings = dataclasses.replace(settings, firestore_project="")
c, src = sense.load_sense_candidates("애플")
assert src == "LLM (DB 없음)" and llm_calls == ["애플"] and saved == []

sense.settings = dataclasses.replace(settings, firestore_project="fake-project")
# 2) DB에 없음 → LLM으로 만들고 저장
reset(); c, src = sense.load_sense_candidates(" 애플 ")
assert src == "LLM → DB 저장" and llm_calls == ["애플"] and saved == ["애플"]
# 3) DB에 있음 → LLM 안 부름
store["애플"] = apple
reset(); c, src = sense.load_sense_candidates("애플")
assert src == "DB" and llm_calls == [] and saved == [] and c.senses[0].name == "기업 Apple"
# 4) LLM이 후보를 못 만들면 저장 안 함 (다음에 다시 시도)
reset(); c, src = sense.load_sense_candidates("베라")
assert c.senses == [] and saved == [] and "저장 안 함" in src
# 5) DB 읽기 실패 → LLM으로 계속
def boom(*a): raise ConnectionError("down")
db.get_sense_candidates = boom
reset(); c, src = sense.load_sense_candidates("애플")
assert src == "LLM (DB 오류)" and llm_calls == ["애플"] and c.senses
# 6) DB 저장 실패 → 후보는 그대로 돌려줌
db.get_sense_candidates = lambda q: None; db.save_sense_candidates = boom
reset(); c, src = sense.load_sense_candidates("애플")
assert src == "LLM (DB 저장 실패)" and c.senses

# 7) 직접 입력으로 저장된 의미도 후보로 나온다 (많이 고른 순, 최대 MAX_SAVED_CUSTOM개) → LLM 안 부름
db.save_sense_candidates = lambda c: saved.append(c.query)
cust = lambda n: Sense(name=n, definition="d", related_terms=["t"], is_custom=True)
with_custom = SenseCandidates(query="애플", model="m",
                              senses=[cust("애플 주식"), apple.senses[0], *[cust(f"c{i}") for i in range(4)]],
                              custom_includes={"애플 주식": ["기업 Apple"]}, use_counts={"애플 주식": 5})
db.get_sense_candidates = lambda q: with_custom
reset(); c, src = sense.load_sense_candidates("애플")
assert llm_calls == [] and src == "DB (직접 입력 3개 포함)"
assert [s.name for s in c.senses] == ["애플 주식", "기업 Apple", "c0", "c1"]      # 직접 입력은 3개까지
# 8) 직접 입력만 저장된 검색어(예전에 LLM이 후보를 못 만든 경우) → LLM을 다시 시도하고 직접 입력은 뒤에 붙인다
only_custom = SenseCandidates(query="베라", senses=[cust("배우 베라")], custom_includes={"배우 베라": []})
db.get_sense_candidates = lambda q: only_custom
reset(); c, src = sense.load_sense_candidates("베라")
assert llm_calls == ["베라"] and [s.name for s in c.senses] == ["배우 베라"] and "직접 입력 1개" in src

# 9) 고른 의미 기록: 횟수 +1, 직접 입력이면 포함 후보(다른 의미로 두지 않은 LLM 후보)를 같이 저장
recorded = []
db.record_sense_use = lambda q, s, inc: recorded.append((q, s.name, list(inc)))
two = SenseCandidates(query="애플", senses=[apple.senses[0], Sense(name="과일 사과", definition="d", related_terms=["농가"])])
sense.remember_selection(two, sense.select_sense(two, 1))
my = sense.select_custom_sense(two, cust("애플 주식"), includes=["기업 Apple"])
sense.remember_selection(two, my)
assert recorded == [("애플", "과일 사과", []), ("애플", "애플 주식", ["기업 Apple"])]
sense.settings = dataclasses.replace(settings, firestore_project="")
recorded.clear(); sense.remember_selection(two, my); assert recorded == []     # DB 없으면 기록 안 함
print("ok: DB 캐싱 9가지 (DB 없음 / 저장 / 캐시 사용 / 빈 결과 미저장 / 읽기 실패 / 저장 실패 / 직접 입력 후보 / 직접 입력만 / 선택 기록)")
