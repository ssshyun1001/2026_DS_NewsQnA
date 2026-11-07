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
sense.settings = dataclasses.replace(settings, database_url="")
c, src = sense.load_sense_candidates("애플")
assert src == "LLM (DB 없음)" and llm_calls == ["애플"] and saved == []

sense.settings = dataclasses.replace(settings, database_url="postgresql://fake")
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
print("ok: DB 캐싱 6가지 (DB 없음 / 저장 / 캐시 사용 / 빈 결과 미저장 / 읽기 실패 / 저장 실패)")
